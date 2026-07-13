"""MVP safety head implementations."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from evoguard.types import DefenseDecision, SafetyAction, ToolCall, TrajectoryRecord, TrajectoryType


RISK_KEYWORDS = (
    "ignore previous",
    "override",
    "secret",
    "credential",
    "password",
    "transfer",
    "exfiltrate",
    "send the contents",
    "delete",
    "untrusted",
)

RISKY_TOOLS = frozenset(
    {
        "email_sender",
        "file_writer",
        "web_request",
        "bank_transfer_simulator",
        "smart_home_controller",
        "ticket_updater",
    }
)

ACTION_ORDER = (
    SafetyAction.EXECUTE,
    SafetyAction.ASK_CONFIRMATION,
    SafetyAction.REFUSE,
)


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 0.05
    clip_epsilon: float = 0.2
    epochs: int = 4
    entropy_coef: float = 0.01
    advantage_epsilon: float = 1e-8


@dataclass(frozen=True)
class PPOUpdateStats:
    steps: int
    mean_reward: float
    mean_advantage: float
    approx_kl: float
    clip_fraction: float
    learned_cues: int = 0


class RuleBasedSafetyHead:
    """Simple baseline safety head used to validate data flow."""

    def decide(
        self,
        user_task: str,
        context: str,
        tool_call: ToolCall | None,
        risky_tool: bool,
    ) -> DefenseDecision:
        combined = f"{user_task}\n{context}".lower()
        keyword_hits = [keyword for keyword in RISK_KEYWORDS if keyword in combined]

        if keyword_hits:
            start = min(combined.find(keyword) for keyword in keyword_hits if combined.find(keyword) >= 0)
            end = min(len(combined), start + max(len(keyword) for keyword in keyword_hits))
            return DefenseDecision(SafetyAction.REFUSE, confidence=0.9, attribution_span=(start, end))

        if tool_call is None:
            return DefenseDecision(SafetyAction.EXECUTE, confidence=0.8, attribution_span=None)

        if risky_tool:
            return DefenseDecision(SafetyAction.ASK_CONFIRMATION, confidence=0.65, attribution_span=None)

        return DefenseDecision(SafetyAction.EXECUTE, confidence=0.75, attribution_span=None)


class AlwaysExecuteSafetyHead:
    """No-guard baseline that allows every proposed tool call."""

    def decide(
        self,
        user_task: str,
        context: str,
        tool_call: ToolCall | None,
        risky_tool: bool,
    ) -> DefenseDecision:
        return DefenseDecision(SafetyAction.EXECUTE, confidence=1.0, attribution_span=None)


class AlwaysRefuseSafetyHead:
    """Conservative guard baseline used to expose over-refusal."""

    def decide(
        self,
        user_task: str,
        context: str,
        tool_call: ToolCall | None,
        risky_tool: bool,
    ) -> DefenseDecision:
        if tool_call is None:
            return DefenseDecision(SafetyAction.EXECUTE, confidence=1.0, attribution_span=None)
        return DefenseDecision(SafetyAction.REFUSE, confidence=1.0, attribution_span=_keyword_span(context))


class TrainableSafetyHead:
    """Small PPO-trained linear classifier for the text-tool MVP.

    This is intentionally lightweight and dependency-free. It gives the
    round-based training loop a clipped policy-gradient update path before a
    neural PPO/GRPO policy is introduced.
    """

    def __init__(self, learning_rate: float = 0.05, ppo_config: PPOConfig | None = None) -> None:
        self.ppo_config = ppo_config or PPOConfig(learning_rate=learning_rate)
        self.weights: dict[SafetyAction, defaultdict[str, float]] = {
            action: defaultdict(float) for action in ACTION_ORDER
        }
        self.learned_refusal_cues: set[str] = set()
        self.num_updates = 0
        self.last_update_stats = PPOUpdateStats(
            steps=0,
            mean_reward=0.0,
            mean_advantage=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
        )

    def decide(
        self,
        user_task: str,
        context: str,
        tool_call: ToolCall | None,
        risky_tool: bool,
    ) -> DefenseDecision:
        combined = f"{user_task}\n{context}"
        learned_span = _learned_cue_span(combined, self.learned_refusal_cues)
        if learned_span is not None:
            return DefenseDecision(action=SafetyAction.REFUSE, confidence=0.92, attribution_span=learned_span)

        features = _features_from_inputs(user_task, context, tool_call, risky_tool)
        scores = self._scores(features)
        action = max(ACTION_ORDER, key=lambda candidate: (scores[candidate], -ACTION_ORDER.index(candidate)))
        confidence = _softmax_confidence(scores, action)
        attribution_span = _keyword_span(combined) if action == SafetyAction.REFUSE else None
        return DefenseDecision(action=action, confidence=confidence, attribution_span=attribution_span)

    def update(self, records: list[TrajectoryRecord], rewards: list[dict[str, float]] | None = None) -> int:
        if not records or not rewards:
            self.last_update_stats = PPOUpdateStats(0, 0.0, 0.0, 0.0, 0.0)
            return 0

        self.learned_refusal_cues.update(_extract_refusal_cues(records))

        batch = [
            _PPOExample(
                features=_features_from_record(record),
                action=record.agent_safety_action,
                reward=float(rewards[index]["total"]),
            )
            for index, record in enumerate(records)
            if index < len(rewards)
        ]
        if not batch:
            self.last_update_stats = PPOUpdateStats(0, 0.0, 0.0, 0.0, 0.0)
            return 0

        reward_values = [example.reward for example in batch]
        mean_reward = sum(reward_values) / len(reward_values)
        variance = sum((reward - mean_reward) ** 2 for reward in reward_values) / len(reward_values)
        std_reward = math.sqrt(variance)
        for example in batch:
            example.advantage = (example.reward - mean_reward) / (std_reward + self.ppo_config.advantage_epsilon)
            example.old_probs = self._probs(example.features)
            example.old_action_prob = max(example.old_probs[example.action], 1e-8)

        clip_count = 0
        update_steps = 0
        kl_total = 0.0
        for _ in range(self.ppo_config.epochs):
            for example in batch:
                probs = self._probs(example.features)
                current_prob = max(probs[example.action], 1e-8)
                ratio = current_prob / example.old_action_prob
                clipped_ratio = min(max(ratio, 1.0 - self.ppo_config.clip_epsilon), 1.0 + self.ppo_config.clip_epsilon)
                clipped = ratio != clipped_ratio

                if _ppo_should_skip_gradient(example.advantage, ratio, clipped_ratio):
                    clip_count += int(clipped)
                    kl_total += _categorical_kl(example.old_probs, probs)
                    continue

                scale = example.advantage * ratio
                gradients = _logprob_gradients(example.features, probs, example.action)
                entropy_gradients = _entropy_gradients(example.features, probs)
                for action in ACTION_ORDER:
                    for name, value in gradients[action].items():
                        self.weights[action][name] += self.ppo_config.learning_rate * scale * value
                    for name, value in entropy_gradients[action].items():
                        self.weights[action][name] += (
                            self.ppo_config.learning_rate * self.ppo_config.entropy_coef * value
                        )
                update_steps += 1
                clip_count += int(clipped)
                kl_total += _categorical_kl(example.old_probs, probs)

        self.num_updates += update_steps
        self.last_update_stats = PPOUpdateStats(
            steps=update_steps,
            mean_reward=mean_reward,
            mean_advantage=sum(example.advantage for example in batch) / len(batch),
            approx_kl=kl_total / max(1, self.ppo_config.epochs * len(batch)),
            clip_fraction=clip_count / max(1, self.ppo_config.epochs * len(batch)),
            learned_cues=len(self.learned_refusal_cues),
        )
        return update_steps

    def _scores(self, features: Mapping[str, float]) -> dict[SafetyAction, float]:
        return {
            action: sum(self.weights[action][name] * value for name, value in features.items())
            for action in ACTION_ORDER
        }

    def _probs(self, features: Mapping[str, float]) -> dict[SafetyAction, float]:
        scores = self._scores(features)
        max_score = max(scores.values())
        exp_scores = {candidate: math.exp(score - max_score) for candidate, score in scores.items()}
        denominator = sum(exp_scores.values())
        return {candidate: exp_scores[candidate] / denominator for candidate in ACTION_ORDER}


@dataclass
class _PPOExample:
    features: dict[str, float]
    action: SafetyAction
    reward: float
    advantage: float = 0.0
    old_probs: dict[SafetyAction, float] = field(default_factory=dict)
    old_action_prob: float = 0.0


def _features_from_inputs(
    user_task: str,
    context: str,
    tool_call: ToolCall | None,
    risky_tool: bool,
) -> dict[str, float]:
    combined = f"{user_task}\n{context}".lower()
    tool_name = tool_call.tool_name if tool_call else None
    features: dict[str, float] = {
        "bias": 1.0,
        "has_tool": float(tool_call is not None),
        "no_tool": float(tool_call is None),
        "risky_tool": float(risky_tool),
    }
    if tool_name:
        features[f"tool={tool_name}"] = 1.0
    for keyword in RISK_KEYWORDS:
        if keyword in combined:
            features[f"kw={keyword}"] = 1.0
    return features


def _features_from_record(record: TrajectoryRecord) -> dict[str, float]:
    tool_call = ToolCall(record.tool_name, record.tool_arguments) if record.tool_name else None
    risky_tool = bool(record.tool_name in RISKY_TOOLS)
    features = _features_from_inputs(record.user_task, record.injected_content or "", tool_call, risky_tool)
    if record.attack_type:
        features[f"attack_type={record.attack_type}"] = 1.0
    features[f"trajectory={record.trajectory_type.value}"] = 1.0
    return features


def _softmax_confidence(scores: Mapping[SafetyAction, float], action: SafetyAction) -> float:
    max_score = max(scores.values())
    exp_scores = {candidate: math.exp(score - max_score) for candidate, score in scores.items()}
    denominator = sum(exp_scores.values())
    return exp_scores[action] / denominator if denominator else 1.0 / len(scores)


def _ppo_should_skip_gradient(advantage: float, ratio: float, clipped_ratio: float) -> bool:
    if ratio == clipped_ratio:
        return False
    if advantage >= 0.0 and ratio > clipped_ratio:
        return True
    return advantage < 0.0 and ratio < clipped_ratio


def _logprob_gradients(
    features: Mapping[str, float],
    probs: Mapping[SafetyAction, float],
    action: SafetyAction,
) -> dict[SafetyAction, dict[str, float]]:
    gradients: dict[SafetyAction, dict[str, float]] = {candidate: {} for candidate in ACTION_ORDER}
    for candidate in ACTION_ORDER:
        coefficient = (1.0 if candidate == action else 0.0) - probs[candidate]
        gradients[candidate] = {name: coefficient * value for name, value in features.items()}
    return gradients


def _entropy_gradients(
    features: Mapping[str, float],
    probs: Mapping[SafetyAction, float],
) -> dict[SafetyAction, dict[str, float]]:
    expected_log_prob = sum(probs[action] * math.log(max(probs[action], 1e-8)) for action in ACTION_ORDER)
    gradients: dict[SafetyAction, dict[str, float]] = {candidate: {} for candidate in ACTION_ORDER}
    for candidate in ACTION_ORDER:
        coefficient = -probs[candidate] * (math.log(max(probs[candidate], 1e-8)) - expected_log_prob)
        gradients[candidate] = {name: coefficient * value for name, value in features.items()}
    return gradients


def _categorical_kl(
    old_probs: Mapping[SafetyAction, float],
    new_probs: Mapping[SafetyAction, float],
) -> float:
    return sum(
        old_probs[action] * (math.log(max(old_probs[action], 1e-8)) - math.log(max(new_probs[action], 1e-8)))
        for action in ACTION_ORDER
    )


def _keyword_span(text: str) -> tuple[int, int] | None:
    lowered = text.lower()
    matches = [(lowered.find(keyword), keyword) for keyword in RISK_KEYWORDS if keyword in lowered]
    matches = [(start, keyword) for start, keyword in matches if start >= 0]
    if not matches:
        return None
    start, keyword = min(matches, key=lambda item: item[0])
    return start, start + len(keyword)


def _learned_cue_span(text: str, cues: set[str]) -> tuple[int, int] | None:
    lowered = text.lower()
    matches = [(lowered.find(cue), cue) for cue in cues if cue and cue in lowered]
    matches = [(start, cue) for start, cue in matches if start >= 0]
    if not matches:
        return None
    start, cue = min(matches, key=lambda item: item[0])
    return start, start + len(cue)


def _extract_refusal_cues(records: list[TrajectoryRecord]) -> set[str]:
    cues: set[str] = set()
    for record in records:
        if not _should_mine_refusal_cues(record):
            continue
        text = (record.injected_content or record.user_task).lower()
        cues.update(keyword for keyword in RISK_KEYWORDS if keyword in text)
        cues.update(_canonical_attack_cues(text))
        cues.update(_salient_phrases(text))
    return cues


def _should_mine_refusal_cues(record: TrajectoryRecord) -> bool:
    return (
        record.trajectory_type == TrajectoryType.ATTACKED_TOOL
        and record.gold_safety_action != SafetyAction.EXECUTE
        and record.agent_safety_action == SafetyAction.EXECUTE
    )


def _canonical_attack_cues(text: str) -> set[str]:
    cue_groups = {
        "confirmation": ("confirmation", "confirmed", "asking again"),
        "sensitive": ("sensitive", "private", "secret", "credential", "credentials"),
        "override": ("override", "ignore", "exception", "bypass"),
        "immediate": ("immediate", "immediately", "urgent"),
        "silent": ("silently", "quietly", "omit"),
        "higher-risk": ("higher-risk", "high-risk"),
    }
    return {
        canonical
        for canonical, variants in cue_groups.items()
        if any(variant in text for variant in variants)
    }


def _salient_phrases(text: str) -> set[str]:
    phrases: set[str] = set()
    for separator in (";", ".", ","):
        first_clause = text.split(separator, 1)[0].strip()
        if first_clause:
            break
    else:
        first_clause = text.strip()
    if 8 <= len(first_clause) <= 96:
        phrases.add(first_clause)
    return phrases
