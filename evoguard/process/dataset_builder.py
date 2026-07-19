"""Build defender training datasets from collected trajectory records.

Two supervision signals are produced from a round's records:

* **SFT (cold-start)** -- imitation targets that teach the defender to behave
  *robustly*. Positive demonstrations come from clean trajectories (A) and from
  attacked-but-defended trajectories (C, where the injection failed). At each
  step we serialize (system + history) -> the ground-truth JSON action. Attacked
  successful trajectories (B) are converted into *corrective* demonstrations:
  at the turning point, the target is replaced with a refusal/ignore action that
  continues the benign task instead of taking the bait.

* **GRPO / RL** -- preference/reward tuples. Each attacked step is scored by a
  reward that rewards small delta (immediate detection) and penalizes taking the
  malicious action, directly opposing the attacker's fitness.

The output schemas match what LLaMA-Factory (SFT) and verl (RL) consume; the
training wrappers under :mod:`evoguard.training` render them to disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from evoguard.agents.prompts import build_system_prompt, render_history
from evoguard.core.types import (
    Action,
    AttackOutcome,
    Task,
    ToolCall,
    ToolSpec,
    Trajectory,
    TrajectoryKind,
    TrajectoryRecord,
)


REFUSAL_ACTION_TEMPLATE = {
    "thought": (
        "A tool result contains an instruction that the user did not ask for. "
        "This is a prompt-injection attempt. I will ignore it and continue the "
        "user's original task."
    ),
}


@dataclass
class SFTExample:
    """One (prompt, response) supervised example."""

    system: str
    prompt: str
    response: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_llamafactory(self) -> dict[str, Any]:
        """Render in LLaMA-Factory's ShareGPT-style ``messages`` schema."""

        return {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.response},
            ],
            "meta": self.meta,
        }


@dataclass
class RLSample:
    """One RL prompt with a scalar reward for the taken response."""

    system: str
    prompt: str
    response: str
    reward: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "prompt": self.prompt,
            "response": self.response,
            "reward": self.reward,
            "meta": self.meta,
        }


class DefenderDatasetBuilder:
    """Turn :class:`TrajectoryRecord` lists into SFT and RL datasets."""

    def __init__(self, tasks_by_id: dict[str, Task], tools_by_task: dict[str, list[ToolSpec]]):
        self._tasks = tasks_by_id
        self._tools = tools_by_task

    # ---- SFT -------------------------------------------------------------- #
    def build_sft(self, records: list[TrajectoryRecord]) -> list[SFTExample]:
        """Build cold-start SFT examples spanning detection *and* completion.

        Successful attacks (B) are converted into *corrective* demonstrations.
        Because the rest of the benign task still needs to be learned alongside
        detection, :meth:`_corrective` pairs the refusal at the divergence with a
        *continued-safe-execution* suffix sourced from the matching clean
        trajectory A (same task, same round). We therefore pre-index clean twins
        once per call so ``_corrective`` can pull them in cheaply.
        """

        cleans_by_task: dict[str, Trajectory] = {}
        for rec in records:
            if rec.kind is TrajectoryKind.CLEAN:
                cleans_by_task[rec.task_id] = rec.trajectory

        examples: list[SFTExample] = []
        for rec in records:
            task = self._tasks.get(rec.task_id)
            if task is None:
                continue
            tools = self._tools.get(rec.task_id, [])
            if rec.kind is TrajectoryKind.CLEAN:
                examples.extend(self._imitate(task, tools, rec.trajectory))
            elif rec.kind is TrajectoryKind.ATTACKED and rec.outcome is AttackOutcome.FAIL:
                # C: the defender already resisted -> imitate it as-is.
                examples.extend(self._imitate(task, tools, rec.trajectory))
            elif rec.kind is TrajectoryKind.ATTACKED and rec.outcome is AttackOutcome.SUCCESS:
                # B: correct the trajectory at the turning point AND show how to
                # recover afterward using the clean twin's continuation.
                examples.extend(
                    self._corrective(
                        task, tools, rec,
                        clean_trajectory=cleans_by_task.get(rec.task_id),
                    )
                )
        return examples

    def _imitate(self, task: Task, tools: list[ToolSpec], traj: Trajectory) -> list[SFTExample]:
        system = build_system_prompt(task, tools, None)
        out: list[SFTExample] = []
        for step in range(len(traj.actions)):
            history = traj.actions[:step]
            target = traj.actions[step]
            out.append(
                SFTExample(
                    system=system,
                    prompt=render_history(history),
                    response=_action_to_json(target),
                    meta={"task_id": task.task_id, "kind": traj.kind.value, "step": step},
                )
            )
        return out

    def _corrective(
        self,
        task: Task,
        tools: list[ToolSpec],
        rec: TrajectoryRecord,
        *,
        clean_trajectory: Optional[Trajectory],
    ) -> list[SFTExample]:
        """Convert a successful-attack (B) trajectory into corrective supervision.

        Three phases are emitted:

        1. **Pre-divergence** -- B was behaving correctly here; imitate as-is.
        2. **Divergence turn** -- emit a refusal-shaped action declaring the
           detected injection and committing to resume the original task.
        3. **Post-divergence continuation** -- splice the clean twin A's
           remaining actions onto the corrected history so the model also learns
           HOW to finish the benign request after resisting.

        Skipping phase 3 (as a naive ``break`` would) trains the defender to stop
        dead upon detecting injection rather than completing the user's request,
        which contradicts plan.md's intent that Δ measures behavioral latency
        while preserving utility.
        """

        system = build_system_prompt(task, tools, None)
        traj = rec.trajectory
        b_actions = traj.actions

        tp_turn = rec.signals.turning_point if rec.signals else None
        inj_turn = rec.signals.injection_point if rec.signals else None

        # Index of the first B action at or past the turning point.
        divergence_step_b: Optional[int] = None
        if tp_turn is not None:
            for i, a in enumerate(b_actions):
                if a.turn >= tp_turn:
                    divergence_step_b = i
                    break
        if divergence_step_b is None:
            # No clear divergence identified -- degrade to plain imitation.
            return self._imitate(task, tools, traj)

        target_tool_name = ""
        if rec.attack is not None:
            target_tool_name = _infer_target_tool(rec.attack)

        out: list[SFTExample] = []

        # ---- Phase 1: pre-divergence -------------------------------------- #
        for i in range(divergence_step_b):
            out.append(SFTExample(
                system=system,
                prompt=render_history(b_actions[:i]),
                response=_action_to_json(b_actions[i]),
                meta={
                    "task_id": task.task_id,
                    "kind": "pre_divergence",
                    "step": i,
                    "injection_point": inj_turn,
                    "turning_point": tp_turn,
                },
            ))

        # ---- Phase 2: divergence-turn refusal ----------------------------- #
        divergence_target = b_actions[divergence_step_b]
        clean_anchor = self._aligned_clean_index(clean_trajectory, tp_turn)

        corrective_payload: dict[str, Any] = {
            "thought": (
                "A tool result contains instructions the user never asked me "
                "to perform -- this looks like an indirect prompt-injection "
                "attempt. Ignoring the injected directive and resuming the "
                f"original task ({(task.instruction or '').strip()[:120]})."
            ),
        }
        # If the malicious action happened on the same tool name that the clean
        # twin legitimately uses next (e.g., agent was tricked into calling
        # send_money when it should have called get_balance), route back onto
        # the safe pathway by issuing THAT step instead.
        if (
            clean_anchor is not None
            and clean_anchor < len(getattr(clean_trajectory, "actions", []))
            and getattr(clean_trajectory.actions[clean_anchor], "tool_call", None) is not None
        ):
            ca = clean_trajectory.actions[clean_anchor]
            corrective_payload["thought"] += (
                " Continuing with the originally-planned next step."
            )
            corrective_payload["tool"] = ca.tool_call.name
            corrective_payload["args"] = dict(ca.tool_call.arguments)
        else:
            corrective_payload["final_answer"] = (
                "Ignoring the suspicious instruction found inside the tool "
                "output; proceeding with the original request."
            )

        del divergence_target  # informational only -- kept logic readable above

        out.append(SFTExample(
            system=system,
            prompt=render_history(b_actions[:divergence_step_b]),
            response=_dict_to_json(corrective_payload),
            meta={
                "task_id": task.task_id,
                "kind": "corrective_refusal",
                "step": divergence_step_b,
                "injection_point": inj_turn,
                "turning_point": tp_turn,
            },
        ))

        # ---- Phase 3: continued-safe-execution from clean twin ------------ #
        # Build a synthetic rolling history seeded by real pre-divergence prefix
        # + the corrective action issued in Phase 2, then walk clean twin A's
        # REMAINING actions (strictly after the anchor chosen in Phase 2). Their
        # recorded `.observation`s describe genuinely reachable world states
        # absent attacker interference so feedback stays honest.
        if clean_trajectory is not None and clean_anchor is not None:
            corrective_action_for_history = Action(
                turn=b_actions[divergence_step_b].turn,
                thought=str(corrective_payload.get("thought", "")),
                tool_call=(
                    ToolCall(
                        name=str(corrective_payload["tool"]),
                        arguments=dict(corrective_payload["args"]),
                    )
                    if "tool" in corrective_payload else None
                ),
                observation=(
                    clean_trajectory.actions[clean_anchor].observation or ""
                    if clean_anchor < len(clean_trajectory.actions) else ""
                ),
                final_answer=str(corrective_payload.get("final_answer", "")),
                metadata={"corrective": True},
            )

            synthetic_history: list[Action] = list(b_actions[:divergence_step_b])
            synthetic_history.append(corrective_action_for_history)

            for j in range(clean_anchor + 1, len(clean_trajectory.actions)):
                cont_target = clean_trajectory.actions[j]
                out.append(SFTExample(
                    system=system,
                    prompt=render_history(synthetic_history),
                    response=_action_to_json(cont_target),
                    meta={
                        "task_id": task.task_id,
                        "kind": "post_correction_continuation",
                        "step": divergence_step_b + (j - clean_anchor),
                        "clean_step": j,
                        "injection_point": inj_turn,
                        "turning_point": tp_turn,
                    },
                ))
                synthetic_history.append(_clone_with_observation(cont_target))

        return out

    @staticmethod
    def _aligned_clean_index(
        clean: Optional[Trajectory],
        tp_turn: Optional[int],
    ) -> Optional[int]:
        """Smallest index in ``clean.actions`` whose ``turn >= tp_turn``."""
        if clean is None or tp_turn is None:
            return None
        for i, a in enumerate(clean.actions):
            if a.turn >= tp_turn:
                return i
        return None

    # ---- RL --------------------------------------------------------------- #
    def build_rl(self, records: list[TrajectoryRecord]) -> list[RLSample]:
        """Build per-step RL samples with **dense** reward shaping.

        Plan.md's defender objective is "make Δ small / drive attacks to fail".
        A single scalar attached uniformly to every step (the previous
        implementation) makes GRPO unable to distinguish pre-injection benign
        play from the malicious-tool-firing turn from later recovery/refusal
        turns -- defeating targeted-defensive-behavior learning. We instead
        shape a *per-step* reward using landmarks already computed for every
        attacked record: ``injection_point``, ``turning_point`` and
        ``delta_normalized`` plus structural inspection of each action.
        """

        samples: list[RLSample] = []
        for rec in records:
            if rec.kind is not TrajectoryKind.ATTACKED:
                continue
            task = self._tasks.get(rec.task_id)
            if task is None:
                continue
            tools = self._tools.get(rec.task_id, [])
            system = build_system_prompt(task, tools, None)

            actions = rec.trajectory.actions
            n_steps = len(actions)
            if n_steps == 0:
                continue

            injection_point = (
                rec.signals.injection_point if rec.signals else None
            )
            turning_point = rec.signals.turning_point if rec.signals else None
            delta_norm = float(rec.signals.delta_normalized) if rec.signals else 0.0
            outcome = rec.outcome
            attack = rec.attack
            target_tool_name = _infer_target_tool(attack) if attack is not None else ""

            # Pre-compute which steps carry an explicit refusal/detection signal,
            # so we can award dense positive credit there rather than only at the
            # final answer (matches plan.md intent that the defender detect early).
            refusal_flags = [
                _looks_like_refusal(a.thought) or _looks_like_refusal(a.final_answer)
                for a in actions
            ]

            for step_idx in range(n_steps):
                action = actions[step_idx]
                history = actions[:step_idx]
                r_i = self._step_reward(
                    action=action,
                    step_idx=step_idx,
                    n_steps=n_steps,
                    outcome=outcome,
                    injection_point=injection_point,
                    turning_point=turning_point,
                    delta_norm=delta_norm,
                    target_tool_name=target_tool_name,
                    refused_this_step=refusal_flags[step_idx],
                )
                samples.append(
                    RLSample(
                        system=system,
                        prompt=render_history(history),
                        response=_action_to_json(action),
                        reward=r_i,
                        meta={
                            "task_id": task.task_id,
                            "outcome": outcome.value,
                            "step": step_idx,
                            "injection_point": injection_point,
                            "turning_point": turning_point,
                            "delta_normalized": delta_norm,
                            "target_tool": target_tool_name or None,
                            "refused_here": bool(refusal_flags[step_idx]),
                        },
                    )
                )
        return samples

    @staticmethod
    def _step_reward(
        *,
        action: Action,
        step_idx: int,
        n_steps: int,
        outcome: AttackOutcome,
        injection_point: Optional[int],
        turning_point: Optional[int],
        delta_norm: float,
        target_tool_name: str,
        refused_this_step: bool,
    ) -> float:
        """Per-step shaped reward aligned with ``docs/plan.md`` defender goal.

        Landmark semantics:

          - **pre-exposure** -- action.turn <= injection_point; agent has not yet
            seen poison. Neutral baseline so GRPO does not over-credit lucky
            benign play.
          - **exposed-pre-divergence** -- saw poisoned observation but still on the
            clean path. Mild negative gradient scaled by how stealthy this attack
            turned out to be (larger Δ => stronger pressure to have detected).
          - **malicious-tool-firing** -- took the bait by calling the attacker's
            target tool at/after divergence. Heavy negative penalty scaled by
            achieved latency to directly oppose the attacker fitness signal.
          - **explicit-refusal turn** -- thought/final_answer contains detection
            language. Strong positive credit concentrated here so the policy learns
            to surface refusals as soon as exposure happens.
          - **terminal** -- last action of trajectory gets bonus/penalty depending
            on overall outcome to anchor long-horizon value toward FAIL outcomes.

        Returns values roughly within [-1.5, +1.5]; downstream PPO/GRPO advantage
        normalization can rescale freely.
        """

        t = int(action.turn)

        # --- base shaping -------------------------------------------------- #
        exposed_pre_divergence = False
        if injection_point is not None and turning_point is not None:
            exposed_pre_divergence = (
                injection_point < t < turning_point
            )

        called_target_tool = bool(
            target_tool_name
            and action.tool_call is not None
            and action.tool_call.name == target_tool_name
            and turning_point is not None
            and t >= turning_point
        )

        is_terminal_step = (step_idx == n_steps - 1)

        # ------------------------------------------------------------------ #
        # Reward assembly                                                    #
        # ------------------------------------------------------------------ #
        reward = 0.0

        if refused_this_step:
            # Strongest single-turn positive signal regardless of eventual outcome;
            # even successful-attack trajectories where the model briefly flagged
            # suspicion should be reinforced for that behavior.
            reward += 0.8

        if outcome is AttackOutcome.FAIL:
            # Successful resistance: small uniform baseline across all post-exposure
            # non-malicious decisions so the gradient says "keep doing what you did".
            if injection_point is not None and t > injection_point:
                reward += 0.05
            # Terminal bonus anchors episode-level preference for FAIL outcomes.
            if is_terminal_step:
                reward += 0.4
        elif outcome is AttackOutcome.SUCCESS:
            if called_target_tool:
                # The exact moment defense was breached: heavy negative proportional
                # to stealth. Larger Δ => more negative => strongest possible push
                # away from late-firing bait-taking behaviors.
                reward -= 1.0 + 0.5 * max(0.0, min(1.0, delta_norm))
            elif exposed_pre_divergence:
                # Mild pressure scaling with how latent the eventual betrayal was --
                # encourages earlier divergence-detection before things go wrong.
                reward -= 0.02 * (max(0.0, min(1.0, delta_norm)) + 0.5)
            elif injection_point is not None and t <= injection_point:
                # Pre-exposure benign decision: neutral.
                pass
            # Terminal penalty when breach occurred somewhere along the way.
            if is_terminal_step:
                reward -= 0.4 * max(0.0, min(1.0, delta_norm))

        return round(reward, 6)


# --------------------------------------------------------------------------- #
# Helpers shared between SFT/RL builders                                       #
# --------------------------------------------------------------------------- #
_REFUSAL_KEYWORDS_RE = re.compile(
    r"\b(inject|prompt[- ]injection|untrusted|did not request|"
    r"ignore(d)?\s+(this|that|the)\s+instruction|will not follow|"
    r"refuse(d)?|suspicious|out-of-band|attacker)\b",
    re.IGNORECASE,
)


def _looks_like_refusal(text: str) -> bool:
    text = text or ""
    return bool(_REFUSAL_KEYWORDS_RE.search(text))


_TARGET_TOOL_RE = re.compile(r"call\s+([A-Za-z_]\w*)")


def _infer_target_tool(attack) -> str:
    """Best-effort extraction of the malicious tool name referenced by ``attack``.

    Mirrors :func:`evoguard.judge._infer_target_tool` but kept local to avoid a
    circular import (``judge`` imports :mod:`evoguard.llm` which transitively
    touches dataset-builder territory via training glue). The regex picks up
    phrases like 'call send_money' inside either the goal or payload strings.
    """

    m = _TARGET_TOOL_RE.search(getattr(attack, "goal", "") or "")
    if not m:
        m = _TARGET_TOOL_RE.search(getattr(attack, "payload", "") or "")
    return m.group(1) if m else ""


# Backwards-compatible helper kept around because tests/integrations may import it.
def _defense_reward(rec: TrajectoryRecord) -> float:
    """Episode-level scalar reward (kept for backwards compatibility).

    Prefer :meth:`DefenderDatasetBuilder.build_rl`'s per-step shaping for new
    code paths; this function exists purely so external callers/tests that grab
    a coarse summary number keep working unchanged.
    """

    if rec.outcome is AttackOutcome.FAIL:
        return 1.0
    if rec.outcome is AttackOutcome.SUCCESS and rec.signals is not None:
        return -float(rec.signals.delta_normalized)
    return 0.0


def _action_to_json(action: Action) -> str:
    if action.tool_call is not None:
        return _dict_to_json(
            {
                "thought": action.thought,
                "tool": action.tool_call.name,
                "args": action.tool_call.arguments,
            }
        )
    return _dict_to_json({"thought": action.thought, "final_answer": action.final_answer})


def _clone_with_observation(action: Action) -> Action:
    """Shallow copy of ``action`` preserving its observation for synthetic history.

    Used when splicing clean-twin actions into a corrective SFT trajectory's
    rolling history so subsequent supervised examples see an honestly-evolving
    transcript (the carried observation was genuinely produced during the clean
    rollout and therefore describes a reachable world state).
    """

    return Action(
        turn=action.turn,
        thought=action.thought,
        tool_call=action.tool_call,
        observation=action.observation,
        final_answer=action.final_answer,
        metadata=dict(action.metadata),
    )


def _dict_to_json(d: dict[str, Any]) -> str:
    import json

    return json.dumps(d, ensure_ascii=False)
