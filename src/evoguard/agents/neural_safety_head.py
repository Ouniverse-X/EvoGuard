"""Torch-backed trainable safety head.

This module is the first real local training backend for EvoGuard. It keeps the
model intentionally small so the full train/evaluate/checkpoint loop is easy to
debug before moving to LoRA or PPO/GRPO over language models.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evoguard.types import DefenseDecision, SafetyAction, ToolCall, TrajectoryRecord


ACTION_TO_INDEX = {
    SafetyAction.EXECUTE: 0,
    SafetyAction.ASK_CONFIRMATION: 1,
    SafetyAction.REFUSE: 2,
}
INDEX_TO_ACTION = {index: action for action, index in ACTION_TO_INDEX.items()}


@dataclass(frozen=True)
class NeuralSafetyConfig:
    max_features: int = 512
    hidden_dim: int = 64
    learning_rate: float = 1e-2
    epochs: int = 40
    batch_size: int = 32
    min_token_count: int = 1
    seed: int = 7
    device: str = "auto"


@dataclass(frozen=True)
class NeuralSafetyStats:
    steps: int
    train_loss: float
    train_accuracy: float
    vocab_size: int
    epochs: int


class NeuralSafetyHead:
    """Bag-of-words MLP safety classifier trained from trajectory labels."""

    def __init__(self, config: NeuralSafetyConfig | None = None) -> None:
        self.config = config or NeuralSafetyConfig()
        self.vocab: dict[str, int] = {}
        self.model: Any | None = None
        self.last_update_stats = NeuralSafetyStats(0, 0.0, 0.0, 0, 0)

    def decide(
        self,
        user_task: str,
        context: str,
        tool_call: ToolCall | None,
        risky_tool: bool,
    ) -> DefenseDecision:
        if self.model is None or not self.vocab:
            fallback = SafetyAction.ASK_CONFIRMATION if risky_tool else SafetyAction.EXECUTE
            return DefenseDecision(fallback, confidence=0.5, attribution_span=_keyword_span(context))

        torch = _require_torch()
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            features = self._featurize_inputs(user_task, context, tool_call, risky_tool).unsqueeze(0).to(device)
            probs = torch.softmax(self.model(features), dim=-1)[0]
        index = int(torch.argmax(probs).item())
        action = INDEX_TO_ACTION[index]
        confidence = float(probs[index].item())
        return DefenseDecision(action, confidence=confidence, attribution_span=_keyword_span(context))

    def update(self, records: list[TrajectoryRecord], rewards: list[dict[str, float]] | None = None) -> int:
        if not records:
            self.last_update_stats = NeuralSafetyStats(0, 0.0, 0.0, len(self.vocab), 0)
            return 0

        torch = _require_torch()
        torch.manual_seed(self.config.seed)
        device = _resolve_device(torch, self.config.device)
        self.vocab = _build_vocab(records, self.config)
        self.model = _SafetyMLP(input_dim=len(self.vocab), hidden_dim=self.config.hidden_dim).to(device)

        x = torch.stack([self._featurize_record(record) for record in records]).to(device)
        y = torch.tensor([ACTION_TO_INDEX[record.gold_safety_action] for record in records], dtype=torch.long).to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        steps = 0
        last_loss = 0.0
        indices = torch.arange(len(records), device=device)
        for _ in range(self.config.epochs):
            permutation = indices[torch.randperm(len(indices))]
            for start in range(0, len(records), self.config.batch_size):
                batch_idx = permutation[start : start + self.config.batch_size]
                logits = self.model(x[batch_idx])
                loss = loss_fn(logits, y[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps += 1
                last_loss = float(loss.item())

        with torch.no_grad():
            predictions = torch.argmax(self.model(x), dim=-1)
            accuracy = float((predictions == y).float().mean().item())
        self.last_update_stats = NeuralSafetyStats(
            steps=steps,
            train_loss=last_loss,
            train_accuracy=accuracy,
            vocab_size=len(self.vocab),
            epochs=self.config.epochs,
        )
        return steps

    def save(self, path: str | Path) -> Path:
        torch = _require_torch()
        if self.model is None:
            raise RuntimeError("Cannot save NeuralSafetyHead before training")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "vocab": self.vocab,
                "state_dict": self.model.state_dict(),
                "last_update_stats": asdict(self.last_update_stats),
            },
            output_path,
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "NeuralSafetyHead":
        torch = _require_torch()
        checkpoint = torch.load(Path(path), map_location="cpu")
        head = cls(NeuralSafetyConfig(**checkpoint["config"]))
        head.vocab = dict(checkpoint["vocab"])
        head.model = _SafetyMLP(input_dim=len(head.vocab), hidden_dim=head.config.hidden_dim)
        head.model.load_state_dict(checkpoint["state_dict"])
        stats = checkpoint.get("last_update_stats")
        if isinstance(stats, dict):
            head.last_update_stats = NeuralSafetyStats(**stats)
        return head

    def _featurize_record(self, record: TrajectoryRecord) -> Any:
        tool_call = ToolCall(record.tool_name, record.tool_arguments) if record.tool_name else None
        risky_tool = record.tool_name in {
            "email_sender",
            "file_writer",
            "web_request",
            "bank_transfer_simulator",
            "smart_home_controller",
            "ticket_updater",
        }
        return self._featurize_inputs(record.user_task, record.injected_content or "", tool_call, risky_tool)

    def _featurize_inputs(
        self,
        user_task: str,
        context: str,
        tool_call: ToolCall | None,
        risky_tool: bool,
    ) -> Any:
        torch = _require_torch()
        vector = torch.zeros(len(self.vocab), dtype=torch.float32)
        text = json.dumps(
            {
                "user_task": user_task,
                "context": context,
                "tool": tool_call.tool_name if tool_call else "none",
                "arguments": dict(tool_call.arguments) if tool_call else {},
                "risky_tool": risky_tool,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for token in _tokenize(text):
            index = self.vocab.get(token)
            if index is not None:
                vector[index] += 1.0
        norm = float(vector.sum().item())
        return vector / norm if norm > 0 else vector


def _build_vocab(records: list[TrajectoryRecord], config: NeuralSafetyConfig) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(_tokenize(_record_text(record)))
    tokens = [
        token
        for token, count in counts.most_common(config.max_features)
        if count >= config.min_token_count
    ]
    return {token: index for index, token in enumerate(tokens)}


def _record_text(record: TrajectoryRecord) -> str:
    return json.dumps(
        {
            "user_task": record.user_task,
            "context": record.injected_content,
            "tool": record.tool_name or "none",
            "arguments": dict(record.tool_arguments),
            "trajectory": record.trajectory_type.value,
            "attack_type": record.attack_type,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_.$:-]+", text.lower())


def _keyword_span(context: str) -> tuple[int, int] | None:
    lowered = context.lower()
    for keyword in ("ignore", "override", "secret", "credential", "confirmation", "private", "quietly"):
        start = lowered.find(keyword)
        if start >= 0:
            return start, start + len(keyword)
    return None


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only without torch installed.
        raise RuntimeError("NeuralSafetyHead requires PyTorch. Install torch to use this backend.") from exc
    return torch


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class _SafetyMLP(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        torch = _require_torch()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(ACTION_TO_INDEX)),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)
