"""Prompt/template-driven attack generation for controlled benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field

from evoguard.attacks.prompt_templates import (
    AUTOMATED_JAILBREAK_ATTACK_TEMPLATES,
    HARD_HELDOUT_ATTACK_TEMPLATES,
    HELDOUT_ATTACK_TEMPLATES,
    TRAIN_ATTACK_TEMPLATES,
)
from evoguard.types import AttackSample, SafetyAction, Task, TrajectoryRecord


@dataclass
class AttackMemory:
    """Stores successful failures by round for adaptive generation context."""

    failures: list[TrajectoryRecord] = field(default_factory=list)
    success_cases: list[TrajectoryRecord] = field(default_factory=list)
    blocked_cases: list[TrajectoryRecord] = field(default_factory=list)

    def add(self, round_id: int, records: list[TrajectoryRecord]) -> None:
        self.add_success_cases(round_id, records)

    def add_success_cases(self, round_id: int, records: list[TrajectoryRecord]) -> None:
        for record in records:
            record.metadata.setdefault("memory_round_id", round_id)
        self.failures.extend(records)
        self.success_cases.extend(records)

    def add_blocked_cases(self, round_id: int, records: list[TrajectoryRecord]) -> None:
        for record in records:
            record.metadata.setdefault("memory_round_id", round_id)
        self.blocked_cases.extend(records)

    def get_recent_failures(self, limit: int = 5) -> list[TrajectoryRecord]:
        return self.failures[-limit:]

    def get_recent_successes(self, limit: int = 5) -> list[TrajectoryRecord]:
        return self.success_cases[-limit:]

    def get_recent_blocked(self, limit: int = 5) -> list[TrajectoryRecord]:
        return self.blocked_cases[-limit:]


class PromptDrivenAttackGenerator:
    """Fixed generator that adapts only through failure-case conditioning."""

    def __init__(self, templates: dict[str, str] | None = None, split: str = "train") -> None:
        self.split = split
        self.templates = templates or _templates_for_split(split)

    def generate(
        self,
        tasks: list[Task],
        round_id: int,
        failure_cases: list[TrajectoryRecord] | None = None,
    ) -> list[AttackSample]:
        samples: list[AttackSample] = []
        target_tasks = [task for task in tasks if task.preferred_tool is not None]
        failure_tools = {case.tool_name for case in failure_cases or [] if case.tool_name}

        for task in target_tasks:
            if failure_tools and task.preferred_tool not in failure_tools:
                continue
            for attack_type, template in self.templates.items():
                injected_content = template.format(tool=task.preferred_tool)
                offset = len(task.user_task) + 1
                samples.append(
                    AttackSample(
                        attack_id=f"{self.split}_r{round_id}_{task.task_id}_{attack_type}",
                        task=task,
                        round_id=round_id,
                        attack_type=attack_type,
                        target_tool=task.preferred_tool,
                        injected_content=injected_content,
                        insertion_offset=offset,
                        gold_safety_action=SafetyAction.REFUSE,
                    )
                )

        if not samples and failure_tools:
            return self.generate(tasks=tasks, round_id=round_id, failure_cases=None)
        return samples


def build_attack_generator(split: str = "train") -> PromptDrivenAttackGenerator:
    return PromptDrivenAttackGenerator(split=split)


def _templates_for_split(split: str) -> dict[str, str]:
    if split == "train":
        return TRAIN_ATTACK_TEMPLATES
    if split in {"eval", "heldout", "test"}:
        return HELDOUT_ATTACK_TEMPLATES
    if split in {"hard_heldout", "hard_eval", "hard_test"}:
        return HARD_HELDOUT_ATTACK_TEMPLATES
    if split in {"automated_jailbreak", "pair_tap_gptfuzzer", "modern_redteam"}:
        return AUTOMATED_JAILBREAK_ATTACK_TEMPLATES
    raise ValueError(f"Unknown attack split: {split}")
