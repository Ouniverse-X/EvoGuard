"""Simple in-memory replay buffer."""

from __future__ import annotations

from dataclasses import dataclass, field

from evoguard.types import TrajectoryRecord


@dataclass
class ReplayBuffer:
    records: list[TrajectoryRecord] = field(default_factory=list)

    def add_many(self, records: list[TrajectoryRecord]) -> None:
        self.records.extend(records)

    def recent(self, limit: int = 100) -> list[TrajectoryRecord]:
        return self.records[-limit:]
