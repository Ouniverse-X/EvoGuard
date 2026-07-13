"""Attribution reward component."""

from __future__ import annotations

from evoguard.types import TrajectoryRecord


def attribution_reward(record: TrajectoryRecord) -> float:
    return record.attribution_score
