"""Unified rollout interface (``rollouts/intro.md``).

A *rollout strategy* turns a task (and, for attacked rollouts, an
:class:`AttackSpec`) into a fully-populated :class:`TrajectoryRecord`: it runs the
controller to obtain a trajectory, and for attacked runs it also invokes the
judge (success/failure) and the signal computation (injection point, turning
point, delta) against the matching clean trajectory.

The interface is deliberately small and dependency-injected so new datasets or
rollout regimes plug in without touching the pipeline: everything a strategy
needs (controller, judge, process config, clean-trajectory cache) is supplied at
construction time.
"""

from __future__ import annotations

import abc
from typing import Optional

from evoguard.config import ProcessConfig
from evoguard.controller import Controller
from evoguard.core.types import (
    AttackOutcome,
    AttackSpec,
    Task,
    Trajectory,
    TrajectoryKind,
    TrajectoryRecord,
)
from evoguard.judge import AttackJudge
from evoguard.process.signals import compute_signals
from evoguard.utils.logging import get_logger

logger = get_logger("rollouts")


class RolloutStrategy(abc.ABC):
    """Produces :class:`TrajectoryRecord` objects for a given round."""
    def __init__(self, controller: Controller, round_id: int):
        self.controller = controller
        self.round_id = round_id

    @abc.abstractmethod
    def rollout(self, task: Task, **kwargs) -> TrajectoryRecord:
        """Produce a single record for ``task``."""


class CleanRollout(RolloutStrategy):
    def rollout(self, task: Task, **kwargs) -> TrajectoryRecord:
        traj = self.controller.run_clean(task)
        utility = self.controller.env.score_utility(task, traj)
        return TrajectoryRecord(
            record_id=TrajectoryRecord.new_id(),
            round_id=self.round_id,
            task_id=task.task_id,
            kind=TrajectoryKind.CLEAN,
            trajectory=traj,
            utility=utility,
        )


class AttackedRollout(RolloutStrategy):
    def __init__(
        self,
        controller: Controller,
        round_id: int,
        judge: AttackJudge,
        process_config: ProcessConfig,
    ):
        super().__init__(controller, round_id)
        self.judge = judge
        self.process_config = process_config

    def rollout(self, task: Task, *, attack: AttackSpec, clean: Trajectory, **kwargs) -> TrajectoryRecord:
        traj = self.controller.run_attacked(task, attack)
        success, reason = self.judge.judge(traj, attack)
        outcome = AttackOutcome.SUCCESS if success else AttackOutcome.FAIL
        signals = compute_signals(clean, traj, attack, self.process_config)
        return TrajectoryRecord(
            record_id=TrajectoryRecord.new_id(),
            round_id=self.round_id,
            task_id=task.task_id,
            kind=TrajectoryKind.ATTACKED,
            trajectory=traj,
            outcome=outcome,
            attack=attack,
            signals=signals,
            metadata={"judge_reason": reason},
        )
