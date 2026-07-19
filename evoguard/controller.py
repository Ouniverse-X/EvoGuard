"""Controller: drive dual-trajectory collection (``docs/plan.md``).

The controller owns the agent<->environment interaction loop and is the single
place where attacker-controlled content is spliced into the context. It exposes
two primitives:

* :meth:`Controller.run_clean` -- trajectory A: the agent solves the task with a
  clean (un-poisoned) context.
* :meth:`Controller.run_attacked` -- trajectory B/C: identical to A except that,
  at the attack's ``target_turn``, the observation returned to the agent is
  poisoned with the attack payload (through the declared ``injection_channel``).
  The injection point is thus recorded exactly where the controller injects it.

For each attacked run the controller records the interaction turn at which the
poison first became *visible* to the agent, so :mod:`evoguard.process` can align
it against A and compute the behavior turning point and delta.
"""

from __future__ import annotations

from typing import Optional

from evoguard.agents.base import DefenseAgent
from evoguard.config import DefenseConfig
from evoguard.core.types import (
    Action,
    AttackSpec,
    Task,
    Trajectory,
    TrajectoryKind,
)
from evoguard.envs.base import ToolEnv
from evoguard.utils.logging import get_logger

logger = get_logger("controller")


class Controller:
    def __init__(self, agent: DefenseAgent, env: ToolEnv, defense_config: DefenseConfig):
        self.agent = agent
        self.env = env
        self.defense_config = defense_config

    # ------------------------------------------------------------------ #
    # Public rollout primitives
    # ------------------------------------------------------------------ #
    def run_clean(self, task: Task) -> Trajectory:
        return self._run(task, attack=None, kind=TrajectoryKind.CLEAN)

    def run_attacked(self, task: Task, attack: AttackSpec) -> Trajectory:
        return self._run(task, attack=attack, kind=TrajectoryKind.ATTACKED)

    # ------------------------------------------------------------------ #
    # Core interaction loop
    # ------------------------------------------------------------------ #
    def _run(
        self,
        task: Task,
        attack: Optional[AttackSpec],
        kind: TrajectoryKind,
    ) -> Trajectory:
        self.agent.reset()
        tools = self.env.get_tools(task)
        history: list[Action] = []
        max_turns = self.defense_config.max_turns
        injection_visible_turn: Optional[int] = None

        for turn in range(max_turns):
            action = self.agent.decide(task, tools, history)
            action.turn = turn

            if action.tool_call is None:
                history.append(action)
                break

            observation = self.env.execute(task, action.tool_call, history)

            # Splice the attack payload into the observation at the target turn.
            if attack is not None and turn == attack.target_turn:
                observation = self._inject(observation, attack)
                action.metadata["injected"] = True
                injection_visible_turn = turn

            action.observation = observation
            history.append(action)
        else:
            # Loop exhausted without an explicit final answer.
            history.append(
                Action(turn=max_turns, thought="Turn budget exhausted.", final_answer="")
            )

        return Trajectory(
            task_id=task.task_id,
            actions=history,
            kind=kind,
            metadata={
                "instruction": task.instruction,
                "attack_id": attack.attack_id if attack else None,
                "injection_visible_turn": injection_visible_turn,
            },
        )

    # ------------------------------------------------------------------ #
    # Injection
    # ------------------------------------------------------------------ #
    def _inject(self, observation: str, attack: AttackSpec) -> str:
        """Splice ``attack.payload`` into ``observation`` per its channel.
        """

        channel = (attack.injection_channel or "tool_result").lower()
        payload = attack.payload
        if channel in ("document", "file", "attachment"):
            return (
                f"{observation}\n\n--- BEGIN DOCUMENT CONTENT ---\n"
                f"{payload}\n--- END DOCUMENT CONTENT ---"
            )
        if channel in ("email", "email_body", "message"):
            return f"{observation}\n\n[Message body]: {payload}"
        # Default: tool_result / spoofed field, appended in-band.
        return f"{observation}\n{payload}"
