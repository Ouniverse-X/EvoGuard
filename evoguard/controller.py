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

from typing import TYPE_CHECKING, Optional

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

if TYPE_CHECKING:
    from evoguard.process.vendored_attack_parser import VendoredAttack

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

    def run_attacked(
        self, task: Task, attack: AttackSpec, *, clean: Optional[Trajectory] = None,
    ) -> Trajectory:
        """Run an attacked trajectory.

        When ``clean`` is provided, actions before ``attack.target_turn`` are
        replayed verbatim from the clean trajectory instead of re-querying the
        agent.  This guarantees A and B are byte-identical up to the injection
        point, eliminating stochastic pre-injection divergence (negative delta)
        caused by non-zero sampling temperature.
        """
        return self._run(task, attack=attack, kind=TrajectoryKind.ATTACKED, clean=clean)

    def run_replay(
        self,
        task: Task,
        vendored_attack: "VendoredAttack",
    ) -> tuple[Trajectory, AttackSpec]:
        """Faithful replay of a real vendored AgentDojo injection.

        Runs the agent against ``task`` with ONE substitution: the first time
        the agent calls ``vendored_attack.target_tool``, the controller returns
        ``vendored_attack.real_observation`` verbatim (the canonical
        ``<INFORMATION>`` injection block is embedded inside) instead of the
        LLM-simulated observation. Every other call goes through ``env.execute``
        as usual.

        Returns ``(trajectory, attack_spec)`` so callers can feed both to
        :class:`evoguard.judge.AttackJudge` with the exact ``target_turn``
        discovered at runtime. The trajectory carries
        ``metadata['injection_visible_turn']`` set to the substituted turn (or
        ``None`` if the agent never called the target tool this rollout).
        """

        self.agent.reset()
        tools = self.env.get_tools(task)
        history: list[Action] = []
        max_turns = self.defense_config.max_turns
        injection_visible_turn: Optional[int] = None

        target_tool = vendored_attack.target_tool
        real_obs = vendored_attack.real_observation

        for turn in range(max_turns):
            action = self.agent.decide(task, tools, history)
            action.turn = turn

            if action.tool_call is None:
                history.append(action)
                break

            # First call to the injection-bearing tool returns the REAL vendored
            # observation (which carries the canonical <INFORMATION> block).
            if (injection_visible_turn is None
                    and action.tool_call.name == target_tool):
                observation = real_obs
                action.metadata["injected"] = True
                action.metadata["vendored_replay"] = True
                injection_visible_turn = turn
            else:
                observation = self.env.execute(task, action.tool_call, history)

            action.observation = observation
            history.append(action)
        else:
            history.append(
                Action(turn=max_turns, thought="Turn budget exhausted.", final_answer="")
            )

        traj = Trajectory(
            task_id=task.task_id,
            actions=history,
            kind=TrajectoryKind.ATTACKED,
            metadata={
                "instruction": task.instruction,
                "injection_visible_turn": injection_visible_turn,
                "vendored_replay": True,
                "suite": vendored_attack.suite,
                "target_tool": target_tool,
                "mal_tool": vendored_attack.mal_tool,
                "ground_truth_score": vendored_attack.ground_truth_score,
            },
        )

        # Build an AttackSpec so the existing AttackJudge can score this
        # trajectory identically to GA-synthetic attacks. target_turn is the
        # turn at which the real injection actually became visible here.
        attack = AttackSpec(
            attack_id=AttackSpec.new_id(),
            task_id=task.task_id,
            target_turn=injection_visible_turn if injection_visible_turn is not None else 0,
            injection_channel="tool_result",
            method="vendored_replay",
            payload=vendored_attack.payload,
            goal=vendored_attack.goal,
            origin="vendored",
            metadata={
                "real_observation": real_obs,
                "mal_tool": vendored_attack.mal_tool,
                "mal_args": vendored_attack.mal_args,
                "ground_truth_score": vendored_attack.ground_truth_score,
                "reference_fooled_action": vendored_attack.reference_fooled_action,
            },
        )
        return traj, attack

    # ------------------------------------------------------------------ #
    # Core interaction loop
    # ------------------------------------------------------------------ #
    def _run(
        self,
        task: Task,
        attack: Optional[AttackSpec],
        kind: TrajectoryKind,
        *,
        clean: Optional[Trajectory] = None,
    ) -> Trajectory:
        self.agent.reset()
        tools = self.env.get_tools(task)
        history: list[Action] = []
        max_turns = self.defense_config.max_turns
        injection_visible_turn: Optional[int] = None

        # Inject turn (None when clean rollout or attack has no target).
        inject_turn: Optional[int] = attack.target_turn if attack else None

        for turn in range(max_turns):
            # --- Replay clean actions before the injection point -------------
            # When a clean trajectory is supplied, copy its actions (including
            # observations) verbatim for every turn strictly before the
            # injection turn.  This makes A and B identical up to injection,
            # so the turning point can only be at or after injection_point,
            # guaranteeing delta >= 0.
            if (
                clean is not None
                and inject_turn is not None
                and turn < inject_turn
                and turn < len(clean.actions)
            ):
                src = clean.actions[turn]
                replayed = Action(
                    turn=turn,
                    thought=src.thought,
                    tool_call=src.tool_call,
                    observation=src.observation,
                    final_answer=src.final_answer,
                    metadata={**src.metadata},
                )
                history.append(replayed)
                if replayed.tool_call is None:
                    break
                continue

            # --- Normal decision (at/after injection, or no clean supplied) --
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
