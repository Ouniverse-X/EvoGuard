"""Core data model for EvoGuard.

This module defines the shared, framework-agnostic data structures that flow
through the whole co-evolution pipeline: messages, tool specifications, agent
actions, trajectories, attack individuals (genomes) and the trajectory records
that bundle a trajectory with the analysis signals described in ``docs/plan.md``
(injection point, behavior turning point and their difference ``delta``).

Every structure is a dataclass with explicit ``to_dict`` / ``from_dict`` helpers
so that entire rounds can be serialized to JSON under ``rounds/`` and reloaded
deterministically. Enums serialize to their string ``value``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    """Chat message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TrajectoryKind(str, Enum):
    """Which of the two trajectory variants a record belongs to.

    Mirrors the A / B / C notion from ``docs/plan.md``:

    * ``CLEAN``    -- trajectory A: tool calling with clean (un-poisoned) context.
    * ``ATTACKED`` -- trajectory B or C: tool calling with injected content. The
      concrete success/failure is captured separately by :class:`AttackOutcome`
      (B = ``SUCCESS``, C = ``FAIL``).
    """

    CLEAN = "clean"
    ATTACKED = "attacked"


class AttackOutcome(str, Enum):
    """Outcome of an attacked trajectory.

    * ``NA``      -- not an attacked trajectory (clean).
    * ``SUCCESS`` -- trajectory B: the injection achieved the attacker goal while
      still fooling the defense agent.
    * ``FAIL``    -- trajectory C: injection present but the attack did not
      succeed (either blocked by defense or the malicious goal was not reached).
    """

    NA = "na"
    SUCCESS = "success"
    FAIL = "fail"


# --------------------------------------------------------------------------- #
# Messages & tools
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    """A single chat message exchanged with an LLM."""

    role: Role
    content: str
    # Optional name of the tool for tool-role messages.
    name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=Role(d["role"]),
            content=d.get("content", ""),
            name=d.get("name"),
        )


@dataclass
class ToolParameter:
    """A single parameter of a tool signature."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolParameter":
        return cls(
            name=d["name"],
            type=d.get("type", "string"),
            description=d.get("description", ""),
            required=d.get("required", True),
        )


@dataclass
class ToolSpec:
    """Definition of a callable tool exposed by an environment.

    The specification is deliberately schema-light so that tool definitions from
    heterogeneous datasets (AgentDojo, AgentHarm, ...) can be normalized into a
    single representation that both the defense agent and the simulated tool
    executor can consume.
    """

    name: str
    description: str = ""
    parameters: list[ToolParameter] = field(default_factory=list)

    def json_schema(self) -> dict[str, Any]:
        """Return an OpenAI-style ``function`` JSON schema for this tool."""

        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            properties[p.name] = {"type": _json_type(p.type), "description": p.description}
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolSpec":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            parameters=[ToolParameter.from_dict(p) for p in d.get("parameters", [])],
        )


def _json_type(t: str) -> str:
    """Map loose dataset type strings onto JSON-schema primitive types."""

    t = (t or "").strip().lower()
    mapping = {
        "str": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "array": "array",
        "dict": "object",
        "object": "object",
    }
    return mapping.get(t, "string")


@dataclass
class ToolCall:
    """A concrete tool invocation emitted by an agent."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        """统一工具参数顺序等，便于后续计算拐点."""
        items = sorted(self.arguments.items(), key=lambda kv: kv[0])
        arg_str = ",".join(f"{k}={_stable_repr(v)}" for k, v in items)
        return f"{self.name}({arg_str})"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCall":
        return cls(name=d["name"], arguments=d.get("arguments", {}) or {})


def _stable_repr(v: Any) -> str:
    if isinstance(v, float):
        return repr(round(v, 6))
    return repr(v)


# --------------------------------------------------------------------------- #
# Actions & trajectories
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    """A single interaction turn of an agent.

    A turn optionally carries the agent's chain-of-thought (``thought``), the
    tool it decided to call (``tool_call``) and the observation returned by the
    environment (``observation``). A turn with no ``tool_call`` and a non-empty
    ``final_answer`` marks trajectory termination.
    """

    turn: int
    thought: str = ""
    tool_call: Optional[ToolCall] = None
    observation: str = ""
    final_answer: str = ""
    # Free-form per-turn metadata (e.g. whether injection is visible this turn).
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.tool_call is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "thought": self.thought,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "observation": self.observation,
            "final_answer": self.final_answer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Action":
        tc = d.get("tool_call")
        return cls(
            turn=d["turn"],
            thought=d.get("thought", ""),
            tool_call=ToolCall.from_dict(tc) if tc else None,
            observation=d.get("observation", ""),
            final_answer=d.get("final_answer", ""),
            metadata=d.get("metadata", {}) or {},
        )


@dataclass
class Task:
    """A single benchmark task to be rolled out."""

    task_id: str
    instruction: str
    suite: str = ""
    dataset: str = ""
    # Names of the tools available for this task (subset of the env registry).
    tool_names: list[str] = field(default_factory=list)
    # Optional ground-truth / oracle information used by the environment scorer.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        return cls(
            task_id=d["task_id"],
            instruction=d["instruction"],
            suite=d.get("suite", ""),
            dataset=d.get("dataset", ""),
            tool_names=list(d.get("tool_names", [])),
            metadata=d.get("metadata", {}) or {},
        )


@dataclass
class Trajectory:
    task_id: str
    actions: list[Action] = field(default_factory=list)
    kind: TrajectoryKind = TrajectoryKind.CLEAN
    # The full message list actually presented to the agent (for auditing/SFT).
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def action_signatures(self) -> list[str]:
        """Signatures of every tool call in order (used for turning point)."""

        return [a.tool_call.signature() for a in self.actions if a.tool_call is not None]

    def tool_call_turns(self) -> list[int]:
        return [a.turn for a in self.actions if a.tool_call is not None]

    @property
    def final_answer(self) -> str:
        for a in reversed(self.actions):
            if a.final_answer:
                return a.final_answer
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "actions": [a.to_dict() for a in self.actions],
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        return cls(
            task_id=d["task_id"],
            kind=TrajectoryKind(d.get("kind", "clean")),
            actions=[Action.from_dict(a) for a in d.get("actions", [])],
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
            metadata=d.get("metadata", {}) or {},
        )


# --------------------------------------------------------------------------- #
# Attacks (genetic individuals)
# --------------------------------------------------------------------------- #
@dataclass
class AttackSpec:
    """An attack individual in the genetic population.

    This is the genome that the attacker's genetic algorithm evolves. It fully
    determines how the controller poisons the context of a task:

    * ``target_turn`` -- the interaction turn whose observation gets poisoned.
      This is the *injection point* recorded at construction time.
    * ``injection_channel`` -- which part of the observation is poisoned (e.g.
      a returned document, an email body, a transaction subject...).
    * ``method`` -- a coarse technique label used for diversity accounting
      (e.g. ``"authority"``, ``"urgency"``, ``"tool_result_spoof"``).
    * ``payload`` -- the concrete injected text spliced into the context.
    * ``goal`` -- the malicious objective the attacker wants the agent to do.
    """

    attack_id: str
    task_id: str
    target_turn: int
    injection_channel: str
    method: str
    payload: str
    goal: str = ""
    # Genealogy for the genetic algorithm.
    generation: int = 0
    parents: list[str] = field(default_factory=list)
    origin: str = "seed"  # one of: seed | crossover | mutation | elite
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AttackSpec":
        return cls(
            attack_id=d["attack_id"],
            task_id=d["task_id"],
            target_turn=d["target_turn"],
            injection_channel=d.get("injection_channel", ""),
            method=d.get("method", ""),
            payload=d.get("payload", ""),
            goal=d.get("goal", ""),
            generation=d.get("generation", 0),
            parents=list(d.get("parents", [])),
            origin=d.get("origin", "seed"),
            metadata=d.get("metadata", {}) or {},
        )


@dataclass
class Signals:
    """Analysis signals derived from an attacked trajectory (``docs/plan.md``).

    * ``injection_point`` -- turn at which the attacker injected (from the spec).
    * ``turning_point`` -- first turn at which the attacked trajectory (B)
      significantly diverges from the clean trajectory (A), found via edit
      distance over ``(tool + args)`` sequences. ``None`` if no divergence.
    * ``delta`` -- ``turning_point - injection_point``. Large delta => latent /
      stealthy attack; small delta => immediate-trigger attack.
    * ``delta_normalized`` -- delta scaled to ``[0, 1]`` by trajectory length,
      used directly as the genetic fitness for successful attacks.
    """

    injection_point: Optional[int] = None
    turning_point: Optional[int] = None
    delta: Optional[int] = None
    delta_normalized: float = 0.0
    edit_distance: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Signals":
        return cls(
            injection_point=d.get("injection_point"),
            turning_point=d.get("turning_point"),
            delta=d.get("delta"),
            delta_normalized=d.get("delta_normalized", 0.0),
            edit_distance=d.get("edit_distance"),
            metadata=d.get("metadata", {}) or {},
        )


@dataclass
class TrajectoryRecord:
    """A trajectory bundled with its provenance, outcome and analysis signals.

    This is the atomic unit persisted per round and consumed by both the
    attacker's genetic algorithm (via ``fitness``) and the defender's training
    dataset builder.
    """

    record_id: str
    round_id: int
    task_id: str
    kind: TrajectoryKind
    trajectory: Trajectory
    outcome: AttackOutcome = AttackOutcome.NA
    attack: Optional[AttackSpec] = None
    signals: Optional[Signals] = None
    # Task-level correctness of the *benign* objective (from the env scorer).
    utility: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    def fitness(self) -> float:
        """Genetic fitness of the underlying attack (``docs/plan.md``).

        ``fitness = 0`` if the attack failed (trajectory C), otherwise the
        normalized delta of a successful attack (trajectory B). Non-attacked
        records have zero fitness.
        """

        if self.kind is not TrajectoryKind.ATTACKED:
            return 0.0
        if self.outcome is not AttackOutcome.SUCCESS:
            return 0.0
        if self.signals is None:
            return 0.0
        return float(self.signals.delta_normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "round_id": self.round_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "trajectory": self.trajectory.to_dict(),
            "outcome": self.outcome.value,
            "attack": self.attack.to_dict() if self.attack else None,
            "signals": self.signals.to_dict() if self.signals else None,
            "utility": self.utility,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrajectoryRecord":
        return cls(
            record_id=d["record_id"],
            round_id=d["round_id"],
            task_id=d["task_id"],
            kind=TrajectoryKind(d["kind"]),
            trajectory=Trajectory.from_dict(d["trajectory"]),
            outcome=AttackOutcome(d.get("outcome", "na")),
            attack=AttackSpec.from_dict(d["attack"]) if d.get("attack") else None,
            signals=Signals.from_dict(d["signals"]) if d.get("signals") else None,
            utility=d.get("utility"),
            created_at=d.get("created_at", time.time()),
            metadata=d.get("metadata", {}) or {},
        )
