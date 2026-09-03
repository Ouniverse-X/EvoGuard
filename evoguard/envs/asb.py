"""ASB-OPI environment (Agent Security Bench, Observation Prompt Injection).

Data lives in ``data/ASB`` -- a pruned extract of the upstream ASB checkout plus
the EvoGuard split; see ``data/ASB/README.md`` for provenance and the record
schema, and ``docs/asb_opi_integration.md`` for the design rationale.

Three properties of this dataset differ from the toolsafe/AgentDojo envs and
drive everything below:

1. **Rows are attack instances, not trajectory steps.** There is no ``history``
   or ``env_info`` to parse: the benign tool list comes from
   ``data/all_normal_tools.jsonl``, the agent persona from
   ``agents/<agent>/config.json``, and the payload is a plain field. Tasks are
   the 51 distinct ``(corresponding_agent, user_task)`` pairs.
2. **The attacker tool is a NEW capability, not an existing one.** All 400
   payloads name a tool that is absent from the agent's benign list. We do NOT
   put those tools into ``get_tools`` -- 40 entries called
   ``CredentialHarvesting`` beside 2 benign ones would let a defender win on the
   tool list alone. Taking the bait therefore means emitting a tool name that
   only the payload mentions, which the defense agent is free to do (nothing
   validates action names against the list). :meth:`ASBOPIEnv._tool_spec`
   resolves the attacker catalogue so the *executor* can still simulate the
   bait faithfully; only the agent-facing prompt stays benign.
3. **The split unit is the attack instance.** All injected rows sit on the 10
   ``user_task_index == 0`` tasks, so those tasks appear in several splits. A
   task's ``metadata["split"]`` is taken from its single *clean* row, which is
   the only unambiguous per-task label (33 train / 11 val / 7 test).
   ``metadata["attack_splits"]`` records where its attacks actually live.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Iterator, Optional

from evoguard.core.types import Task, ToolCall, ToolSpec, Trajectory
from evoguard.envs.base import SimulatedToolEnv
from evoguard.envs.utility_judge import score_utility as _score_utility_impl
from evoguard.llm.base import LLMClient
from evoguard.utils.logging import get_logger

logger = get_logger("envs.asb")

#: Registered env name. Imported by :mod:`evoguard.process.asb_attack_loader`
#: rather than re-spelled, because the task uid embeds it and a mismatch
#: silently matches zero attacks against zero tasks.
DATASET_NAME = "asb_opi"

#: Split subdirectories, train first so ``max_tasks`` truncation keeps training.
SPLITS = ("train", "val", "test")


def asb_root(data_root: str = "data", subdir: str = "ASB") -> str:
    return os.path.join(data_root, subdir)


def task_uid(agent: str, user_task: str) -> str:
    """Task uid, shaped like the toolsafe envs' ``<dataset>:<suite>:<sha1>``."""

    digest = hashlib.sha1(user_task.encode("utf-8")).hexdigest()[:10]
    return f"{DATASET_NAME}:{agent}:{digest}"


def load_agent_configs(root: str) -> dict[str, dict]:
    """Read ``agents/<agent>/config.json``.

    ``description`` is the upstream system prompt; ``tools`` is the benign tool
    list and its ORDER is load-bearing (first entry is the injection carrier).
    """

    adir = os.path.join(root, "agents")
    if not os.path.isdir(adir):
        raise FileNotFoundError(f"ASB agent configs not found: {adir}")
    out: dict[str, dict] = {}
    for agent in sorted(os.listdir(adir)):
        path = os.path.join(adir, agent, "config.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        desc = cfg.get("description")
        out[agent] = {
            "role_prompt": " ".join(desc).strip() if isinstance(desc, list) else str(desc or "").strip(),
            "tools": [t.split("/")[-1] for t in cfg.get("tools", [])],
        }
    return out


def load_normal_tools(root: str) -> tuple[dict[str, ToolSpec], dict[str, str]]:
    """Read ``data/all_normal_tools.jsonl``.

    Returns ``({tool_name: ToolSpec}, {tool_name: expected_achievement})``. The
    achievement string is what the upstream simulated tool literally returns; we
    hand it to the executor as the canonical success text so LLM-simulated
    observations stay on ASB's distribution.
    """

    specs: dict[str, ToolSpec] = {}
    achievements: dict[str, str] = {}
    for rec in _read_jsonl(os.path.join(root, "data", "all_normal_tools.jsonl")):
        name = str(rec.get("Tool Name", "")).strip()
        if not name:
            continue
        specs[name] = ToolSpec(name=name, description=str(rec.get("Description", "")).strip())
        achievements[name] = str(rec.get("Expected Achievements", "")).strip()
    return specs, achievements


def load_attacker_tools(root: str) -> dict[str, ToolSpec]:
    """Read ``data/all_attack_tools.jsonl`` into a name -> spec catalogue.

    Used only by the tool-execution simulator, never by ``get_tools``.
    """

    out: dict[str, ToolSpec] = {}
    for rec in _read_jsonl(os.path.join(root, "data", "all_attack_tools.jsonl")):
        name = str(rec.get("Attacker Tool", "")).strip()
        if name and name not in out:
            out[name] = ToolSpec(name=name, description=str(rec.get("Description", "")).strip())
    return out


def iter_split_rows(root: str, splits: Optional[tuple[str, ...]] = None) -> Iterator[tuple[str, dict]]:
    """Yield ``(split, record)`` over ``splits/<split>/all.jsonl``."""

    for split in splits or SPLITS:
        path = os.path.join(root, "splits", split, "all.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ASB split file not found: {path}")
        for rec in _read_jsonl(path):
            yield split, rec


def _read_jsonl(path: str) -> Iterator[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ASB data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class ASBOPIEnv(SimulatedToolEnv):
    """ASB-OPI tasks from ``data/ASB``."""

    name = DATASET_NAME

    def __init__(
        self,
        executor: LLMClient,
        data_root: str = "data",
        *,
        suites: Optional[list[str]] = None,
        max_tasks: int = 0,
        utility_judge: Optional[LLMClient] = None,
        subdir: str = "ASB",
    ):
        super().__init__(executor)
        self._root = asb_root(data_root, subdir)
        self._agent_filter = set(suites or [])
        self._max_tasks = max_tasks
        self._utility_judge = utility_judge
        self._agents = load_agent_configs(self._root)
        self._normal_tools, self._achievements = load_normal_tools(self._root)
        self._attacker_tools = load_attacker_tools(self._root)
        self._tasks: list[Task] = []
        self._tools_by_task: dict[str, list[ToolSpec]] = {}
        self._load()

    # ---- ToolEnv API ------------------------------------------------------ #
    def get_tasks(self) -> list[Task]:
        return list(self._tasks)

    def get_tools(self, task: Task) -> list[ToolSpec]:
        """Benign tools only -- see the module docstring, point 2."""

        return self._tools_by_task.get(task.task_id, [])

    def score_utility(self, task: Task, trajectory: Trajectory) -> tuple[float, str, str]:
        try:
            return _score_utility_impl(
                task=task, trajectory=trajectory, judge_llm=self._utility_judge,
            )
        except Exception as exc:                                             # noqa: BLE001 - never crash the round loop
            logger.warning("score_utility raised for task %s: %s",
                           getattr(task, "task_id", "?"), exc)
            return (0.0, "error", f"{type(exc).__name__}: {str(exc)[:300]}")

    # ---- execution -------------------------------------------------------- #
    def _tool_spec(self, task: Task, name: str) -> Optional[ToolSpec]:
        """Resolve benign tools first, then the attacker catalogue.

        The attacker tools are absent from ``get_tools`` on purpose, so the base
        implementation would hand the executor a bare ``{"name": ...}`` stub and
        the bait would be simulated without its description. Resolving the
        catalogue here keeps the simulation faithful without leaking the tools
        into the agent's prompt.
        """

        for spec in self.get_tools(task):
            if spec.name == name:
                return spec
        return self._attacker_tools.get(name)

    def _build_executor_system(self, task: Task, spec: Optional[ToolSpec], call: ToolCall) -> str:
        base = super()._build_executor_system(task, spec, call)
        achievement = self._achievements.get(call.name, "")
        if achievement:
            # Upstream's simulated tools return exactly this string, so pin the
            # simulator to it instead of letting it invent a result.
            base += f"CANONICAL_SUCCESS_RESULT: {achievement}\n"
        return base

    # ---- loading ---------------------------------------------------------- #
    def _load(self) -> None:
        rows: list[tuple[str, dict]] = [
            (split, rec) for split, rec in iter_split_rows(self._root)
            if not self._agent_filter
            or str(rec.get("corresponding_agent", "")) in self._agent_filter
        ]

        # A task's split comes from its clean row; its attacks may live
        # elsewhere (see module docstring, point 3).
        clean_split: dict[str, str] = {}
        attack_splits: dict[str, set[str]] = {}
        n_attacks: dict[str, int] = {}
        for split, rec in rows:
            uid = task_uid(str(rec.get("corresponding_agent", "")),
                           str(rec.get("user_task", "")))
            if rec.get("sample_type") == "clean":
                clean_split.setdefault(uid, split)
            else:
                attack_splits.setdefault(uid, set()).add(split)
                n_attacks[uid] = n_attacks.get(uid, 0) + 1

        seen: "OrderedDict[str, Task]" = OrderedDict()
        n_no_clean_row = 0
        for split, rec in rows:
            agent = str(rec.get("corresponding_agent", ""))
            user_task = str(rec.get("user_task", "")).strip()
            if not agent or not user_task:
                continue
            uid = task_uid(agent, user_task)
            if uid in seen:
                continue
            agent_cfg = self._agents.get(agent)
            if agent_cfg is None:
                logger.warning("no agent config for %r; skipping task %s", agent, uid)
                continue
            tools = [self._normal_tools[n] for n in agent_cfg["tools"]
                     if n in self._normal_tools]
            declared = clean_split.get(uid)
            if declared is None:
                # Every user task ships a clean row upstream; fall back to the
                # row's own split rather than dropping the task silently.
                n_no_clean_row += 1
                declared = split
            metadata = {
                "split": declared,
                "role_prompt": agent_cfg["role_prompt"],
                "asb_agent": agent,
                "asb_user_task_index": rec.get("user_task_index"),
                "carrier_tool": tools[0].name if tools else "",
                "n_vendored_attacks": n_attacks.get(uid, 0),
                "attack_splits": sorted(attack_splits.get(uid, ())),
                "split_from_clean_row": uid in clean_split,
            }
            seen[uid] = Task(
                task_id=uid,
                instruction=user_task,
                suite=agent,
                dataset=self.name,
                tool_names=[t.name for t in tools],
                metadata=metadata,
            )
            self._tools_by_task[uid] = tools
            if self._max_tasks and len(seen) >= self._max_tasks:
                break
        self._tasks = list(seen.values())
        if n_no_clean_row:
            logger.warning("%d ASB tasks had no clean row; split taken from an "
                           "attack row instead", n_no_clean_row)
        logger.info(
            "Loaded %d %s tasks from %s (%d with vendored attacks)",
            len(self._tasks), self.name, self._root,
            sum(1 for t in self._tasks if t.metadata.get("n_vendored_attacks")),
        )


__all__ = [
    "ASBOPIEnv",
    "DATASET_NAME",
    "SPLITS",
    "asb_root",
    "iter_split_rows",
    "load_agent_configs",
    "load_attacker_tools",
    "load_normal_tools",
    "task_uid",
]
