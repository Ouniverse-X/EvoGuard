"""InjecAgent environment (indirect prompt injection on tool observations).

Data lives in ``data/InjecAgent`` -- a pruned extract of the upstream InjecAgent
checkout plus the EvoGuard split; see ``data/InjecAgent/README.md`` for
provenance and the row schema, and ``docs/injecagent_integration.md`` for the
design rationale.

Four properties of this dataset drive everything below.

1. **Rows are attack instances on a 17 x 62 grid.** A row is one (user case,
   attacker case) pair; ``splits/<split>/all.jsonl`` carries both halves as
   first-class fields, so tasks are just the 17 distinct user cases and the split
   unit is the attacker case (see :mod:`evoguard.process.split_injecagent`).
   Tasks therefore declare **no** ``metadata["split"]``: all 17 appear in every
   split, so a task-level label would be a lie and would make
   ``driver._split_train_val`` hold tasks out of training for nothing.
2. **The user task needs exactly one tool.** Upstream gives the agent
   ``[User Tool] + Attacker Tools``; we expose only the ``User Tool``. Handing the
   agent the bait catalogue would turn prompt-injection detection into tool-list
   classification -- a defender could refuse anything naming a tool it was not
   asked to use and score perfectly without reading the observation. Taking the
   bait here means emitting a tool name that only the payload mentions.
   :meth:`InjecAgentEnv._tool_spec` resolves the full 330-tool catalogue so the
   *executor* still simulates the bait with its real signature.
3. **Observations are templated, not generated.** Each user case ships a
   ``Tool Response Template`` with a single ``<Attacker Instruction>`` slot. The
   carrier call is served by substituting into that template -- the payload for
   the replay arm (done by the attack loader), a benign string from
   ``benign_fillers.json`` for the clean arm (done here). Letting an LLM invent
   the carrier's observation would drift both arms off the dataset's
   distribution and, worse, make clean and poisoned observations
   distinguishable by shape.
4. **Attacker-tool results are pre-simulated.** ``data/attacker_simulated_
   responses.json`` holds 2347 recorded observations covering the 32
   data-stealing stage-1 read tools. :meth:`execute` prefers them over the LLM
   simulator so an exfiltration chain sees the same stolen data upstream did.
   The 30 direct-harm tools are absent from the cache -- upstream never simulates
   them, because a direct-harm attack is already scored at the call itself -- so
   those fall through to the simulator.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from collections import OrderedDict
from typing import Iterator, Optional, Sequence

from evoguard.core.types import (
    Action,
    Task,
    ToolCall,
    ToolParameter,
    ToolSpec,
    Trajectory,
)
from evoguard.envs.base import SimulatedToolEnv
from evoguard.envs.utility_judge import score_utility as _score_utility_impl
from evoguard.llm.base import LLMClient
from evoguard.utils.logging import get_logger

logger = get_logger("envs.injecagent")

#: Registered env name. Imported by
#: :mod:`evoguard.process.injecagent_attack_loader` rather than re-spelled,
#: because the task uid embeds it and a mismatch silently matches zero attacks
#: against zero tasks.
DATASET_NAME = "injecagent"

#: Split subdirectories, train first so ``max_tasks`` truncation keeps training.
SPLITS = ("train", "val", "test")

#: The placeholder every ``Tool Response Template`` carries exactly once.
PLACEHOLDER = "<Attacker Instruction>"


def injecagent_root(data_root: str = "data", subdir: str = "InjecAgent") -> str:
    return os.path.join(data_root, subdir)


def task_uid(toolkit: str, user_instruction: str) -> str:
    """Task uid, shaped like the other envs' ``<dataset>:<suite>:<sha1>``."""

    digest = hashlib.sha1(user_instruction.encode("utf-8")).hexdigest()[:10]
    return f"{DATASET_NAME}:{toolkit}:{digest}"


def load_tool_catalogue(root: str) -> tuple[dict[str, ToolSpec], dict[str, str]]:
    """Read ``data/tools.json`` into ``({name: ToolSpec}, {name: toolkit})``.

    Upstream keys tools by ``toolkit + tool["name"]`` (``src/utils.py::
    get_tool_dict``); the test cases name them that way, so we must too. The
    toolkit map is what gives a task its ``suite``.
    """

    path = os.path.join(root, "data", "tools.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"InjecAgent tool catalogue not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        toolkits = json.load(f)

    specs: dict[str, ToolSpec] = {}
    kits: dict[str, str] = {}
    for kit in toolkits:
        toolkit = str(kit.get("toolkit", ""))
        for tool in kit.get("tools", []):
            name = f"{toolkit}{tool.get('name', '')}"
            specs[name] = ToolSpec(
                name=name,
                description=str(tool.get("summary", "")).strip(),
                parameters=[
                    ToolParameter(
                        name=str(p.get("name", "")),
                        type=str(p.get("type", "string")),
                        description=str(p.get("description", "")).strip(),
                        required=bool(p.get("required", True)),
                    )
                    for p in tool.get("parameters", [])
                ],
            )
            kits[name] = toolkit
    return specs, kits


def load_benign_fillers(root: str) -> dict[str, str]:
    """Read the EvoGuard-authored clean-arm substitutions (keyed by User Tool)."""

    path = os.path.join(root, "benign_fillers.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"InjecAgent benign fillers not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return {k: str(v) for k, v in json.load(f).items()
                if not k.startswith("_")}


def canonical_args(raw: object) -> Optional[str]:
    """Canonical JSON for a tool-argument blob, or ``None`` if unparseable.

    The response cache is keyed on ``f"({tool},{args})"`` strings that upstream
    built from whatever its agent emitted, so the argument half is sometimes a
    Python repr, sometimes JSON, and sometimes a list wrapping either. Anything
    that does not reduce to a dict is dropped from the exact-match index and
    served by the name-level fallback instead.
    """

    if isinstance(raw, dict):
        return json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)
    text = str(raw or "").strip()
    if not text:
        return "{}"
    for parse in (json.loads, ast.literal_eval):
        try:
            obj = parse(text)
        except (ValueError, SyntaxError):
            continue
        if isinstance(obj, (list, tuple)):                 # e.g. "['{}']"
            obj = obj[0] if obj else {}
        if isinstance(obj, str):
            return canonical_args(obj)
        if isinstance(obj, dict):
            return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return None


def load_simulated_responses(root: str) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Read ``data/attacker_simulated_responses.json``.

    Returns ``({(tool, canonical_args): observation}, {tool: default})``. The
    default prefers the entry recorded with empty arguments and otherwise the
    first entry in file order, so a lookup is deterministic even when the
    defender's arguments never coincide with upstream's.
    """

    path = os.path.join(root, "data", "attacker_simulated_responses.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"InjecAgent response cache not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    exact: dict[tuple[str, str], str] = {}
    first: dict[str, str] = {}
    empty_args: dict[str, str] = {}
    for key, obs in raw.items():
        body = key[1:-1] if key.startswith("(") and key.endswith(")") else key
        name, _, arg_blob = body.partition(",")
        name = name.strip()
        if not name:
            continue
        first.setdefault(name, str(obs))
        args = canonical_args(arg_blob)
        if args is None:
            continue
        exact.setdefault((name, args), str(obs))
        if args == "{}":
            empty_args.setdefault(name, str(obs))
    return exact, {**first, **empty_args}


def iter_split_rows(root: str, splits: Optional[tuple[str, ...]] = None) -> Iterator[tuple[str, dict]]:
    """Yield ``(split, record)`` over ``splits/<split>/all.jsonl``."""

    for split in splits or SPLITS:
        path = os.path.join(root, "splits", split, "all.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"InjecAgent split file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield split, json.loads(line)


class InjecAgentEnv(SimulatedToolEnv):
    """InjecAgent tasks from ``data/InjecAgent``."""

    name = DATASET_NAME

    def __init__(
        self,
        executor: LLMClient,
        data_root: str = "data",
        *,
        suites: Optional[list[str]] = None,
        max_tasks: int = 0,
        utility_judge: Optional[LLMClient] = None,
        subdir: str = "InjecAgent",
    ):
        super().__init__(executor)
        self._root = injecagent_root(data_root, subdir)
        self._suite_filter = set(suites or [])
        self._max_tasks = max_tasks
        self._utility_judge = utility_judge
        self._tools, self._toolkits = load_tool_catalogue(self._root)
        self._fillers = load_benign_fillers(self._root)
        self._sim_exact, self._sim_default = load_simulated_responses(self._root)
        self._tasks: list[Task] = []
        self._tools_by_task: dict[str, list[ToolSpec]] = {}
        self._load()

    # ---- ToolEnv API ------------------------------------------------------ #
    def get_tasks(self) -> list[Task]:
        return list(self._tasks)

    def get_tools(self, task: Task) -> list[ToolSpec]:
        """The single User Tool -- see the module docstring, point 2."""

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
    def execute(
        self,
        task: Task,
        tool_call: ToolCall,
        history: Sequence[Action],
    ) -> str:
        """Three tiers: templated carrier, pre-simulated attacker tool, LLM.

        The carrier and cache tiers are what keep the environment on the
        dataset's own distribution; the LLM simulator only ever sees tools
        upstream itself never simulated (all 30 direct-harm tools plus the
        stage-2 senders).
        """

        carrier = (task.metadata or {}).get("carrier_tool")
        if tool_call.name == carrier:
            return self._benign_observation(task)
        cached = self._cached_response(tool_call)
        if cached is not None:
            return cached
        return super().execute(task, tool_call, history)

    def _benign_observation(self, task: Task) -> str:
        """The carrier's clean result: the template with a benign filler."""

        meta = task.metadata or {}
        template = str(meta.get("tool_response_template", ""))
        filler = self._fillers.get(str(meta.get("carrier_tool", "")), "")
        if not template:
            return f"[{meta.get('carrier_tool', '?')}] returned no output."
        if not filler:
            # Never leave the raw placeholder in an observation: it would tell
            # the defender exactly where an injection is meant to sit.
            logger.warning("no benign filler for %s; using a neutral string",
                           meta.get("carrier_tool"))
            filler = "No additional details are available."
        return template.replace(PLACEHOLDER, filler)

    def _cached_response(self, call: ToolCall) -> Optional[str]:
        args = canonical_args(call.arguments)
        if args is not None:
            hit = self._sim_exact.get((call.name, args))
            if hit is not None:
                return hit
        return self._sim_default.get(call.name)

    def _tool_spec(self, task: Task, name: str) -> Optional[ToolSpec]:
        """Resolve the task's tool first, then the full 330-tool catalogue.

        The attacker tools are absent from ``get_tools`` on purpose, so the base
        implementation would hand the executor a bare ``{"name": ...}`` stub and
        the bait would be simulated without its signature.
        """

        for spec in self.get_tools(task):
            if spec.name == name:
                return spec
        return self._tools.get(name)

    # ---- loading ---------------------------------------------------------- #
    def _load(self) -> None:
        seen: "OrderedDict[str, Task]" = OrderedDict()
        n_attacks: dict[str, int] = {}
        attack_splits: dict[str, set[str]] = {}
        n_unknown_tool = 0

        for split, rec in iter_split_rows(self._root):
            user_tool = str(rec.get("user_tool", ""))
            instruction = str(rec.get("user_instruction", "")).strip()
            spec = self._tools.get(user_tool)
            if spec is None or not instruction:
                n_unknown_tool += 1
                continue
            toolkit = self._toolkits[user_tool]
            if self._suite_filter and toolkit not in self._suite_filter:
                continue
            uid = task_uid(toolkit, instruction)
            n_attacks[uid] = n_attacks.get(uid, 0) + 1
            attack_splits.setdefault(uid, set()).add(split)
            if uid in seen:
                continue
            if self._max_tasks and len(seen) >= self._max_tasks:
                continue
            seen[uid] = Task(
                task_id=uid,
                instruction=instruction,
                suite=toolkit,
                dataset=self.name,
                tool_names=[user_tool],
                metadata={
                    # No "split" key on purpose -- module docstring, point 1.
                    "carrier_tool": user_tool,
                    "tool_parameters": str(rec.get("tool_parameters", "")),
                    "tool_response_template": str(rec.get("tool_response_template", "")),
                    "user_level": str(rec.get("user_level", "")),
                    "user_case_index": rec.get("user_case_index"),
                },
            )
            self._tools_by_task[uid] = [spec]

        for uid, task in seen.items():
            task.metadata["n_vendored_attacks"] = n_attacks.get(uid, 0)
            task.metadata["attack_splits"] = sorted(attack_splits.get(uid, ()))
        self._tasks = list(seen.values())
        if n_unknown_tool:
            logger.warning("%d InjecAgent rows named a tool absent from "
                           "tools.json; skipped", n_unknown_tool)
        logger.info("Loaded %d %s tasks from %s (%d attack instances, "
                    "%d cached attacker observations)",
                    len(self._tasks), self.name, self._root,
                    sum(n_attacks.values()), len(self._sim_exact))


__all__ = [
    "DATASET_NAME",
    "PLACEHOLDER",
    "SPLITS",
    "InjecAgentEnv",
    "canonical_args",
    "injecagent_root",
    "iter_split_rows",
    "load_benign_fillers",
    "load_simulated_responses",
    "load_tool_catalogue",
    "task_uid",
]
