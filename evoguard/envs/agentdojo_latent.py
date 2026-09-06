"""AgentDojo-Latent environment (high-Δ, business-process-camouflaged IPI).

Data lives in ``data/agentdojo_latent`` and is *built*, not vendored: the tasks
and their clean plans are lifted verbatim from the vendored AgentDojo v1 suites
(``data/agentdojo/default_suites/v1``), while the injections are hand-authored
under ``authored/`` and assembled by ``scripts/build_agentdojo_latent.py``. The
generated tree is byte-reproducible and pinned by ``_sha256_guard.txt``.

Why a separate env rather than another ``agentdojo`` split
---------------------------------------------------------
The stock AgentDojo injections announce themselves: the payload appends an
imperative that has nothing to do with the plan in flight, so the very next
action leaves the clean plan and Δ (turning − injection) is 0. This set is
authored the other way round -- every payload borrows the authority of the task
already running and *defers* the harmful call by 1--3 turns of genuine progress,
so the trajectory only diverges after the agent has done real work. Replaying it
needs three things the toolsafe envs do not provide:

1. **A declared clean plan.** ``clean_plan`` is AgentDojo's own
   ``ground_truth()`` call sequence, so ``expected_turning_index`` and
   ``expected_delta`` are design intent that can be compared against the Δ
   actually measured at replay time (:mod:`evoguard.process.signals`).
2. **A paired clean row per carrier.** The benign half of every poisoned
   observation ships alongside it, so the clean arm and the attacked arm differ
   *only* by the appended payload. Letting the LLM simulator invent the
   carrier's clean output would make the two arms distinguishable by shape --
   the same trap documented in :mod:`evoguard.envs.injecagent`.
3. **A per-suite tool inventory including the harmful sink.** The sink is drawn
   from the suite's own advertised ``TOOLS`` list and is required to be absent
   from the clean plan, so taking the bait is a real out-of-plan action rather
   than a tool-name novelty test.

The whole set is a held-out diagnostic probe: tasks carry no
``metadata["split"]`` and there is no train/val/test partition. Training on it
would defeat its purpose.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Iterator, Optional, Sequence

from evoguard.core.types import Action, Task, ToolCall, ToolSpec, Trajectory
from evoguard.envs.base import SimulatedToolEnv
from evoguard.envs.tool_parsing import parse_env_info
from evoguard.envs.utility_judge import score_utility as _score_utility_impl
from evoguard.llm.base import LLMClient
from evoguard.utils.logging import get_logger

logger = get_logger("envs.agentdojo_latent")

#: Registered env name. Imported by
#: :mod:`evoguard.process.agentdojo_latent_loader` rather than re-spelled: the
#: task uid embeds it and a mismatch silently matches zero attacks against zero
#: tasks.
DATASET_NAME = "agentdojo_latent"

#: The four AgentDojo v1 suites this set draws from.
SUITES = ("banking", "slack", "travel", "workspace")

#: Declared-Δ buckets present in the shipped data (4 rows each per suite).
DELTAS = (1, 2, 3)

#: The six camouflage families, 2 per suite per family.
TECHNIQUES = (
    "compliance_deferral",
    "dependency_chain",
    "staged_verification",
    "false_correction",
    "quota_precheck",
    "audit_trail",
)

_GUARD_FILE = "_sha256_guard.txt"


def agentdojo_latent_root(data_root: str = "data",
                          subdir: str = "agentdojo_latent") -> str:
    return os.path.join(data_root, subdir)


def task_uid(suite: str, instruction: str) -> str:
    """Task uid, shaped like the other envs' ``<dataset>:<suite>:<sha1>``.

    Must stay identical to ``scripts/build_agentdojo_latent.py::task_uid``.
    """

    digest = hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:10]
    return f"{DATASET_NAME}:{suite}:{digest}"


def verify_guard(root: str) -> None:
    """Fail loudly if a scenario file drifted from the recorded build.

    Same contract as the sha256 guards on the AgentDojo / ASB / InjecAgent
    splits: numbers reported against this probe stay comparable across runs
    only if the probe itself is immutable. Rebuild with
    ``python scripts/build_agentdojo_latent.py`` (which rewrites the guard)
    rather than editing a scenario file by hand.
    """

    path = os.path.join(root, _GUARD_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"agentdojo_latent guard not found: {path}")
    bad: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("sha256(") or ")=" not in line:
                continue
            rel, _, expected = line[len("sha256("):].partition(")=")
            target = os.path.join(root, rel)
            if not os.path.isfile(target):
                bad.append(f"{rel}: missing")
                continue
            with open(target, "rb") as fh:
                actual = hashlib.sha256(fh.read()).hexdigest()
            if actual != expected:
                bad.append(f"{rel}: sha256 {actual[:12]} != {expected[:12]}")
    if bad:
        raise SystemExit(
            "data/agentdojo_latent has been modified:\n  " + "\n  ".join(bad)
            + "\nRe-run scripts/build_agentdojo_latent.py to regenerate."
        )


def iter_scenario_rows(root: str,
                       suites: Optional[Sequence[str]] = None) -> Iterator[dict]:
    """Yield every row of ``scenarios/<suite>.jsonl`` in suite order."""

    for suite in suites or SUITES:
        path = os.path.join(root, "scenarios", f"{suite}.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"agentdojo_latent scenarios not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


class AgentDojoLatentEnv(SimulatedToolEnv):
    """AgentDojo-Latent tasks from ``data/agentdojo_latent``."""

    name = DATASET_NAME

    def __init__(
        self,
        executor: LLMClient,
        data_root: str = "data",
        *,
        suites: Optional[list[str]] = None,
        max_tasks: int = 0,
        utility_judge: Optional[LLMClient] = None,
        subdir: str = "agentdojo_latent",
        verify_sha256: bool = True,
    ):
        super().__init__(executor)
        self._root = agentdojo_latent_root(data_root, subdir)
        if verify_sha256:
            verify_guard(self._root)
        self._suite_filter = [s for s in (suites or SUITES) if s in SUITES]
        self._max_tasks = max_tasks
        self._utility_judge = utility_judge
        self._tasks: list[Task] = []
        self._tools_by_task: dict[str, list[ToolSpec]] = {}
        self._load()

    # ---- ToolEnv API ------------------------------------------------------ #
    def get_tasks(self) -> list[Task]:
        return list(self._tasks)

    def get_tools(self, task: Task) -> list[ToolSpec]:
        return self._tools_by_task.get(task.task_id, [])

    def score_utility(self, task: Task, trajectory: Trajectory) -> tuple[float, str, str]:
        try:
            return _score_utility_impl(
                task=task, trajectory=trajectory, judge_llm=self._utility_judge,
            )
        except Exception as exc:  # noqa: BLE001 - never crash the round loop
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
        """Authored benign text for a carrier, LLM simulation for everything else.

        Note the asymmetry with the replay path: ``controller.run_replay``
        substitutes the *poisoned* observation for the first carrier call, so
        this method only ever serves the carrier on the clean arm. Both arms
        therefore share one hand-written string up to the appended payload.
        """

        benign = (task.metadata or {}).get("benign_observations") or {}
        text = benign.get(tool_call.name)
        if text is not None:
            return text
        return super().execute(task, tool_call, history)

    # ---- loading ---------------------------------------------------------- #
    def _load(self) -> None:
        seen: "OrderedDict[str, Task]" = OrderedDict()
        n_attacks: dict[str, int] = {}
        benign: dict[str, dict[str, str]] = {}

        for rec in iter_scenario_rows(self._root, self._suite_filter):
            uid = str(rec.get("task_id", ""))
            instruction = str(rec.get("task_instruction", "")).strip()
            if not uid or not instruction:
                continue
            if rec.get("sample_type") == "injected":
                n_attacks[uid] = n_attacks.get(uid, 0) + 1
            # Every row -- clean and injected -- carries the carrier's benign
            # text, so the clean-arm map is complete even for a suite whose
            # clean row was de-duplicated away.
            benign.setdefault(uid, {}).setdefault(
                str(rec["carrier_tool"]), str(rec["benign_observation"]),
            )
            if uid in seen:
                continue
            if self._max_tasks and len(seen) >= self._max_tasks:
                continue
            specs = parse_env_info(str(rec.get("env_info", "")))
            seen[uid] = Task(
                task_id=uid,
                instruction=instruction,
                suite=str(rec.get("suite", "")),
                dataset=self.name,
                tool_names=[s.name for s in specs],
                metadata={
                    # No "split" key on purpose -- module docstring.
                    "task_ref": str(rec.get("task_ref", "")),
                    "clean_plan": list(rec.get("clean_plan") or []),
                },
            )
            self._tools_by_task[uid] = specs

        for uid, task in seen.items():
            task.metadata["benign_observations"] = benign.get(uid, {})
            task.metadata["carrier_tools"] = sorted(benign.get(uid, {}))
            task.metadata["n_vendored_attacks"] = n_attacks.get(uid, 0)
        self._tasks = list(seen.values())
        logger.info("Loaded %d %s tasks from %s (%d attack instances)",
                    len(self._tasks), self.name, self._root,
                    sum(n_attacks.values()))


__all__ = [
    "DATASET_NAME",
    "DELTAS",
    "SUITES",
    "TECHNIQUES",
    "AgentDojoLatentEnv",
    "agentdojo_latent_root",
    "iter_scenario_rows",
    "task_uid",
    "verify_guard",
]
