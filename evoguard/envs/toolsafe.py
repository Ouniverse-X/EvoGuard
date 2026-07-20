"""Dataset environments backed by the toolsafe annotations.

The toolsafe dataset (``data/toolsafe/``) provides step-level annotations of
AgentHarm and AgentDojo tool-call trajectories. Each record carries the task
``instruction``, the available tools serialized in ``env_info`` and a ``score``.
We build :class:`~evoguard.core.types.Task` objects from the *distinct* task
instructions (grouping the per-step records) and reuse
:class:`~evoguard.envs.base.SimulatedToolEnv` for LLM-simulated execution.

Two concrete envs are provided:

* :class:`AgentDojoEnv` -- reads ``data/toolsafe/agentdojo-tragj/*.json`` (one
  file per suite: banking, slack, travel, workspace).
* :class:`AgentHarmEnv` -- reads ``data/toolsafe/agentharm-traj/*.json``
  (benign / harmful step files).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Optional

from evoguard.core.types import Task, ToolSpec, Trajectory
from evoguard.envs.base import SimulatedToolEnv, ToolEnv
from evoguard.envs.tool_parsing import parse_env_info
from evoguard.envs.utility_judge import score_utility as _score_utility_impl
from evoguard.llm.base import LLMClient
from evoguard.utils.logging import get_logger

logger = get_logger("envs.toolsafe")


def _task_uid(dataset: str, suite: str, instruction: str) -> str:
    digest = hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:10]
    return f"{dataset}:{suite}:{digest}"


class _ToolSafeEnv(SimulatedToolEnv):
    """Shared loading logic for toolsafe-backed environments."""

    def __init__(
        self,
        executor: LLMClient,
        data_dir: str,
        *,
        suites: Optional[list[str]] = None,
        max_tasks: int = 0,
        utility_judge: Optional[LLMClient] = None,
    ):
        super().__init__(executor)
        self._data_dir = data_dir
        self._suite_filter = set(suites or [])
        self._max_tasks = max_tasks
        self._tasks: list[Task] = []
        self._tools_by_task: dict[str, list[ToolSpec]] = {}
        # Dedicated judge client used by the benign-completion scorer fallback
        # path (Plan C). May be None when caller opts out of utility scoring;
        # in that case score_utility returns 0.0 with method='skipped_no_llm'.
        self._utility_judge = utility_judge
        self._load()

    # ---- ToolEnv API ------------------------------------------------------ #
    def get_tasks(self) -> list[Task]:
        return list(self._tasks)

    def get_tools(self, task: Task) -> list[ToolSpec]:
        return self._tools_by_task.get(task.task_id, [])

    def score_utility(
        self,
        task: Task,
        trajectory: Trajectory,
    ) -> tuple[float, str, str]:
        """Score how well ``trajectory`` completed the BENIGN objective.

        Returns ``(score∈[0,1], method_tag, evidence)`` per
        :mod:`evoguard.envs.utility_judge`. Never raises.
        """
        try:
            return _score_utility_impl(
                task=task,
                trajectory=trajectory,
                judge_llm=self._utility_judge,
                strict_text_check_only=False,
            )
        except Exception as exc:                                            # noqa: BLE001 - never crash round-loop here
            logger.warning("score_utility raised for task %s: %s",
                           getattr(task, "task_id", "?"), exc)
            return (0.0, "error", f"{type(exc).__name__}: {str(exc)[:300]}")

    # ---- loading ---------------------------------------------------------- #
    def _iter_files(self):
        if not os.path.isdir(self._data_dir):
            raise FileNotFoundError(f"toolsafe data dir not found: {self._data_dir}")
        for fname in sorted(os.listdir(self._data_dir)):
            if not fname.endswith(".json"):
                continue
            suite = os.path.splitext(fname)[0]
            if self._suite_filter and suite not in self._suite_filter:
                continue
            yield suite, os.path.join(self._data_dir, fname)

    def _load(self) -> None:
        # Preserve first-seen order of distinct instructions per suite.
        seen: "OrderedDict[str, Task]" = OrderedDict()
        for suite, path in self._iter_files():
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                instruction = rec.get("instruction", "").strip()
                if not instruction:
                    continue
                uid = _task_uid(self.name, suite, instruction)
                if uid in seen:
                    continue
                tools = parse_env_info(rec.get("env_info", ""))
                task = Task(
                    task_id=uid,
                    instruction=instruction,
                    suite=suite,
                    dataset=self.name,
                    tool_names=[t.name for t in tools],
                    metadata={
                        "source_file": os.path.basename(path),
                        "annotated_score": rec.get("score"),
                        "annotated_history": rec.get("history", ""),
                    },
                )
                seen[uid] = task
                self._tools_by_task[uid] = tools
                if self._max_tasks and len(seen) >= self._max_tasks:
                    break
            if self._max_tasks and len(seen) >= self._max_tasks:
                break
        self._tasks = list(seen.values())
        logger.info("Loaded %d %s tasks from %s", len(self._tasks), self.name, self._data_dir)


class AgentDojoEnv(_ToolSafeEnv):
    """AgentDojo tasks from ``data/toolsafe/agentdojo-tragj``."""

    name = "agentdojo"

    def __init__(
        self,
        executor: LLMClient,
        data_root: str = "data",
        *,
        suites: Optional[list[str]] = None,
        max_tasks: int = 0,
        utility_judge: Optional[LLMClient] = None,
    ):
        data_dir = os.path.join(data_root, "toolsafe", "agentdojo-tragj")
        super().__init__(
            executor, data_dir,
            suites=suites, max_tasks=max_tasks,
            utility_judge=utility_judge,
        )


class AgentHarmEnv(_ToolSafeEnv):
    """AgentHarm tasks from ``data/toolsafe/agentharm-traj``."""

    name = "agentharm"

    def __init__(
        self,
        executor: LLMClient,
        data_root: str = "data",
        *,
        suites: Optional[list[str]] = None,
        max_tasks: int = 0,
        utility_judge: Optional[LLMClient] = None,
    ):
        data_dir = os.path.join(data_root, "toolsafe", "agentharm-traj")
        super().__init__(
            executor, data_dir,
            suites=suites, max_tasks=max_tasks,
            utility_judge=utility_judge,
        )
