"""Environment registry.

New datasets register a builder here so the pipeline can construct them by name
from :class:`~evoguard.config.EnvConfig`.
"""

from __future__ import annotations

import inspect
from typing import Callable

from evoguard.config import EnvConfig
from evoguard.envs.asb import ASBOPIEnv
from evoguard.envs.base import SimulatedToolEnv, ToolEnv
from evoguard.envs.injecagent import InjecAgentEnv
from evoguard.envs.toolsafe import AgentDojoEnv, AgentDojoSplitEnv, AgentHarmEnv
from evoguard.llm import build_client
from evoguard.llm.base import LLMClient

_REGISTRY: dict[str, Callable[..., ToolEnv]] = {
    "agentdojo": AgentDojoEnv,
    "agentdojo_split": AgentDojoSplitEnv,
    "agentharm": AgentHarmEnv,
    ASBOPIEnv.name: ASBOPIEnv,
    InjecAgentEnv.name: InjecAgentEnv,
}


def register_env(name: str, builder: Callable[..., ToolEnv]) -> None:
    """Register a new environment builder under ``name``."""

    _REGISTRY[name] = builder


def available_envs() -> list[str]:
    return sorted(_REGISTRY)


def build_env(config: EnvConfig, *, executor: LLMClient | None = None, seed: int = 0) -> ToolEnv:
    if config.dataset not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset {config.dataset!r}; available: {available_envs()}"
        )
    if executor is None:
        executor = build_client(config.tool_executor_llm, seed=seed)
    utility_judge_client = None
    try:
        # Build a dedicated judge client for utility scoring fallback path.
        utility_judge_client = build_client(config.utility_judge_llm, seed=seed + 1)
    except Exception as exc:
        from evoguard.utils.logging import get_logger
        get_logger("envs").warning(
            "utility judge client construction failed (%s); "
            "score_utility will return 0.0 with method='skipped_no_llm'",
            str(exc)[:200],
        )
    builder = _REGISTRY[config.dataset]
    extra: dict[str, object] = {}
    # Only the toolsafe envs accept this; ASB / InjecAgent ship their own
    # attacker tools. Probed rather than passed blindly so a builder that does
    # not know the knob keeps its ``utility_judge`` instead of falling through
    # to the reduced call below.
    if getattr(config, "inject_harmful_tools", False) and _accepts(
        builder, "inject_harmful_tools"
    ):
        extra["inject_harmful_tools"] = True
    try:
        return builder(
            executor,
            data_root=config.data_root,
            suites=config.suites,
            max_tasks=config.max_tasks,
            utility_judge=utility_judge_client,
            **extra,
        )
    except TypeError:
        return builder(
            executor,
            data_root=config.data_root,
            suites=config.suites,
            max_tasks=config.max_tasks,
        )


def _accepts(builder: Callable[..., ToolEnv], name: str) -> bool:
    try:
        return name in inspect.signature(builder).parameters
    except (TypeError, ValueError):
        return False


__all__ = [
    "ToolEnv",
    "SimulatedToolEnv",
    "AgentDojoEnv",
    "AgentDojoSplitEnv",
    "AgentHarmEnv",
    "ASBOPIEnv",
    "InjecAgentEnv",
    "build_env",
    "register_env",
    "available_envs",
]
