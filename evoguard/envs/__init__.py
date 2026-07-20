"""Environment registry.

New datasets register a builder here so the pipeline can construct them by name
from :class:`~evoguard.config.EnvConfig`.
"""

from __future__ import annotations

from typing import Callable

from evoguard.config import EnvConfig
from evoguard.envs.base import SimulatedToolEnv, ToolEnv
from evoguard.envs.toolsafe import AgentDojoEnv, AgentHarmEnv
from evoguard.llm import build_client
from evoguard.llm.base import LLMClient

_REGISTRY: dict[str, Callable[..., ToolEnv]] = {
    "agentdojo": AgentDojoEnv,
    "agentharm": AgentHarmEnv,
}


def register_env(name: str, builder: Callable[..., ToolEnv]) -> None:
    """Register a new environment builder under ``name``."""

    _REGISTRY[name] = builder


def available_envs() -> list[str]:
    return sorted(_REGISTRY)


def build_env(config: EnvConfig, *, executor: LLMClient | None = None, seed: int = 0) -> ToolEnv:
    """Construct the environment named by ``config.dataset``.

    ``executor`` is the LLM that simulates tool execution; if omitted it is built
    from ``config.tool_executor_llm``. A separate ``utility_judge`` client is
    also constructed from :attr:`EnvConfig.utility_judge_llm` and forwarded to
    envs that support benign-completion scoring (currently the toolsafe family).
    """

    if config.dataset not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset {config.dataset!r}; available: {available_envs()}"
        )
    if executor is None:
        executor = build_client(config.tool_executor_llm, seed=seed)
    utility_judge_client = None
    try:
        # Build a dedicated judge client for utility scoring fallback path.
        # Failure here (e.g. missing creds for qianfan backend) must NOT block
        # environment construction -- score_utility handles None gracefully.
        utility_judge_client = build_client(config.utility_judge_llm, seed=seed + 1)
    except Exception as exc:
        from evoguard.utils.logging import get_logger
        get_logger("envs").warning(
            "utility judge client construction failed (%s); "
            "score_utility will return 0.0 with method='skipped_no_llm'",
            str(exc)[:200],
        )
    builder = _REGISTRY[config.dataset]
    try:
        return builder(
            executor,
            data_root=config.data_root,
            suites=config.suites,
            max_tasks=config.max_tasks,
            utility_judge=utility_judge_client,
        )
    except TypeError:
        # Env doesn't accept utility_judge kwarg yet; fall back to legacy signature.
        return builder(
            executor,
            data_root=config.data_root,
            suites=config.suites,
            max_tasks=config.max_tasks,
        )


__all__ = [
    "ToolEnv",
    "SimulatedToolEnv",
    "AgentDojoEnv",
    "AgentHarmEnv",
    "build_env",
    "register_env",
    "available_envs",
]
