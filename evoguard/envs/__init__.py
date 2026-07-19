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
    from ``config.tool_executor_llm``.
    """

    if config.dataset not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset {config.dataset!r}; available: {available_envs()}"
        )
    if executor is None:
        executor = build_client(config.tool_executor_llm, seed=seed)
    builder = _REGISTRY[config.dataset]
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
