"""Defense agents."""

from __future__ import annotations

from evoguard.agents.base import DefenseAgent
from evoguard.agents.llm_agent import LLMDefenseAgent
from evoguard.config import DefenseConfig
from evoguard.llm.base import LLMClient


def build_defense_agent(
    config: DefenseConfig,
    *,
    client: LLMClient | None = None,
    seed: int = 0,
) -> DefenseAgent:
    return LLMDefenseAgent(config, client=client, seed=seed)


__all__ = ["DefenseAgent", "LLMDefenseAgent", "build_defense_agent"]
