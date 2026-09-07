"""Defense agents.

``build_defense_agent`` is the single seam every caller uses
(``eval/vendored_replay.py``, ``eval/stepwise_eval.py``, ``eval/autodojo_eval.py``,
``pipeline/driver.py``, ``preliminary/collect.py``), so the two eval-only
baselines are selected by ``DefenseConfig.agent`` rather than by touching those
five call sites. An unknown name is a hard error: silently falling back to
``llm`` would report a baseline's numbers under the base model's behaviour.
"""

from __future__ import annotations

from evoguard.agents.base import DefenseAgent
from evoguard.agents.llm_agent import LLMDefenseAgent
from evoguard.agents.secalign_agent import SecAlignDefenseAgent
from evoguard.agents.shieldagent_guard import ShieldAgentGuardAgent
from evoguard.agents.struq_agent import StruQDefenseAgent
from evoguard.config import DefenseConfig
from evoguard.llm.base import LLMClient

_AGENTS: dict[str, type[DefenseAgent]] = {
    "llm": LLMDefenseAgent,
    ShieldAgentGuardAgent.name: ShieldAgentGuardAgent,
    "secalign": SecAlignDefenseAgent,
    "struq": StruQDefenseAgent,
}


def build_defense_agent(
    config: DefenseConfig,
    *,
    client: LLMClient | None = None,
    seed: int = 0,
) -> DefenseAgent:
    name = (config.agent or "llm").strip()
    try:
        cls = _AGENTS[name]
    except KeyError:
        raise ValueError(
            f"unknown defense.agent {name!r}; known: {sorted(_AGENTS)}"
        ) from None
    return cls(config, client=client, seed=seed)


__all__ = [
    "DefenseAgent",
    "LLMDefenseAgent",
    "SecAlignDefenseAgent",
    "ShieldAgentGuardAgent",
    "StruQDefenseAgent",
    "build_defense_agent",
]
