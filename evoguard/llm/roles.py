"""Role markers embedded at the top of each system prompt.

Every EvoGuard prompt template begins its system message with one of these
marker lines (e.g. ``[[evoguard:role=tool_executor]]``). The marker is harmless
for real LLMs but lets the deterministic :class:`~evoguard.llm.mock_client.
MockClient` route a request to the correct offline behavior without any network
access, which is what the smoke tests rely on.
"""

from __future__ import annotations

ROLE_DEFENSE = "defense_agent"
ROLE_TOOL_EXECUTOR = "tool_executor"
ROLE_JUDGE = "attack_judge"
ROLE_ATTACKER_GENERATE = "attacker_generate"
ROLE_ATTACKER_CROSSOVER = "attacker_crossover"
ROLE_ATTACKER_MUTATE = "attacker_mutate"

_PREFIX = "[[evoguard:role="
_SUFFIX = "]]"


def marker(role: str) -> str:
    """Return the marker line to prepend to a system prompt for ``role``."""

    return f"{_PREFIX}{role}{_SUFFIX}"


def detect_role(text: str) -> str | None:
    """Extract the role encoded by :func:`marker` from ``text`` (or ``None``)."""

    start = text.find(_PREFIX)
    if start == -1:
        return None
    end = text.find(_SUFFIX, start)
    if end == -1:
        return None
    return text[start + len(_PREFIX):end]
