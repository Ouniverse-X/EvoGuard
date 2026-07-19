"""Abstract LLM client interface.

All components that need an LLM (defense agent, attacker, tool executor, judge)
depend only on :class:`LLMClient`, so backends (vLLM/OpenAI-compatible, mock)
are interchangeable. The interface is intentionally minimal: a single
``chat`` call that maps a message list to a text completion.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Sequence

from evoguard.core.types import Message


@dataclass
class LLMResponse:
    """A single completion returned by an :class:`LLMClient`."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    raw: Optional[dict] = None


class LLMClient(abc.ABC):
    """Backend-agnostic chat LLM."""

    @abc.abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """Return a completion for ``messages``.

        Implementations must be side-effect free with respect to ``messages``
        (they may not mutate the input list).

        ``response_format`` is an optional OpenAI-style structured-output
        directive (e.g. ``{"type": "json_schema", "json_schema": {...}}``).
        Backends that support it constrain decoding to the supplied schema;
        backends that do not should silently degrade (see
        :class:`evoguard.llm.openai_client.OpenAIClient` for the canonical
        retry-without-format fallback). The deterministic
        :class:`~evoguard.llm.mock_client.MockClient` accepts the argument
        but ignores it because its output already conforms.
        """

    def complete(self, prompt: str, **kwargs) -> str:
        """Convenience helper for single-user-message prompts."""

        from evoguard.core.types import Role

        return self.chat([Message(role=Role.USER, content=prompt)], **kwargs).text
