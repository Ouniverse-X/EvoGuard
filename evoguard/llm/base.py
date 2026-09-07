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
        enable_thinking: Optional[bool] = None,
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

        ``enable_thinking`` toggles model-side chain-of-thought on backends that
        expose such a knob (currently QianFan GLM-5 family). Other clients
        silently ignore this parameter.
        """

    def complete(self, prompt: str, **kwargs) -> str:
        """Convenience helper for single-user-message prompts."""

        from evoguard.core.types import Role

        return self.chat([Message(role=Role.USER, content=prompt)], **kwargs).text

    def text_completion(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> LLMResponse:
        """Complete a RAW prompt string, bypassing any chat template.

        Needed by defenses whose protection *is* a text format that the served
        model's chat template does not emit.
        :class:`~evoguard.agents.struq_agent.StruQDefenseAgent` is the motivating
        case: StruQ's trained delimiters (``[MARK] [INST] [COLN]`` / ``[INPT]`` /
        ``[RESP]``) appear nowhere in the generic Llama-2 ``chat_template``
        shipped inside its own checkpoint, so routing that arm through
        :meth:`chat` would measure the base model under a mismatched format and
        the defense would never engage. See ``docs/struq_arm_blockers.md``.

        The default implementation degrades to :meth:`chat` with the prompt as a
        single USER message, which keeps
        :class:`~evoguard.llm.mock_client.MockClient` usable offline. Backends
        with a real text-completion route should override it;
        :class:`~evoguard.llm.openai_client.OpenAIClient` maps it onto
        ``POST /v1/completions``.
        """

        from evoguard.core.types import Role

        return self.chat(
            [Message(role=Role.USER, content=prompt)],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
