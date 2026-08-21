"""llama.cpp ``llama-server`` backend (local GGUF models, e.g. GLM-5.2).

Why a separate backend rather than reusing ``backend: "openai"``: the endpoint is
wire-compatible but behaves differently enough that the attacker path needs its
own branch.

* **Rate limits vs. slot limits.** The QianFan gateway throttles by RPM, which
  our retry/backoff absorbs but cannot get around -- attacker throughput was the
  binding constraint on round wall-clock. ``llama-server`` instead has a fixed
  number of parallel decode *slots* (``--parallel``); requests beyond that queue
  server-side instead of being rejected, so the ceiling moves from "requests per
  minute someone else grants us" to "tokens per second this box can produce".
  Keep the caller's concurrency at or below the slot count.
* **Thinking mode.** GLM-5.2 emits ``<think>...</think>`` reasoning inline in the
  content field (llama.cpp has no separate reasoning channel unless started with
  ``--reasoning-format``). Left in place it corrupts the attacker's JSON parsing,
  so the block is stripped from the visible text here.
* **Sampling.** Temperature is pinned by config as usual; the attacker config for
  this backend uses 1.0 by design -- population diversity in the MCTS/GA search
  depends on high-entropy sampling, and unlike a paid gateway there is no cost
  reason to shrink it.

Structured output degrades exactly like :class:`OpenAIClient` (llama.cpp does
support ``response_format: json_schema``, but builds vary, and the inherited
sticky-false capability cache handles either case at the cost of one probe).
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from evoguard.core.types import Message
from evoguard.llm.base import LLMResponse
from evoguard.llm.openai_client import OpenAIClient
from evoguard.utils.logging import get_logger

logger = get_logger("llm.llamacpp")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove GLM-style ``<think>`` blocks from a completion.

    Two cases: a well-formed block anywhere in the text, and a block that ran
    into the token budget and never closed. The latter leaves nothing usable, so
    it collapses to an empty string rather than a truncated monologue that the
    caller's JSON parser would choke on.
    """
    cleaned = _THINK_RE.sub("", text)
    if "<think>" in cleaned.lower():
        cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
    return cleaned.strip()


class LlamaCppClient(OpenAIClient):
    """OpenAI-compatible client tuned for a local ``llama-server`` instance."""

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
        resp = super().chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            response_format=response_format,
            enable_thinking=enable_thinking,
        )
        if "<think>" in (resp.text or "").lower():
            cleaned = strip_thinking(resp.text)
            if not cleaned:
                logger.warning(
                    "llama.cpp completion contained only an unclosed <think> block "
                    "(%d chars); returning empty text so the caller's parser falls back.",
                    len(resp.text),
                )
            resp.text = cleaned
        return resp
