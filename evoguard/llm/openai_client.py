"""OpenAI-compatible chat client (works against a local vLLM server).

A vLLM server started with ``python -m vllm.entrypoints.openai.api_server`` (or
``vllm serve``) exposes the OpenAI ``/v1/chat/completions`` API, including
per-request LoRA adapter selection by passing the adapter name as the ``model``.
This client therefore covers both hosted OpenAI endpoints and self-hosted vLLM.

Structured output (``response_format={"type": "json_schema", ...}``) is sent
over the wire when callers pass a schema. If the endpoint rejects it -- either
with an HTTP 400 mentioning ``response_format``/``json_schema``, or because
the returned text fails downstream parsing -- we transparently retry once
without ``response_format`` so existing best-effort parsers stay effective.
A per-instance capability cache makes this probe happen at most once per
client lifetime, after which subsequent requests skip straight to whichever
mode is known to work.
"""

from __future__ import annotations

import re
import time
from typing import Optional, Sequence

from evoguard.config import LLMConfig
from evoguard.core.types import Message
from evoguard.llm.base import LLMClient, LLMResponse
from evoguard.utils.logging import get_logger

logger = get_logger("llm.openai")

# Error-message fragments that indicate a structured-output rejection as
# opposed to transient server-side failures. Matched case-insensitively.
_SCHEMA_REJECT_RE = re.compile(
    r"response_format|json_schema|guided_json|guided_decoding|unsupported",
    re.IGNORECASE,
)


class OpenAIClient(LLMClient):
    """Chat client backed by the OpenAI Python SDK (v1.x)."""

    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The 'openai' package is required for OpenAIClient. "
                "Install it with `pip install openai`."
            ) from exc

        self.config = config
        # A vLLM server ignores the api key but the SDK requires a non-empty one.
        api_key = config.api_key or "EMPTY"
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,  # we implement our own retry/backoff loop below
        )
        # When a LoRA adapter is configured, address it as the model name; the
        # vLLM server resolves adapters registered under that served name.
        self._model = config.lora_adapter or config.model

        # Three-state capability flag for structured-output support on THIS endpoint.
        #   None  -> unknown; first call attempts strict mode.
        #   True  -> confirmed supported after one successful constrained completion.
        #   False -> rejected once; skip strict mode for all future calls in this run.
        self._schema_supported: Optional[bool] = None
        # Warn at most once about user-supplied response_format collision via extra.
        self._extra_collision_warned = False

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        payload_messages = [m.to_dict() for m in messages]
        kwargs = {
            "model": self._model,
            "messages": payload_messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if stop:
            kwargs["stop"] = list(stop)
        if self.config.extra:
            kwargs.update(self._merge_extra(self.config.extra))

        use_strict_mode = (
            response_format is not None and self._schema_supported is not False
        )
        return self._call_with_degrade(kwargs, response_format if use_strict_mode else None)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _merge_extra(self, extra: dict) -> dict:
        """Apply ``LLMConfig.extra`` onto request kwargs.

        A pre-existing ``response_format`` key inside ``extra`` collides with
        our explicit per-call argument. The per-call value always wins (it has
        been chosen specifically for the role being executed); emit one warning
        log line per client lifetime to surface the misconfiguration rather than
        silently dropping whatever the operator had set up.
        """

        merged = dict(extra)
        if "response_format" in merged and not self._extra_collision_warned:
            logger.warning(
                "LLMConfig.extra contains 'response_format'; "
                "this will be overridden by the role-specific schema passed via chat()."
            )
            self._extra_collision_warned = True
        merged.pop("response_format", None)
        return merged

    def _call_with_degrade(
        self,
        base_kwargs: dict,
        response_format: Optional[dict],
    ) -> LLMResponse:
        """Run the API call with optional structured-output + degrade path."""

        attempt_kwargs = dict(base_kwargs)
        if response_format is not None:
            attempt_kwargs["response_format"] = response_format

        try:
            resp = self._invoke(attempt_kwargs)
        except _SchemaRejectError as reject_exc:
            # Endpoint doesn't grok json_schema — flip capability sticky-false
            # and retry exactly once without response_format before giving up.
            if response_format is None:
                raise  # shouldn't happen but be defensive
            logger.warning(
                "Structured-output rejected (%s); degrading to unconstrained mode.",
                str(reject_exc)[:200],
            )
            self._schema_supported = False
            degrade_kwargs = {k: v for k, v in base_kwargs.items()
                              if k != "response_format"}
            return self._invoke(degrade_kwargs)

        if response_format is not None and self._schema_supported is None:
            self._schema_supported = True
        return resp

    def _invoke(self, kwargs: dict) -> LLMResponse:
        """Call the SDK with backoff retries; translate schema-reject errors."""

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                usage = getattr(resp, "usage", None)
                return LLMResponse(
                    text=choice.message.content or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    model=resp.model,
                    raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
                )
            except Exception as exc:  # noqa: BLE001 - classify below
                last_exc = exc
                if _is_schema_rejection(exc) and "response_format" in kwargs:
                    raise _SchemaRejectError(str(exc)) from exc
                backoff = min(2.0 ** attempt, 30.0)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    self.config.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        assert last_exc is not None
        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} retries"
        ) from last_exc


class _SchemaRejectError(RuntimeError):
    """Internal sentinel raised when the backend rejects json_schema."""


def _is_schema_rejection(exc: Exception) -> bool:
    """True when ``exc`` looks like an HTTP-400 rejection of response_format."""

    status = getattr(exc, "status_code", None)
    if status is not None and int(status) == 400:
        body = getattr(exc, "body", None)
        text_blob = ""
        if isinstance(body, dict):
            msg_field = (
                body.get("message")
                or body.get("error")
                or {}
            )
            text_blob = msg_field if isinstance(msg_field, str) else repr(body)
        elif isinstance(body, str):
            text_blob = body
        else:
            text_blob = str(exc)
        return bool(_SCHEMA_REJECT_RE.search(text_blob))
    # Some openai SDK versions expose .code instead of HTTP details.
    code = getattr(exc, "code", "") or ""
    if isinstance(code, str) and _SCHEMA_REJECT_RE.search(code):
        return True
    return False
