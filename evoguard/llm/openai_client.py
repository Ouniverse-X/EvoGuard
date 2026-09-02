"""OpenAI-compatible chat client (works against a local vLLM server).

A vLLM server started with ``python -m vllm.entrypoints.openai.api_server`` (or
``vllm serve``) exposes the OpenAI ``/v1/chat/completions`` API, including
per-request LoRA adapter selection by passing the adapter name as the ``model``.
This client therefore covers both hosted OpenAI endpoints and self-hosted vLLM.

Structured output is negotiated down a THREE-rung ladder, each rung cached
sticky-false per client instance so the probe happens at most once per rung
per client lifetime:

1. ``response_format={"type": "json_schema", "json_schema": {...}}`` -- the
   OpenAI wire format.
2. ``extra_body={"guided_json": <schema>}`` -- vLLM's native guided-decoding
   parameter, which older servers accept when they reject rung 1.
3. unconstrained decoding, leaving the caller's best-effort text parsers as the
   only defence.

Callers pass a BARE JSON Schema (see :mod:`evoguard.llm.schemas`); rung 1 wraps
it. Passing the bare schema straight through as ``response_format`` -- which is
what every call site did until 2026-09-02 -- makes an OpenAI-compatible server
read the schema's own ``"type": "object"`` as the response-format discriminator
and reject the request with a pydantic ``literal_error`` on
``body.response_format.type``. The rejection was indistinguishable from "this
endpoint has no structured output", so rung 3 absorbed it and EVERY structured
call in the project decoded unconstrained.
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

        # Build a hardened httpx.Client that DISABLES HTTP keep-alive entirely.
        # Motivation: long-running experiments against local vLLM servers were
        # hitting chronic timeouts because the OpenAI SDK's default transport
        # reuses pooled connections; when the server eventually closes an idle
        # keep-alive socket the client side may remain in CLOSE_WAIT and every
        # subsequent request dispatched on it hangs until read-timeout fires.
        # Setting max_keepalive_connections=0 forces each call to open a fresh
        # TCP connection, eliminating the failure mode at the cost of ~few ms
        # extra handshake overhead per LLM call -- negligible vs inference time.
        try:
            import httpx as _httpx
            _hardened_http = _httpx.Client(
                timeout=_httpx.Timeout(
                    connect=30.0,
                    read=float(config.timeout),
                    write=30.0,
                    pool=10.0,
                ),
                limits=_httpx.Limits(
                    max_keepalive_connections=0,  # KEY: no persistent conns.
                    max_connections=32,
                    keepalive_expiry=1.0,
                ),
            )
            self._client = OpenAI(
                api_key=api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=0,  # we implement our own retry/backoff loop below
                http_client=_hardened_http,
            )
        except ImportError:
            # Fallback: SDK-only construction (older envs without httpx tweak).
            self._client = OpenAI(
                api_key=api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=0,
            )
        # When a LoRA adapter is configured, address it as the model name; the
        # vLLM server resolves adapters registered under that served name.
        self._model = config.lora_adapter or config.model

        # Three-state capability flag for structured-output support on THIS endpoint.
        #   None  -> unknown; first call attempts strict mode.
        #   True  -> confirmed supported after one successful constrained completion.
        #   False -> rejected once; skip strict mode for all future calls in this run.
        self._schema_supported: Optional[bool] = None
        # Same three states for the vLLM ``guided_json`` rung, probed only once
        # ``response_format`` has been ruled out.
        self._guided_supported: Optional[bool] = None
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
        enable_thinking: Optional[bool] = None,  # noqa: ARG002 - ignored on OpenAI-compatible
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

        use_strict_mode = response_format is not None and not (
            self._schema_supported is False and self._guided_supported is False
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
        schema: Optional[dict],
    ) -> LLMResponse:
        """Run the API call, walking the constraint ladder on rejection.

        ``schema`` is a BARE JSON Schema. Each rung is attempted at most once per
        client lifetime: a rejection flips that rung's capability flag
        sticky-false so later calls start at the first rung still viable.
        """

        if schema is None:
            return self._invoke(dict(base_kwargs))

        # Rung 1: OpenAI-style json_schema wrapper.
        if self._schema_supported is not False:
            attempt = dict(base_kwargs)
            attempt["response_format"] = _as_openai_response_format(schema)
            try:
                resp = self._invoke(attempt)
            except _SchemaRejectError as exc:
                logger.warning(
                    "response_format json_schema rejected (%s); trying guided_json.",
                    str(exc)[:200],
                )
                self._schema_supported = False
            else:
                self._schema_supported = True
                return resp

        # Rung 2: vLLM's own guided-decoding parameter.
        if self._guided_supported is not False:
            attempt = dict(base_kwargs)
            attempt["extra_body"] = {
                **(base_kwargs.get("extra_body") or {}),
                "guided_json": schema,
            }
            try:
                resp = self._invoke(attempt)
            except _SchemaRejectError as exc:
                logger.warning(
                    "guided_json rejected (%s); degrading to unconstrained decoding.",
                    str(exc)[:200],
                )
                self._guided_supported = False
            else:
                self._guided_supported = True
                return resp

        # Rung 3: unconstrained.
        return self._invoke(dict(base_kwargs))

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
                if _is_schema_rejection(exc) and _carries_constraint(kwargs):
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


#: Name attached to the wrapped schema. Servers echo it in error messages only;
#: it is not part of the constraint.
_SCHEMA_WRAPPER_NAME = "evoguard_structured_output"

#: ``response_format`` values that are already in OpenAI wire form and must be
#: forwarded verbatim instead of being wrapped a second time.
_WIRE_FORMAT_TYPES = frozenset({"json_schema", "json_object", "text"})


def _as_openai_response_format(schema: dict) -> dict:
    """Wrap a bare JSON Schema in the OpenAI ``response_format`` envelope.

    A caller that already built the envelope (``{"type": "json_schema", ...}``)
    gets it back untouched -- the discriminator is the ``type`` field, which in a
    bare schema is a JSON Schema type name (``object``/``array``/...) and never
    one of :data:`_WIRE_FORMAT_TYPES`.

    ``$schema`` is dropped: it is meaningless to the server and hosted OpenAI
    rejects unknown keywords at the root of a strict-mode schema.
    """

    if str(schema.get("type", "")) in _WIRE_FORMAT_TYPES:
        return schema
    inner = {k: v for k, v in schema.items() if k != "$schema"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": _SCHEMA_WRAPPER_NAME,
            "schema": inner,
            "strict": True,
        },
    }


def _carries_constraint(kwargs: dict) -> bool:
    """Whether ``kwargs`` asks the server to constrain decoding.

    Used to decide if an HTTP 400 should be re-raised as a ladder-advancing
    :class:`_SchemaRejectError` rather than retried with backoff. A request with
    no constraint at all has nothing left to degrade to, so its 400 is a genuine
    failure.
    """

    if "response_format" in kwargs:
        return True
    extra = kwargs.get("extra_body")
    return isinstance(extra, dict) and "guided_json" in extra


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
