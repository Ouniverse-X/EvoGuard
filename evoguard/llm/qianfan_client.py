"""Baidu QianFan chat-completions backend (:class:`LLMClient`).

The QianFan ``https://qianfan.baidubce.com/v2/chat/completions`` endpoint is
OpenAI-shaped on the *request body* side but authenticates via two custom
HTTP headers -- ``appid`` and ``Authorization: Bearer bce-v3/<token>`` --
rather than the single ``Authorization`` header OpenAI's SDK expects. We
therefore talk to it directly through :mod:`requests` instead of going through
:class:`openai.OpenAI`.

Configuration mapping
---------------------

* ``config.backend == "qianfan"`` selects this class.
* ``config.model`` is forwarded as the request body ``model`` field. Use
  values like ``"glm-5"`` for judging and ``"glm-5.2"`` elsewhere.
* Credentials are read from environment variables to keep YAML files safe:

    EVOGUARD_QIANFAN_APPID   e.g. "app-<your-appid>"
    EVOGUARD_QIANFAN_TOKEN   full bearer value, e.g.
                             "bce-v3/<your-bearer-token>"

  If unset we fall back to whatever literal strings live in ``config.api_key``
  using a ``"<appid>|<bearer>"`` split so configs CAN carry credentials inline
  when convenient, though env vars are preferred in shared environments.

Structured output
-----------------

If callers pass ``response_format={"type": "json_schema", ...}`` we forward it
verbatim into the payload body. If the API rejects it (400 mentioning
response_format / json_schema / guided_decoding) we transparently retry once
without that field and flip an instance-level capability flag sticky-false,
mirroring :class:`evoguard.llm.openai_client.OpenAIClient`.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional, Sequence

from evoguard.config import LLMConfig
from evoguard.core.types import Message
from evoguard.llm.base import LLMResponse, LLMClient
from evoguard.utils.logging import get_logger

logger = get_logger("llm.qianfan")

_ENDPOINT_URL = "https://qianfan.baidubce.com/v2/chat/completions"

# Error-message fragments indicating structured-output rejection vs transient failure.
_SCHEMA_REJECT_RE = re.compile(
    r"response_format|json_schema|guided_json|unsupported",
    re.IGNORECASE,
)


def _resolve_credentials(config: LLMConfig) -> tuple[str, str]:
    """Return ``(appid, bearer_token)`` honouring env-var overrides first."""

    appid_env = os.environ.get("EVOGUARD_QIANFAN_APPID", "").strip()
    token_env = os.environ.get("EVOGUARD_QIANFAN_TOKEN", "").strip()
    if appid_env or token_env:
        if not (appid_env and token_env):
            raise ValueError(
                "Both EVOGUARD_QIANFAN_APPID and EVOGUARD_QIANFAN_TOKEN must be set "
                f"(got appid={'set' if appid_env else 'unset'}, "
                f"token={'set' if token_env else 'unset'})."
            )
        return appid_env, token_env

    # Fallback: parse "<appid>|<full-bearer>" out of config.api_key.
    raw = (config.api_key or "").lstrip()
    if "|" not in raw:
        # Allow bare-token form too: assume caller pre-set Authorization header style;
        # treat entire string as the bearer and require explicit APPID env then.
        fallback_appid = os.environ.get("EVOGUARD_QIANFAN_APPID_FALLBACK", "")
        if not fallback_appid:
            raise ValueError(
                "QianFan credentials missing: either set both "
                "EVOGUARD_QIANFAN_APPID + EVOGUARD_QIANFAN_TOKEN, OR populate "
                "config.api_key='<appid>|<bce-v3/...>' (pipe-separated)."
            )
        return fallback_appid, raw.strip()
    appid_part, _, token_part = raw.partition("|")
    return appid_part.strip(), token_part.strip()


class QianFanClient(LLMClient):
    """Chat client backed by direct HTTP POST against the QianFan v2 gateway."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._appid, self._bearer = _resolve_credentials(config)
        self._endpoint_url = os.environ.get(
            "EVOGUARD_QIANFAN_URL", _ENDPOINT_URL,
        )
        try:
            import requests  # noqa: F401 - presence check only
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "'requests' package required by QianFanClient."
            ) from exc

        # Three-state schema-support flag identical semantics to OpenAIClient:
        # None=probe, True=confirmed-supported, False=rejected-once-skip-strict-mode.
        self._schema_supported: Optional[bool] = None

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        base_payload = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": (
                float(temperature)
                if temperature is not None
                else float(self.config.temperature)
            ),
            "disable_search": False,
            "enable_citation": False,
            "safety": {"input_level": "none"},
        }
        max_tok = (
            int(max_tokens) if max_tokens is not None else int(self.config.max_tokens)
        )
        if max_tok > 0:
            base_payload["max_tokens"] = max_tok
        if stop:
            base_payload["stop"] = list(stop)

        use_strict_mode = response_format is not None and self._schema_supported is not False
        return self._call_with_degrade(base_payload, response_format if use_strict_mode else None)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _call_with_degrade(self, base_payload: dict, response_format: Optional[dict]) -> LLMResponse:
        attempt_payload = dict(base_payload)
        if response_format is not None:
            attempt_payload["response_format"] = response_format

        try:
            resp_text, usage_dict, model_name = self._invoke(attempt_payload)
        except _SchemaRejectError as reject_exc:
            logger.warning(
                "QianFan rejected json_schema (%s); degrading to unconstrained mode.",
                str(reject_exc)[:200],
            )
            self._schema_supported = False
            degrade_payload = {k: v for k, v in base_payload.items()
                               if k != "response_format"}
            resp_text, usage_dict, model_name = self._invoke(degrade_payload)

        if response_format is not None and self._schema_supported is None:
            self._schema_supported = True

        return LLMResponse(
            text=resp_text,
            prompt_tokens=int((usage_dict or {}).get("prompt_tokens", 0)) or 0,
            completion_tokens=int((usage_dict or {}).get("completion_tokens", 0)) or 0,
            model=model_name or self.config.model,
        )

    def _invoke(self, payload: dict) -> tuple[str, dict | None, str | None]:
        """POST one request with backoff retries; classify errors."""

        last_exc: Optional[Exception] = None
        headers = {
            "Content-Type": "application/json",
            "appid": self._appid,
            "Authorization": f"Bearer {self._bearer}",
        }
        encoded_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        for attempt in range(1, self.config.max_retries + 1):
            try:
                import requests as rq
                http_resp = rq.post(
                    self._endpoint_url,
                    data=encoded_body,
                    headers=headers,
                    timeout=self.config.timeout,
                )
                status_code = getattr(http_resp, "status_code", 0)

                if status_code >= 400:
                    err_blob = ""
                    try:
                        jbody = http_resp.json()
                        err_msg_field = (
                            (jbody.get("error") or {}).get("message")
                            if isinstance(jbody.get("error"), dict)
                            else jbody.get("error_msg")
                            or jbody.get("error")
                        )
                        err_blob = str(err_msg_field) if err_msg_field else http_resp.text[:500]
                    except Exception:                                # noqa: BLE001
                        err_blob = http_resp.text[:500]
                    exc_cls = type("QianFanAPIError", (Exception,), {})
                    api_err = exc_cls(f"{status_code}: {err_blob}")
                    setattr(api_err, "status_code", status_code)
                    setattr(api_err, "_err_text", err_blob)
                    if (
                        status_code == 400
                        and "response_format" in payload
                        and bool(_SCHEMA_REJECT_RE.search(err_blob))
                    ):
                        raise _SchemaRejectError(err_blob) from api_err
                    last_exc = api_err
                    raise RuntimeError(str(api_err))

                parsed = http_resp.json()

                choice_list = parsed.get("choices")
                text_out = ""
                finish_reason = None
                if isinstance(choice_list, list) and choice_list:
                    ch0 = choice_list[0] or {}
                    msg_obj = ch0.get("message") or {}
                    text_out = msg_obj.get("content") or ""
                    finish_reason = ch0.get("finish_reason")

                usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None
                returned_model = parsed.get("model") or self.config.model

                # Surface truncation explicitly so downstream parsers can degrade gracefully.
                if finish_reason == "length":
                    logger.warning(
                        "[qianfan] %s truncated due to length limit; partial JSON may follow",
                        returned_model,
                    )
                return text_out, usage, returned_model

            except (_SchemaRejectError, RuntimeError):
                raise
            except Exception as exc:                  # noqa: BLE001 - surface after retry loop
                last_exc = exc
                backoff = min(2.0 ** attempt, 30.0)
                logger.warning(
                    "[qianfan] call failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    self.config.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        assert last_exc is not None
        raise RuntimeError(
            f"QianFan call failed after {self.config.max_retries} retries"
        ) from last_exc


class _SchemaRejectError(RuntimeError):
    """Internal sentinel raised when the gateway returns a 4xx rejecting json_schema."""
