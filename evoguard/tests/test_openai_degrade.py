"""OpenAIClient structured-output degrade-path tests.

We don't hit any real network. Instead we monkeypatch ``OpenAIClient._client``
with a fake whose ``chat.completions.create`` decides per-call whether to
return success or raise an HTTP-shaped exception that mimics OpenAI's
:class:`BadRequestError`. This pins down the three-rung constraint ladder:

* A bare JSON Schema is WRAPPED into ``{"type": "json_schema", ...}`` before it
  goes on the wire -- sending the bare schema makes the server read its
  ``"type": "object"`` as the response-format discriminator.
* Rejection of rung 1 advances to ``extra_body={"guided_json": ...}``, and
  rejection of rung 2 advances to unconstrained; each rung is probed at most
  once per client (capability flags flip sticky-false).
* Once both flags read FALSE subsequent requests go straight to unconstrained --
  zero probes against the structured endpoint ever happen again.
* A successful constrained completion flips that rung's flag TRUE and sticks.

Run::

    python -m evoguard.tests.test_openai_degrade
"""

from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from evoguard.config import LLMConfig  # noqa: E402
from evoguard.core.types import Message, Role  # noqa: E402
from evoguard.llm.openai_client import OpenAIClient  # noqa: E402


class _FakeHTTP400(Exception):
    """Lightweight stand-in for ``openai.BadRequestError``.

    Real :class:`BadRequestError` cannot be constructed offline because its
    initializer demands a live ``httpx.Response``; tests don't care about the
    exact subclass, they only need the same observable surface used by
    :func:`evoguard.llm.openai_client._is_schema_rejection`: a ``status_code``
    int attribute plus a serialisable ``body`` field.
    """

    def __init__(self, message: str, *, body=None):
        super().__init__(message)
        self.status_code = 400
        self.body = body if body is not None else {"error": {"message": message}}


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5


class _FakeChoice:
    def __init__(self, text: str):
        self.message = type("M", (), {"content": text})()


class _FakeResponse:
    model = "fake-model"

    def __init__(self, text: str):
        self.choices = [_FakeChoice(text)]
        self.usage = _FakeUsage()

    @staticmethod
    def model_dump():
        return {"fake": True}


class _FakeCreate:
    """Records every invocation; rejects the constraint rungs it is told to."""

    def __init__(self, reject_strict_mode: bool = True, reject_guided: bool = True):
        self.calls: list[dict] = []
        self.reject_strict_mode = reject_strict_mode
        self.reject_guided = reject_guided

    def __call__(self, **kwargs):
        snapshot = {k: v for k, v in kwargs.items()}     # shallow copy keeps assertions stable
        self.calls.append(snapshot)
        if self.reject_strict_mode and "response_format" in kwargs:
            raise _FakeHTTP400(
                "unsupported parameter: response_format json_schema",
                body={"error": {
                    "message": ("'response_format' parameter "
                                "'json_schema' is unsupported"),
                    "type": "invalid_request_error"}},
            )
        if self.reject_guided and "guided_json" in (kwargs.get("extra_body") or {}):
            raise _FakeHTTP400(
                "unsupported parameter: guided_json",
                body={"error": {"message": "'guided_json' is unsupported",
                                "type": "invalid_request_error"}},
            )
        return _FakeResponse('{"thought":"ok","final_answer":"done"}')


def _wire_fake(reject_strict_mode=True, reject_guided=True):
    cfg = LLMConfig(
        backend="openai", base_url="http://localhost/v1",
        api_key="EMPTY", max_retries=1,
    )
    client = OpenAIClient(cfg)
    fake = _FakeCreate(reject_strict_mode=reject_strict_mode,
                       reject_guided=reject_guided)
    client._client.chat.completions.create = fake      # swap-in callable
    return client, fake


SCHEMA_SNIPPET = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {"x": {"type": "string"}},
    "required": ["x"],
}


def _messages():
    return [
        Message(role=Role.SYSTEM, content="sys-marker-test"),
        Message(role=Role.USER, content="u"),
    ]


def _guided(call: dict):
    return (call.get("extra_body") or {}).get("guided_json")


class DegradePathTests(unittest.TestCase):

    def test_bare_schema_is_wrapped_in_openai_envelope(self):
        client, fake = _wire_fake(reject_strict_mode=False)
        client.chat(_messages(), response_format=SCHEMA_SNIPPET)

        self.assertEqual(len(fake.calls), 1)
        rf = fake.calls[0]["response_format"]
        self.assertEqual(rf["type"], "json_schema",
                         f"schema went on the wire unwrapped: {rf}")
        self.assertEqual(rf["json_schema"]["schema"]["properties"],
                         SCHEMA_SNIPPET["properties"])
        # $schema is stripped: hosted OpenAI rejects it at a strict schema root.
        self.assertNotIn("$schema", rf["json_schema"]["schema"])

    def test_already_wrapped_response_format_passes_through(self):
        client, fake = _wire_fake(reject_strict_mode=False)
        wrapped = {"type": "json_object"}
        client.chat(_messages(), response_format=wrapped)
        self.assertEqual(fake.calls[0]["response_format"], wrapped)

    def test_strict_rejection_advances_to_guided_json(self):
        client, fake = _wire_fake(reject_strict_mode=True, reject_guided=False)
        resp = client.chat(_messages(), response_format=SCHEMA_SNIPPET)

        strict_attempts = [c for c in fake.calls if c.get("response_format") is not None]
        guided_attempts = [c for c in fake.calls if _guided(c) is not None]
        plain_attempts = [c for c in fake.calls
                          if c.get("response_format") is None and _guided(c) is None]

        self.assertEqual(len(strict_attempts), 1, fake.calls)
        self.assertEqual(len(guided_attempts), 1, fake.calls)
        self.assertEqual(plain_attempts, [],
                         f"guided_json worked, must not degrade further: {fake.calls}")
        # guided_json carries the BARE schema, not the envelope.
        self.assertEqual(_guided(guided_attempts[0]), SCHEMA_SNIPPET)
        self.assertIs(client._schema_supported, False)
        self.assertIs(client._guided_supported, True)
        self.assertIn('"final_answer"', resp.text)

    def test_both_rungs_rejected_falls_through_to_unconstrained(self):
        client, fake = _wire_fake(reject_strict_mode=True, reject_guided=True)
        resp = client.chat(_messages(), response_format=SCHEMA_SNIPPET)

        strict_attempts = [c for c in fake.calls if c.get("response_format") is not None]
        guided_attempts = [c for c in fake.calls if _guided(c) is not None]
        plain_attempts = [c for c in fake.calls
                          if c.get("response_format") is None and _guided(c) is None]

        self.assertEqual(len(strict_attempts), 1, fake.calls)
        self.assertEqual(len(guided_attempts), 1, fake.calls)
        self.assertEqual(len(plain_attempts), 1, fake.calls)
        self.assertIs(client._schema_supported, False)
        self.assertIs(client._guided_supported, False)
        self.assertIn('"final_answer"', resp.text)

    def test_capability_cache_sticky_false_skips_every_probe(self):
        client, fake = _wire_fake(reject_strict_mode=False, reject_guided=False)
        # Pretend prior probes already ruled out both constrained rungs.
        client._schema_supported = False
        client._guided_supported = False
        client.chat(_messages(), response_format=SCHEMA_SNIPPET)

        self.assertEqual(len(fake.calls), 1)
        self.assertIsNone(fake.calls[0].get("response_format"),
                          f"strict disabled but sent attempts: {fake.calls}")
        self.assertIsNone(_guided(fake.calls[0]),
                          f"guided disabled but sent attempts: {fake.calls}")
        # Cache stays False (no flip-up permitted by spec).
        self.assertIs(client._schema_supported, False)
        self.assertIs(client._guided_supported, False)

    def test_successful_strict_completion_flips_cache_true_and_sticks(self):
        client, fake = _wire_fake(reject_strict_mode=False)
        # Initial state unknown -> probe succeeds -> cache flips TRUE.
        client.chat(_messages(), response_format=SCHEMA_SNIPPET)
        self.assertIs(client._schema_supported, True)
        self.assertEqual(len(fake.calls), 1)

        # Second call must remain strictly constrained (cache sticks TRUE).
        client.chat(_messages(), response_format=SCHEMA_SNIPPET)
        self.assertIs(client._schema_supported, True)
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(all(c.get("response_format") is not None for c in fake.calls))
        # The guided rung is never probed while rung 1 works.
        self.assertIsNone(client._guided_supported)


if __name__ == "__main__":
    unittest.main()
