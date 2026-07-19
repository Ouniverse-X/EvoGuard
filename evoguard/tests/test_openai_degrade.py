"""OpenAIClient structured-output degrade-path tests.

We don't hit any real network. Instead we monkeypatch ``OpenAIClient._client``
with a fake whose ``chat.completions.create`` decides per-call whether to
return success or raise an HTTP-shaped exception that mimics OpenAI's
:class:`BadRequestError`. This pins down three behaviours:

* First-call rejection triggers exactly one retry WITHOUT ``response_format``;
  capability cache flips sticky-FALSE afterwards.
* Once cache reads FALSE subsequent requests skip straight to unconstrained mode
  -- zero probes against the structured endpoint ever happen again.
* Successful strict-mode completion flips capability TRUE and sticks across future calls.

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
    """Records every invocation; rejects whenever caller sends ``response_format``."""

    def __init__(self, reject_strict_mode: bool = True):
        self.calls: list[dict] = []
        self.reject_strict_mode = reject_strict_mode

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
        return _FakeResponse('{"thought":"ok","final_answer":"done"}')


def _wire_fake(reject_strict_mode=True):
    cfg = LLMConfig(
        backend="openai", base_url="http://localhost/v1",
        api_key="EMPTY", max_retries=1,
    )
    client = OpenAIClient(cfg)
    fake = _FakeCreate(reject_strict_mode=reject_strict_mode)
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


class DegradePathTests(unittest.TestCase):

    def test_first_call_rejection_triggers_unconstrained_retry(self):
        client, fake = _wire_fake(reject_strict_mode=True)
        resp = client.chat(_messages(), response_format=SCHEMA_SNIPPET)

        strict_attempts = [c for c in fake.calls if c.get("response_format") is not None]
        plain_attempts  = [c for c in fake.calls if c.get("response_format") is None]

        self.assertEqual(len(strict_attempts), 1,
                         f"expected exactly 1 strict-attempt; saw {len(strict_attempts)} ({fake.calls})")
        self.assertEqual(len(plain_attempts), 1,
                         f"expected exactly 1 degraded attempt; saw {len(plain_attempts)} ({fake.calls})")
        # Capability flipped sticky-false.
        self.assertIs(client._schema_supported, False,
                       f"_schema_supported={client._schema_supported} (expected False)")
        # Returned text came from the successful degraded call.
        self.assertIn('"final_answer"', resp.text)

    def test_capability_cache_sticky_false_skips_strict_attempt(self):
        client, fake = _wire_fake(reject_strict_mode=False)
        # Pretend prior probe already determined unsupportability.
        client._schema_supported = False
        resp = client.chat(_messages(), response_format=SCHEMA_SNIPPET)
        del resp

        strict_attempts = [c for c in fake.calls if c.get("response_format") is not None]
        plain_attempts  = [c for c in fake.calls if c.get("response_format") is None]
        self.assertEqual(strict_attempts, [],
                         f"strict disabled but sent attempts: {fake.calls}")
        self.assertEqual(len(plain_attempts), 1)
        # Cache stays False (no flip-up permitted by spec).
        self.assertIs(client._schema_supported, False)

    def test_successful_strict_completion_flips_cache_true_and_sticks(self):
        client, fake = _wire_fake(reject_strict_mode=False)
        # Initial state unknown -> probe succeeds -> cache flips TRUE.
        resp_a = client.chat(_messages(), response_format=SCHEMA_SNIPPET)
        self.assertIs(client._schema_supported, True)
        self.assertEqual(len(fake.calls), 1)
        all_strict_a = all(c.get("response_format") is not None for c in fake.calls)
        self.assertTrue(all_strict_a)

        # Second call must remain strictly constrained (cache sticks TRUE).
        resp_b = client.chat(_messages(), response_format=SCHEMA_SNIPPET)
        self.assertEqual(client._schema_supported, True)
        self.assertEqual(len(fake.calls), 2)
        all_strict_ab = all(c.get("response_format") is not None for c in fake.calls)
        self.assertTrue(all_strict_ab)
        del resp_a, resp_b


if __name__ == "__main__":
    unittest.main()
