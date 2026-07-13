"""Small JSON-oriented LLM client layer.

The production client intentionally uses only the standard library so the core
research package stays lightweight. Tests can use `ScriptedLLMClient` to avoid
network calls while exercising the same parsing and validation paths.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class LLMClientError(RuntimeError):
    """Raised when an LLM backend cannot produce a valid response."""


class LLMClient(Protocol):
    def generate_json(self, messages: list[dict[str, str]], *, schema_name: str) -> dict[str, Any]:
        """Generate one JSON object for the supplied chat-style messages."""


@dataclass
class OpenAIResponsesClient:
    """Minimal OpenAI Responses API JSON client.

    Required environment:
    - `OPENAI_API_KEY`
    - `EVOGUARD_LLM_MODEL`

    Optional environment:
    - `EVOGUARD_OPENAI_BASE_URL`, defaults to `https://api.openai.com/v1`
    """

    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 60.0

    def generate_json(self, messages: list[dict[str, str]], *, schema_name: str) -> dict[str, Any]:
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        model = self.model or os.environ.get("EVOGUARD_LLM_MODEL")
        if not api_key:
            raise LLMClientError("OPENAI_API_KEY is required for OpenAIResponsesClient")
        if not model:
            raise LLMClientError("EVOGUARD_LLM_MODEL is required for OpenAIResponsesClient")

        payload = {
            "model": model,
            "input": _messages_to_input(messages),
            "text": {"format": {"type": "json_object"}},
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"{(self.base_url or os.environ.get('EVOGUARD_OPENAI_BASE_URL') or 'https://api.openai.com/v1').rstrip('/')}/responses"
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMClientError(f"{schema_name} request failed: {exc}") from exc

        text = _extract_response_text(body)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"{schema_name} returned non-JSON text: {text[:200]}") from exc
        if not isinstance(parsed, dict):
            raise LLMClientError(f"{schema_name} returned JSON {type(parsed).__name__}, expected object")
        return parsed


class ScriptedLLMClient:
    """Deterministic client for tests and local dry runs."""

    def __init__(self, responses: Iterable[Mapping[str, Any]]) -> None:
        self._responses = [dict(response) for response in responses]
        self.requests: list[tuple[str, list[dict[str, str]]]] = []

    def generate_json(self, messages: list[dict[str, str]], *, schema_name: str) -> dict[str, Any]:
        self.requests.append((schema_name, messages))
        if not self._responses:
            raise LLMClientError(f"No scripted response left for {schema_name}")
        return self._responses.pop(0)


def _messages_to_input(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "role": message["role"],
            "content": [{"type": "input_text", "text": message["content"]}],
        }
        for message in messages
    ]


def _extract_response_text(body: Mapping[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return str(body["output_text"])
    chunks: list[str] = []
    for item in body.get("output", []) if isinstance(body.get("output"), list) else []:
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if isinstance(content, Mapping) and isinstance(content.get("text"), str):
                chunks.append(str(content["text"]))
    if chunks:
        return "".join(chunks)
    raise LLMClientError("Responses API body did not contain output text")
