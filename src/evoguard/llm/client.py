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


@dataclass
class OpenAICompatibleChatClient:
    """OpenAI-compatible chat completions JSON client for attack generation.

    Required environment:
    - `EVOGUARD_ATTACK_API_KEY`
    - `EVOGUARD_ATTACK_MODEL`

    Optional environment:
    - `EVOGUARD_ATTACK_BASE_URL`
    """

    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 90.0
    temperature: float = 0.8
    max_tokens: int = 1400

    def generate_json(self, messages: list[dict[str, str]], *, schema_name: str) -> dict[str, Any]:
        api_key = self.api_key or os.environ.get("EVOGUARD_ATTACK_API_KEY")
        model = self.model or os.environ.get("EVOGUARD_ATTACK_MODEL")
        base_url = self.base_url or os.environ.get("EVOGUARD_ATTACK_BASE_URL")
        if not api_key:
            raise LLMClientError("EVOGUARD_ATTACK_API_KEY is required for OpenAICompatibleChatClient")
        if not model:
            raise LLMClientError("EVOGUARD_ATTACK_MODEL is required for OpenAICompatibleChatClient")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMClientError("OpenAICompatibleChatClient requires the openai Python package") from exc

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise LLMClientError(f"{schema_name} request failed: {exc}") from exc

        text = response.choices[0].message.content or ""
        try:
            parsed = json.loads(_extract_json_object_text(text))
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


@dataclass
class LocalTransformersJSONClient:
    """Local Hugging Face/Transformers chat model client for JSON generation.

    This keeps red-team generation local on the training server. Heavy
    dependencies are imported lazily so the core package remains lightweight in
    CPU-only test environments.
    """

    model_name_or_path: str
    device_map: str | dict[str, Any] = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 768
    temperature: float = 0.8
    top_p: float = 0.9
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional stack.
            raise LLMClientError(
                "LocalTransformersJSONClient requires torch and transformers. "
                "Install the server training dependencies or use the LLaMaFactory conda env."
            ) from exc

        dtype = self.torch_dtype
        if dtype == "auto":
            model_dtype = "auto"
        else:
            model_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=model_dtype,
            device_map=self.device_map,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_json(self, messages: list[dict[str, str]], *, schema_name: str) -> dict[str, Any]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        try:
            parsed = json.loads(_extract_json_object_text(text))
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"{schema_name} returned non-JSON text: {text[:500]}") from exc
        if not isinstance(parsed, dict):
            raise LLMClientError(f"{schema_name} returned JSON {type(parsed).__name__}, expected object")
        return parsed


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


def _extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object start found", stripped, 0)
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    raise json.JSONDecodeError("No complete JSON object found", stripped, start)
