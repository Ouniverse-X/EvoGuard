"""LLM client abstraction and backends.

Use :func:`build_client` to construct the right backend from an
:class:`~evoguard.config.LLMConfig`.
"""

from __future__ import annotations

from evoguard.config import LLMConfig
from evoguard.llm.base import LLMClient, LLMResponse


def build_client(config: LLMConfig, *, seed: int = 0) -> LLMClient:
    """Instantiate an :class:`LLMClient` for ``config.backend``."""

    backend = (config.backend or "openai").lower()
    if backend == "mock":
        from evoguard.llm.mock_client import MockClient

        return MockClient(config, seed=seed)
    if backend in ("openai", "vllm"):
        from evoguard.llm.openai_client import OpenAIClient

        return OpenAIClient(config)
    if backend == "qianfan":
        # Baidu QianFan v2 gateway -- uses custom appid/Bearer headers so it
        # needs its own HTTP client rather than the openai SDK.
        from evoguard.llm.qianfan_client import QianFanClient

        return QianFanClient(config)
    raise ValueError(f"Unknown LLM backend: {config.backend!r}")


__all__ = ["LLMClient", "LLMResponse", "build_client"]
