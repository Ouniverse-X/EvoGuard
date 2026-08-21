"""Defense-agent wrapper enabling vLLM-side logprob capture (spec §2026-07-28 §1).

The captured token logprobs are NOT consumed by the GRPO loss directly -- local
teacher-forced forward recompute owns the gradient-bearing logp path. They serve
purely as a sanity-check telemetry channel: if remote-server top-token LPs drift
too far from locally-recomputed ones we flag the sample as suspicious and skip
its batch contribution to avoid back-propagating through a corrupted rollout.

Two public surfaces:

* :func:`parse_lp_list_from_raw` -- pure-Python extractor walking the OpenAI-style
  ``raw["choices"][0]["logprobs"]["content"][k].logprob`` array into ``list[float]``.
  Defensive against missing keys / malformed shapes returning empty list rather
  than raising so callers can treat absence as "no telemetry" without try/except.

* :class:`LogpAgent` -- thin :class:`evoguard.agents.base.DefenseAgent` wrapper
  that flips on the underlying client's request-level ``logprobs=True`` via the
  existing :attr:`LLMConfig.extra` merge hook inside
  :meth:`OpenAIClient._merge_extra`, then parses each returned response and stashes
  parsed floats onto every emitted Action's ``metadata['token_lps']`` slot.

Identifier-length discipline per MEMORY.md lora_layer_probe.md summary kept tight.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from evoguard.agents.base import DefenseAgent
from evoguard.agents.llm_agent import LLMDefenseAgent
from evoguard.config import DefenseConfig
from evoguard.core.types import Action, Task, ToolSpec


# --------------------------------------------------------------------------- #
# Pure helpers                                                                #
# --------------------------------------------------------------------------- #
def parse_lp_list_from_raw(raw_payload: Optional[dict[str, Any]]) -> list[float]:
    """Extract ordered list of selected-token logprobs from an OpenAI/vLLM raw payload.

    Walks ``choices[0].logprobs.content[*].logprob`` defensively -- returns []
    whenever any segment of the chain is absent or malformed so callers may use a
    simple truthiness check downstream instead of nested try/except blocks.
    """
    if not isinstance(raw_payload, dict):
        return []
    choices = raw_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return []
    lp_block = first_choice.get("logprobs")
    if not isinstance(lp_block, dict):
        return []
    content_arr = lp_block.get("content")
    if not isinstance(content_arr, list):
        return []
    out_lps: list[float] = []
    for entry in content_arr:
        if not isinstance(entry, dict):
            continue
        val_raw = entry.get("logprob")
        try:
            out_lps.append(float(val_raw))
        except Exception:
            continue                                            # noqa: BLE001
    return out_lps


# --------------------------------------------------------------------------- #
# Agent wrapper                                                               #
# --------------------------------------------------------------------------- #
_LOGPROBS_EXTRA_KEY_VALS: dict[str, Any] = {"logprobs": True,
                                              "top_logprobs": 0}


class LogpAgent(DefenseAgent):
    """Wraps :class:`LLMDefenseAgent` flipping on server-side logprobs capture."""

    name = "online_logp_defense"

    def __init__(self, inner_agent: LLMDefenseAgent) -> None:
        self.inner = inner_agent

        # Mutate the wrapped agent's underlying config.extra dict in-place so the
        # next chat() call merges our keys into outgoing SDK kwargs. Idempotent:
        # repeated wrapping leaves extra unchanged after first pass.
        cfg_llm_extra_ref = getattr(self.inner.config.llm, "extra", None)
        if not isinstance(cfg_llm_extra_ref, dict):
            # Some configs default extra=None; replace with new dict we own.
            self.inner.config.llm.extra = dict(_LOGPROBS_EXTRA_KEY_VALS)
        else:
            for k_extras_k, v_extras_v in _LOGPROBS_EXTRA_KEY_VALS.items():
                cfg_llm_extra_ref.setdefault(k_extras_k, v_extras_v)

        self.last_token_lps: list[float] = []

    def decide(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        history: Sequence[Action],
    ) -> Action:
        action_obj = self.inner.decide(task, tools, history)
        # Recover this turn's raw response from inner's last-call cache when available;
        # fall back to [] silently otherwise since telemetry is best-effort only.
        last_resp_raw = getattr(self.inner._client, "_last_response_raw", None)
        if last_resp_raw is None:
            # Try alternative attribute names used historically by various clients.
            alt_attr_names = ("last_response", "_last_response",
                              "last_response_raw")
            for nm_alt in alt_attr_names:
                cand_val = getattr(self.inner._client, nm_alt, None)
                cand_raw_dict = getattr(cand_val, "raw", None)
                if cand_raw_dict is not None:
                    last_resp_raw = cand_raw_dict
                    break
        extracted_lps = parse_lp_list_from_raw(last_resp_raw)
        self.last_token_lps = extracted_lps
        try:
            action_obj.metadata["token_lps"] = extracted_lps
        except Exception:
            # metadata might be frozen/None in exotic subclasses; never fatal here.
            pass                                                    # noqa: BLE001
        return action_obj

    @property
    def config(self) -> DefenseConfig:
        return self.inner.config


__all__: list[str] = [
    "LogpAgent",
    "parse_lp_list_from_raw",
]
