"""KL divergence measurement primitives for the tool-return surprise experiment.

Pure-Python helpers for:
  * splitting the toolsafe workspace.json ``history`` field to isolate the last observation
  * building C_pre / C_post message lists (differing only in the last observation content)
  * computing D_KL(P_post || P_pre) in nats from logits

Heavy torch/transformers imports are kept INSIDE the functions that need them so the
module imports cleanly for offline unit-testing of the KL arithmetic and string parsing.

See ``docs/superpowers/specs/2026-08-02-preliminary-ipi-tool-return-kld-surprise-design.md``.
"""
from __future__ import annotations

from typing import Any

NEUTRAL_DEFAULT_OBSERVATION = "Done."
_OBSERVATION_MARKER = "Observation:"


def split_last_observation(history_text: str) -> tuple[str, str]:
    """Split ``history_text`` into ``(prefix, last_observation_content)``.

    The toolsafe history format is alternating blocks each ending with
    ``Observation: <text>``. The LAST occurrence of the marker starts the
    most-recent observation (the one under study). ``prefix`` includes the
    marker itself; ``last_observation_content`` is everything after it.
    If no marker is found, returns ``("", history_text)`` (degenerate).
    """
    idx = history_text.rfind(_OBSERVATION_MARKER)
    if idx < 0:
        return ("", history_text)
    prefix = history_text[: idx + len(_OBSERVATION_MARKER)]
    obs_content = history_text[idx + len(_OBSERVATION_MARKER):]
    return (prefix, obs_content)


def replace_last_observation(history_text: str, replacement: str) -> str:
    """Return ``history_text`` with the most-recent observation content swapped for ``replacement``."""
    prefix, _ = split_last_observation(history_text)
    if not prefix:
        return replacement
    return prefix + " " + replacement


def compute_kl_from_logits(logits_post, logits_pre, *, eps: float = 1e-12) -> float:
    """Compute ``D_KL(P_post || P_pre)`` in nats from raw vocab logits.

    ``logits_post`` / ``logits_pre`` may be torch tensors or array-likes; both are
    converted to float64 numpy arrays. Numerically-stable softmax via subtractive max.
    """
    import numpy as np

    def _to_np(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype="float64")

    lp = _to_np(logits_post)
    lq = _to_np(logits_pre)
    p = np.exp(lp - lp.max()); p /= p.sum()
    q = np.exp(lq - lq.max()); q /= q.sum()
    p_safe = p.clip(min=eps)
    q_safe = q.clip(min=eps)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def build_messages_for_toolsafe_segment(
    record: dict[str, Any],
    *,
    neutral_default: str = NEUTRAL_DEFAULT_OBSERVATION,
    domain: str = "workspace",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build (msgs_post, msgs_pre) for a toolsafe workspace.json record.

    ``msgs_post`` renders the history verbatim (real most-recent observation).
    ``msgs_pre`` replaces the most-recent observation content with ``neutral_default``
    while leaving all earlier observations and the system message untouched.

    Returns two OpenAI-format message lists ``[system, user]`` ready for
    ``tokenizer.apply_chat_template(add_generation_prompt=True)``.
    """
    from evoguard.agents.prompts import build_system_prompt
    from evoguard.core.types import Task
    from evoguard.envs.tool_parsing import parse_env_info

    instruction = str(record.get("instruction", ""))
    env_info = str(record.get("env_info", ""))
    history = str(record.get("history", ""))

    tools = parse_env_info(env_info)
    task = Task(
        task_id=str(record.get("id-interaction", "?")),
        instruction=instruction,
        suite=domain,
        dataset="agentdojo",
        tool_names=[t.name for t in tools],
        metadata={},
    )
    system_str = build_system_prompt(task=task, tools=tools, custom=None)

    if history:
        user_post = history + "\nDecide your next step."
        user_pre = replace_last_observation(history, neutral_default) + "\nDecide your next step."
    else:
        # No prior observation (segment 0); C_pre == C_post -> KL will be 0 (degenerate, skipped upstream).
        user_post = "Decide your next step."
        user_pre = user_post

    msgs_post = [{"role": "system", "content": system_str}, {"role": "user", "content": user_post}]
    msgs_pre = [{"role": "system", "content": system_str}, {"role": "user", "content": user_pre}]
    return msgs_post, msgs_pre
