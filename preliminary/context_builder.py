"""Convert a scenario record from ``bench/corpus_*.jsonl`` into an OpenAI-format
messages list ready to be POSTed to vLLM's ``/v1/chat/completions`` endpoint.

Mirrors the EXACT chat-template conventions used by live EvoGuard rollouts:

  * Exactly TWO messages per call: ``[{role:'system', ...}, {role:'user', ...}]``
    (see evoguard/agents/llm_agent.py and prompts.py).
  * System message contains role marker + DEFAULT_DEFENSE_SYSTEM body +
    AVAILABLE_TOOLS_JSON block + TASK line.
  * User message renders the action history as
        [turn N] you called TOOL(args)
        [turn N] tool result: OBSERVATION
        ...
        Decide your next step.
  * The LAST tool-result rendered into the user-message is the POISONED observation,
    so the model generates its next-step response right after exposure to injection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from evoguard.agents.prompts import build_system_prompt, render_history, DEFAULT_DEFENSE_SYSTEM
from evoguard.core.types import Action, Task, ToolSpec
from evoguard.envs.tool_parsing import parse_env_info

if __package__ in (None, ""):                                                   # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from preliminary.config import ExperimentConfig

logger = logging.getLogger("preliminary.context_builder")


# --------------------------------------------------------------------------- #
# Benchmark-side env_info cache (shared with miner.py)
# --------------------------------------------------------------------------- #
_BENCH_CACHE: dict[tuple[str, str], tuple[str, str]] = {}


def _load_benchmark_entry(domain: str, repo_root: str) -> tuple[str, str] | None:
    """Return ``(instruction_text, env_info_raw_text)`` for a given AgentDojo suite domain.

    Returns the FIRST distinct instruction encountered in the file -- this matches the original
    AgentDojo env loading semantics where each unique instruction becomes one task. For our purposes
    we only need ANY representative instruction+env_info pair per domain since all tasks in same domain
    share the SAME tool definitions; the actual task-specific instruction comes from scenario.goal_instruction.
    """

    cache_k = ("agentdojo:" + domain, repo_root)
    if cache_k in _BENCH_CACHE:
        return _BENCH_CACHE[cache_k]

    path = os.path.join(repo_root, "data", "toolsafe", "agentdojo-tragj", f"{domain}.json")
    if not os.path.isfile(path):
        logger.warning("benchmark file missing for domain %r at %s", domain, path)
        return None
    try:
        rec_list = json.load(open(path, "r", encoding="utf-8"))
    except Exception as exc:
        logger.warning("failed reading benchmark file %s : %s", path, exc)
        return None
    if isinstance(rec_list, list) and rec_list:
        first = rec_list[0]
        instr = first.get("instruction", "").strip()
        einfo = first.get("env_info", "")
        out = (instr, einfo) if instr else None
    else:
        out = None
    _BENCH_CACHE[cache_k] = out or ("", "")
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_tools_for_domain(domain: str, repo_root: str) -> list[ToolSpec]:
    """Parse env_info text of a given domain via shared helper used by real rollouts."""
    entry = _load_benchmark_entry(domain, repo_root)
    raw_env_info = entry[1] if entry else ""
    try:
        tools = parse_env_info(raw_env_info)
    except Exception as exc:                                                    # pragma: no cover - defensive
        logger.warning("parse_env_info failed for domain=%s (%s); falling back to empty tools", domain, exc)
        tools = []
    # Deduplicate preserving order to avoid double-listing in system prompt JSON blob.
    seen_names = set()
    deduped: list[ToolSpec] = []
    for t in tools:
        if t.name in seen_names:
            continue
        seen_names.add(t.name)
        deduped.append(t)
    return deduped


def reconstruct_actions_from_prefix(
    prefix_dicts: list[dict[str, Any]],
) -> list[Action]:
    """Rebuild Action dataclass instances assigning sequential turn indices based on position."""
    actions: list[Action] = []
    for idx, d in enumerate(prefix_dicts):
        tc_dict = d.get("tool_call")
        new_action_dct = {
            "turn": int(idx),
            "thought": str(d.get("thought", "")),
            "tool_call": tc_dict,
            "observation": str(d.get("observation", "")),
            "final_answer": str(d.get("final_answer", "") or ""),
            "metadata": dict(d.get("metadata") or {}),
        }
        actions.append(Action.from_dict(new_action_dct))
    return actions


def build_messages_for_scenario(scenario: dict[str, Any], repo_root: str) -> list[dict[str, str]]:
    """Assemble OpenAI-format messages list ready for vLLM chat-completions endpoint."""

    goal_instruction = str(scenario.get("goal_instruction", ""))
    domain = str(scenario.get("domain", "?"))
    T_i = int(scenario.get("injection_target_turn_index", -1))
    prefix_raw = scenario.get("context_prefix_actions") or []

    if T_i < 0 or T_i >= len(prefix_raw):
        raise ValueError(
            f"scenario {scenario.get('scenario_id')} has inconsistent "
            f"T_i={T_i} vs prefix_len={len(prefix_raw)}"
        )

    tools_list = load_tools_for_domain(domain, repo_root)

    task_obj = Task(
        task_id=str(scenario.get("task_id", f"agentdojo:{domain}:unknown")),
        instruction=goal_instruction,
        suite=domain,
        dataset="agentdojo",
        tool_names=[t.name for t in tools_list],
        metadata={},
    )

    system_str = build_system_prompt(task=task_obj, tools=tools_list, custom=None)

    actions_for_history = reconstruct_actions_from_prefix(prefix_raw[: T_i + 1])
    user_str = render_history(actions_for_history)

    messages_out = [
        {"role": "system", "content": system_str},
        {"role": "user", "content": user_str},
    ]
    return messages_out


# --------------------------------------------------------------------------- #
# Self-test entrypoint invoked only when run directly without args.
# --------------------------------------------------------------------------- #
def _self_test(repo_root_arg: str | None = None):
    """Smoke-test against the bench dataset producing sample assembled prompt previews."""

    repo_root = repo_root_arg or os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    bench_dir = os.path.join(repo_root, "bench")
    import glob
    files = sorted(glob.glob(os.path.join(bench_dir, "corpus_*.jsonl")))
    print(f"=== self-test on {len(files)} corpus files ===")
    for fp in files[:2]:
        n_total_chars_sys = 0
        n_total_chars_user = 0
        max_n_turns_rendered = 0
        last_observation_tail_snippet = "<none>"
        samples_inspected = 0
        bucket_label_seen = set()
        with open(fp, "r", encoding="utf-8") as fin:
            for i, ln in enumerate(fin):
                scen = json.loads(ln)
                msgs = build_messages_for_scenario(scen, repo_root)
                assert len(msgs) == 2
                assert {m["role"] for m in msgs} == {"system", "user"}
                assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
                sys_c = msgs[0]["content"]
                usr_c = msgs[1]["content"]
                assert "[[evoguard:role=defense_agent]]" in sys_c, \
                    "missing defense-agent marker in system msg"
                assert "AVAILABLE_TOOLS_JSON:" in sys_c, \
                    "missing AVAILABLE_TOOLS_JSON sentinel in system msg"
                assert usr_c.endswith("Decide your next step."), \
                    "user-msg must end with 'Decide your next step.'"
                n_total_chars_sys += len(sys_c)
                n_total_chars_user += len(usr_c)
                # Count how many turns were actually emitted in user history rendering.
                turn_count_in_usr = usr_c.count("[turn ")
                max_n_turns_rendered = max(max_n_turns_rendered, turn_count_in_usr)
                # Inspect tail snippet of user content to confirm poisoned observation landed there.
                last_idx_of_tool_result_marker = usr_c.rfind("tool result:")
                if last_idx_of_tool_result_marker != -1:
                    last_observation_tail_snippet = usr_c[last_idx_of_tool_result_marker:last_idx_of_tool_result_marker + 200]
                bucket_label_seen.add(scen.get("bucket"))
                samples_inspected += 1
                if i >= 4:
                    break   # inspect up-to-5 scenarios per file for quick smoke check
        bn = os.path.basename(fp).replace("corpus_", "").replace(".jsonl", "")
        avg_sys = n_total_chars_sys / max(samples_inspected, 1)
        avg_usr = n_total_chars_user / max(samples_inspected, 1)
        print(f"\n[{bn}] inspected={samples_inspected} buckets={bucket_label_seen}")
        print(f"     avg(sys_msg_len)={avg_sys:.0f} chars ; avg(user_msg_len)={avg_usr:.0f} chars")
        print(f"     max(turns_emitted_per_scenario)={max_n_turns_rendered}")
        print(f"     sample_last_poisoned_observation_excerpt:\n       {last_observation_tail_snippet!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()
    _self_test(args.repo_root)
