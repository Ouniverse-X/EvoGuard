"""Measure StruQ prompt length against the checkpoint's 2048-token window.

Blocker 3 of ``docs/struq_arm_blockers.md``: ``llama-7b_Spcl`` has
``max_position_embeddings: 2048`` while the other probe arms run 16384. This
script renders :func:`evoguard.agents.struq_agent.build_struq_prompt` at every
turn depth of an ALREADY-COLLECTED replay and reports, per turn index, how many
prompts fit -- which is what picks ``defense.max_turns`` for the StruQ configs.

It reads the base arm's records, not the StruQ arm's: the point is to choose the
budget BEFORE spending GPU hours, and the base arm's trajectories are the closest
available proxy for how long an observation is on these tasks. StruQ's own
trajectories will be shorter (it cannot act), so this is a conservative bound in
the direction that matters.

Usage::

    python scripts/measure_struq_prompt_len.py agentdojo_latent \
        rounds/replay_test_adjlatent_base [banking,slack]
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evoguard.agents.struq_agent import build_struq_prompt
from evoguard.config import EnvConfig
from evoguard.core.types import Action, ToolCall
from evoguard.envs import build_env
from evoguard.llm.mock_client import MockClient

STRUQ_DIR = "/root/paddlejob/workspace/yangxiao/models/struQ/llama-7b_Spcl"
LIMIT = 2048


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    dataset, replay_dir = argv[1], argv[2]
    suites = [s for s in (argv[3].split(",") if len(argv) > 3 else []) if s]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(STRUQ_DIR, use_fast=False)

    env = build_env(EnvConfig(dataset=dataset, data_root="data", suites=suites),
                    executor=MockClient())
    tasks = {t.task_id: t for t in env.get_tasks()}
    tools = {tid: env.get_tools(t) for tid, t in tasks.items()}

    records = []
    with open(os.path.join(replay_dir, "records.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # turn index -> token counts for the prompt the agent WOULD see at that turn
    by_turn: dict[int, list[int]] = {}
    n_used = 0
    for rec in records:
        if rec.get("kind") == "clean":
            continue
        task = tasks.get(rec.get("task_id", ""))
        if task is None:
            continue
        n_used += 1
        steps = (rec.get("trajectory") or {}).get("actions") or []
        history: list[Action] = []
        for i in range(len(steps) + 1):
            prompt = build_struq_prompt(task, tools[task.task_id], history)
            by_turn.setdefault(i, []).append(len(tok(prompt).input_ids))
            if i >= len(steps):
                break
            st = steps[i]
            call = st.get("tool_call") or {}
            history.append(Action(
                turn=st.get("turn", i),
                thought=st.get("thought", "") or "",
                observation=st.get("observation", "") or "",
                final_answer=st.get("final_answer", "") or "",
                tool_call=(ToolCall(name=call.get("name", ""),
                                    arguments=call.get("arguments") or {})
                           if call.get("name") else None),
            ))

    print(f"dataset={dataset} attacked records used: {n_used}")
    print(f"{'turn':>5} {'n':>5} {'min':>6} {'median':>7} {'max':>7} "
          f"{'fits<=2048':>12}")
    for turn in sorted(by_turn):
        vals = sorted(by_turn[turn])
        fits = sum(1 for v in vals if v <= LIMIT)
        print(f"{turn:>5} {len(vals):>5} {vals[0]:>6} {vals[len(vals)//2]:>7} "
              f"{vals[-1]:>7} {fits:>7}/{len(vals):<4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
