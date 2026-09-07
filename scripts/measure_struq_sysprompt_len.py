"""Break the turn-0 StruQ prompt into its parts, in StruQ tokens.

The turn-0 number from ``measure_struq_prompt_len.py`` already overflows 2048 on
half the tasks with an EMPTY observation history, so no ``max_turns`` reduction
can fix blocker 3. This tells us which part is responsible.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evoguard.agents.prompts import build_system_prompt
from evoguard.config import EnvConfig
from evoguard.envs import build_env
from evoguard.llm.mock_client import MockClient

STRUQ_DIR = "/root/paddlejob/workspace/yangxiao/models/struQ/llama-7b_Spcl"


def main(argv: list[str]) -> int:
    dataset = argv[1] if len(argv) > 1 else "agentdojo_latent"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(STRUQ_DIR, use_fast=False)
    env = build_env(EnvConfig(dataset=dataset, data_root="data"),
                    executor=MockClient())

    rows = []
    for task in env.get_tasks():
        tools = env.get_tools(task)
        sys_prompt = build_system_prompt(task, tools, None)
        n_sys = len(tok(sys_prompt).input_ids)
        n_instr = len(tok(task.instruction).input_ids)
        rows.append((task.suite, len(tools), n_sys, n_instr, task.task_id))

    rows.sort()
    print(f"{'suite':<12} {'tools':>5} {'sys_tok':>8} {'instr_tok':>9}")
    for suite, n_tools, n_sys, n_instr, _ in rows:
        print(f"{suite:<12} {n_tools:>5} {n_sys:>8} {n_instr:>9}")

    by_suite: dict[str, list[int]] = {}
    for suite, _, n_sys, _, _ in rows:
        by_suite.setdefault(suite, []).append(n_sys)
    print()
    print(f"{'suite':<12} {'n':>4} {'min':>6} {'median':>7} {'max':>6}")
    for suite in sorted(by_suite):
        vals = sorted(by_suite[suite])
        print(f"{suite:<12} {len(vals):>4} {vals[0]:>6} "
              f"{vals[len(vals)//2]:>7} {vals[-1]:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
