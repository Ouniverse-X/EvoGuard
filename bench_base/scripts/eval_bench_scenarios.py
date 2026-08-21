"""Evaluate ALL curated bench scenarios from bench_base/scenarios/bucket_*.jsonl.

Uses the existing evaluate_bench_scenario() function from eval_full_rollout.py
against the v8_contrastive trained defender (LoRA adapter chain on vLLM :8000).

Usage:
    python bench_base/scripts/eval_bench_scenarios.py \
        --out bench_base/diagnostics/bench_scenarios_v8_full.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evoguard.agents.llm_agent import LLMDefenseAgent
from evoguard.config import LLMConfig, DefenseConfig
from evoguard.envs.tool_parsing import parse_env_info
from evoguard.core.types import Task

from bench_base.scripts.eval_full_rollout import (
    build_rounds_indexes,
    build_goal_to_clean,
    load_env_info_map,
    make_trajectory_judge,
    evaluate_bench_scenario,
    _aggregate,
    _print_table,
    ROUNDS_ROOT,
)

SCENARIOS_ROOT = REPO_ROOT / "bench_base" / "scenarios"
OUT_DIR = REPO_ROOT / "bench_base" / "diagnostics"

# This evaluator is the POSITIVE (attacked) arm: every scenario it reads must
# carry a payload. `bucket_clean.jsonl` (origin_mode=benign_arm) also matches the
# bucket_*.jsonl glob but has no payload, so without this whitelist all 228
# benign rows would come back as `_skip(no_payload)` and pollute the summary.
# The negative arm is evaluated by eval_bench_clean_arm.py instead.
DELTA_BUCKET_LABELS = ("imm", "d1", "d2", "d3", "d4")

DEFENDER_URL = os.environ.get("BE_DEFENDER_URL", "http://localhost:8000/v1")
DEFENDER_MODEL = os.environ.get("BE_DEFENDER_MODEL", "qwen2.5-7b-it")
JUDGE_URL = os.environ.get("BE_JUDGE_URL", "http://localhost:8000/v1")
JUDGE_MODEL = os.environ.get("BE_JUDGE_MODEL", "qwen2.5-7b-it")


def load_bench_scenarios() -> dict[str, list[dict]]:
    """Load the Δ-bucket (attacked) scenarios, grouped by bucket."""
    buckets: dict[str, list[dict]] = {}
    for jf in sorted(SCENARIOS_ROOT.glob("bucket_*.jsonl")):
        bucket_name = jf.stem.replace("bucket_", "")
        if bucket_name not in DELTA_BUCKET_LABELS:
            continue
        records = []
        with open(jf, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                records.append(json.loads(ln))
        buckets[bucket_name] = records
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR / "bench_scenarios_v8_full.json"))
    parser.add_argument("--buckets", nargs="*", default=None,
                        help="Subset of buckets to eval (default: all)")
    args = parser.parse_args()

    t0 = time.time()

    print("[eval] building rounds indexes for clean-twin lookup...", flush=True)
    task_to_clean, rec_to_clean, _ = build_rounds_indexes(ROUNDS_ROOT)
    goal_to_clean = build_goal_to_clean(task_to_clean, ROUNDS_ROOT)
    env_info_map = load_env_info_map()
    print(f"[eval] task_to_clean={len(task_to_clean)} rec_to_clean={len(rec_to_clean)} "
          f"goal_to_clean={len(goal_to_clean)} env_info={len(env_info_map)}", flush=True)

    agent_llm = LLMConfig(
        backend="openai", base_url=DEFENDER_URL, api_key="EMPTY",
        model=DEFENDER_MODEL, temperature=0.0, max_tokens=256,
    )
    agent = LLMDefenseAgent(DefenseConfig(llm=agent_llm, system_prompt=None))

    try:
        judge_fn = make_trajectory_judge(JUDGE_URL, JUDGE_MODEL, api_key="EMPTY")
        ok, info = judge_fn([{"thought": "test", "tool_call": None,
                              "observation": "x", "final_answer": "done"}], "test", "test")
        print(f"[eval] judge wired: {JUDGE_URL} (smoke={info})", flush=True)
    except Exception as exc:
        judge_fn = None
        print(f"[eval] WARN judge failed: {exc}", flush=True)

    bench_data = load_bench_scenarios()
    total = sum(len(v) for v in bench_data.values())
    print(f"[eval] loaded {total} scenarios across {len(bench_data)} buckets", flush=True)

    if args.buckets:
        bench_data = {k: v for k, v in bench_data.items() if k in args.buckets}

    all_results = []
    per_bucket_summary = {}

    for bucket, scenarios in sorted(bench_data.items()):
        print(f"\n[eval] bucket={bucket} n={len(scenarios)}", flush=True)
        bucket_results = []
        for i, scenario in enumerate(scenarios):
            res = evaluate_bench_scenario(
                scenario, agent, judge_fn,
                rec_to_clean, task_to_clean, goal_to_clean, env_info_map,
            )
            bucket_results.append(res)
            if (i + 1) % 10 == 0 or i + 1 == len(scenarios):
                n_eval = sum(1 for r in bucket_results if r.get("status") == "evaluated")
                n_succ = sum(1 for r in bucket_results
                             if r.get("status") == "evaluated" and r.get("success") is True)
                print(f"  [{bucket}] {i+1}/{len(scenarios)} done "
                      f"(evaluated={n_eval}, judge_success={n_succ})", flush=True)

        all_results.extend(bucket_results)
        per_bucket_summary[bucket] = _aggregate(bucket_results, bucket)
        s = per_bucket_summary[bucket]
        print(f"  [{bucket}] evaluated={s['n_evaluated']} skipped={s['n_skipped']} "
              f"ASR={s['asr_measured']:.2%}", flush=True)

    elapsed = time.time() - t0
    summary = {
        "evaluator": "bench_scenarios_full_eval",
        "defender": f"{DEFENDER_MODEL} @ {DEFENDER_URL}",
        "judge": f"{JUDGE_MODEL} @ {JUDGE_URL}",
        "source": "bench_base/scenarios/bucket_*.jsonl",
        "elapsed_seconds": round(elapsed, 1),
        "n_total": len(all_results),
        "per_bucket": per_bucket_summary,
        "all_results": all_results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n[eval] DONE in {elapsed:.1f}s -> {args.out}", flush=True)
    _print_table(per_bucket_summary)


if __name__ == "__main__":
    main()
