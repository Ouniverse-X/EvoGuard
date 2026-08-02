"""Re-classify DROPPED rows in a completed kld run's raw_kl.jsonl using the fixed judge.

Reads raw_kl.jsonl (which stores _generated_action_full + _injection_body_full),
re-runs the VLLMJudgeClient on every DROPPED row, rewrites kl_per_scenario.csv with
corrected buckets. KL values are preserved (no model re-run needed).

Usage:
    python -m preliminary.kld_reclassify --run-dir <dir> [--judge-base-url ...] [--judge-model ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from preliminary.kld_harness import VLLMJudgeClient, CSV_COLUMN_ORDER
from preliminary.kld_classifier import BUCKET_ATTACK_SUCCESS, BUCKET_ATTACK_FAIL

logger = logging.getLogger("preliminary.kld_reclassify")


def _load_instruction_lookup(data_path: str) -> dict[tuple, str]:
    """Build (trajectory_id, segment_id) -> instruction lookup from workspace.json."""
    recs = json.load(open(data_path, "r", encoding="utf-8"))
    out = {}
    for r in recs:
        key = (r.get("id-interaction"), r.get("id-segment"))
        out[key] = str(r.get("instruction", ""))
    return out


def reclassify(run_dir: str, *, data_path: str, judge_base_url: str, judge_model: str) -> dict:
    jsonl_path = os.path.join(run_dir, "raw_kl.jsonl")
    csv_path = os.path.join(run_dir, "kl_per_scenario.csv")
    instr_lookup = _load_instruction_lookup(data_path)
    judge = VLLMJudgeClient(judge_base_url, judge_model)

    rows_out = []
    n_reclassified = n_still_dropped = n_skipped = 0
    for ln in open(jsonl_path, "r", encoding="utf-8"):
        r = json.loads(ln)
        if r.get("bucket") != "DROPPED":
            rows_out.append(r)
            n_skipped += 1
            continue
        # Re-judge with full text.
        instr = instr_lookup.get((r.get("trajectory_id"), r.get("segment_id")), "")
        inj_full = r.get("_injection_body_full", "")
        gen_full = r.get("_generated_action_full", "")
        if not inj_full or not gen_full:
            n_still_dropped += 1
            rows_out.append(r)
            continue
        complies, reason = judge.judge_compliance(instr, inj_full, gen_full)
        if complies is None:
            n_still_dropped += 1
            rows_out.append(r)
            continue
        r["bucket"] = BUCKET_ATTACK_SUCCESS if complies else BUCKET_ATTACK_FAIL
        r["judge_used"] = True
        r["complies"] = complies
        r["dropped"] = False
        n_reclassified += 1
        rows_out.append(r)

    # Rewrite CSV.
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMN_ORDER, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMN_ORDER})

    summary = {"n_reclassified": n_reclassified, "n_still_dropped": n_still_dropped,
               "n_skipped_non_dropped": n_skipped}
    json.dump(summary, open(os.path.join(run_dir, "reclassify_summary.json"), "w"), indent=2)
    logger.info("reclassify done: %s", summary)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-path", default="data/toolsafe/agentdojo-tragj/workspace.json")
    ap.add_argument("--judge-base-url", default="http://localhost:8002/v1")
    ap.add_argument("--judge-model", default="llama3-8b-judge")
    args = ap.parse_args()
    reclassify(args.run_dir, data_path=args.data_path,
               judge_base_url=args.judge_base_url, judge_model=args.judge_model)

if __name__ == "__main__":
    main()
