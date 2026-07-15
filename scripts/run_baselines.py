#!/usr/bin/env python
"""Run baseline comparisons for the text-tool MVP."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.evaluation.baselines import run_baseline_suite
from evoguard.utils.logging import append_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rounds", type=int, default=3)
    parser.add_argument("--eval-attack-split", default="heldout")
    args = parser.parse_args()

    results = run_baseline_suite(train_rounds=args.train_rounds, eval_attack_split=args.eval_attack_split)
    rows = [
        {
            "baseline": result.name,
            "train_rounds": result.train_rounds,
            "train_attack_split": result.train_attack_split,
            "eval_attack_split": result.eval_attack_split,
            "policy_updates": result.policy_updates,
            "update_stats": result.update_stats or {},
            "metrics": result.metrics,
        }
        for result in results
    ]
    append_jsonl(Path("outputs/logs/baseline_comparison.jsonl"), rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
