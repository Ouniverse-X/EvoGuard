#!/usr/bin/env python
"""Run EvoGuard MVP ablation experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.evaluation.ablations import run_ablation_suite
from evoguard.utils.logging import append_jsonl


def main() -> None:
    results = run_ablation_suite(train_rounds=3)
    rows = [
        {
            "ablation": result.name,
            "description": result.description,
            "train_rounds": result.train_rounds,
            "train_attack_split": result.train_attack_split,
            "eval_attack_split": result.eval_attack_split,
            "policy_updates": result.policy_updates,
            "update_stats": result.update_stats,
            "metrics": result.metrics,
        }
        for result in results
    ]
    append_jsonl(Path("outputs/logs/ablation_results.jsonl"), rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
