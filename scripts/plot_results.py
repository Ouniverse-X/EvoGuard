#!/usr/bin/env python
"""Generate paper tables and SVG figures from EvoGuard JSONL logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.evaluation.plots import write_report_artifacts


def main() -> None:
    outputs = write_report_artifacts(
        baseline_log=Path("outputs/logs/baseline_comparison.jsonl"),
        training_log=Path("outputs/logs/train_evoguard_summary.jsonl"),
        ablation_log=Path("outputs/logs/ablation_results.jsonl"),
        report_dir=Path("outputs/reports"),
        figure_dir=Path("outputs/figures"),
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
