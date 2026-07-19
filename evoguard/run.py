"""CLI entry point for running an EvoGuard experiment.

Usage::

    python -m evoguard.run --config configs/example.yaml
    python -m evoguard.run --smoke            # quick mock-only sanity run

The ``--config`` form loads an :class:`ExperimentConfig` from YAML/JSON and
runs the co-evolution loop to completion (or until termination criteria fire).
``--smoke`` is shorthand for the deterministic offline smoke test defined in
:mod:`evoguard.tests.smoke_test`.
"""

from __future__ import annotations

import argparse
import os
import sys


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="evoguard",
        description="EvoGuard co-evolutionary attack/defense pipeline.",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="Path to ExperimentConfig yaml/json.")
    g.add_argument("--smoke", action="store_true", help="Run the offline smoke test.")
    return ap


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.smoke:
        # Defer import so that --help / arg parsing stay fast.
        from evoguard.tests import smoke_test
        rc = smoke_test.main()
        return int(rc or 0)

    cfg_path = os.path.abspath(args.config)
    if not os.path.exists(cfg_path):
        print(f"error: config file not found: {cfg_path}", file=sys.stderr)
        return 2
    from evoguard.config import ExperimentConfig
    from evoguard.pipeline import Pipeline

    cfg = ExperimentConfig.from_file(cfg_path)
    summary = Pipeline(cfg).run()
    print("Experiment complete.")
    print(f"  exp_dir           : {summary.exp_dir}")
    print(f"  rounds completed  : {summary.n_rounds_completed}")
    print(f"  terminated early? : {summary.terminated} ({summary.terminate_reason})")
    if summary.curves_png_path:
        print(f"  curves.png         : {summary.curves_png_path}")
    if os.path.isfile(os.path.join(summary.exp_dir, "metrics.csv")):
        print(f"  metrics.csv       : {os.path.join(summary.exp_dir,'metrics.csv')}")
    # Surface docs/todo.md item #4 deliverables: safety metrics under results/.
    for fname, label in (
        ("safety_metrics.jsonl", "results/safety_metrics.jsonl"),
        ("safety_metrics.csv",   "results/safety_metrics.csv"),
    ):
        cand = os.path.join(summary.exp_dir, "results", fname)
        if os.path.isfile(cand):
            print(f"  {label:<28}: {cand}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
