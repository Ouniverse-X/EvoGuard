"""CLI shim for ``python -m evoguard.training.probes <config.yaml> [flags]``.

Parses the lightweight flag set documented in ``scripts/run_lora_probe.sh`` and
forwards them into :func:`evoguard.training.probes.run_lora_layer_probe`.

Supported flags:
    --dry-run                    render plan JSON only; do NOT touch torch/network.
    --override-gpu <id>          pin CUDA_VISIBLE_DEVICES to this single GPU index.
    --max-pairs <int>            override training.lora_probe_max_pairs cap.
    --stratified-per-domain <n>  reserved for future dynamic version (currently ignored).

The first positional argument is the path to the experiment YAML config.
"""

from __future__ import annotations

import argparse
import sys

from .runner import run_lora_layer_probe


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evoguard.training.probes",
        description="Pre-training LoRA layer probe (one-shot static version).",
    )
    p.add_argument("config", help="Path to ExperimentConfig YAML file.")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   default=False,
                   help="Render plan JSON only; do not invoke torch/GPU.")
    p.add_argument("--override-gpu", dest="override_gpu_id",
                   type=int, default=None,
                   help="Pin CUDA_VISIBLE_DEVICES to this GPU id.")
    p.add_argument("--max-pairs", dest="override_max_pairs",
                   type=int, default=None,
                   help="Override lora_probe_max_pairs from config.")
    p.add_argument("--stratified-per-domain", dest="stratified_per_domain_hint_unused",
                   type=int, default=None,
                   help="(Reserved, currently unused) per-suite task count hint.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outcome = run_lora_layer_probe(
        config=args.config,
        dry_run_override_flag_passed_via_cli=args.dry_run,
        override_gpu_id=args.override_gpu_id,
        override_max_pairs=args.override_max_pairs,
        stratified_per_domain_hint_unused=args.stratified_per_domain_hint_unused,
    )
    # Print a one-line summary so shell wrappers can grep it.
    if getattr(outcome, "artifact_path_written", "") or \
       getattr(outcome, "pairs_collected", 0) > 0:
        print(f"[probe-cli] DONE pairs={outcome.pairs_collected} "
              f"layers={outcome.layers_scored} "
              f"selected={outcome.selected_blocks_sorted_desc} "
              f"artifact={outcome.artifact_path_written} "
              f"elapsed={outcome.elapsed_seconds_total:.1f}s")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
