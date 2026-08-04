"""CLI orchestrator for the preliminary IPI alertness-entropy experiment.

Drives the full pipeline through selectable stages:

    python -m preliminary.cli --config configs/preliminary_entropy.yaml \
                              --stage mine      # build bench dataset from prior rounds/
        ... (continue) ...
                              --stage harness   # collect raw responses from vLLM endpoint
                              --stage entropy   # compute per-scenario entropy CSV from responses
                              --stage stats     # produce summary_stats.json w/ Spearman+bootstrap+t-test suite
                              --stage plot      # render PNGs (main curve / per-domain facets / histogram overlay)
                              --stage all       # runs every stage in order using one shared timestamped output dir

Stage chaining rules:

  * ``mine`` writes under ``bench/`` and is idempotent given same source data.
  * ``harness`` reads ``bench/corpus_*.jsonl`` writing ``rounds/_preliminary/<ts>/raw_responses.jsonl``.
  * Subsequent stages operate on the most-recently-created-or-explicitly-specified output dir.

Override flags allow toggling dry-run mode without editing the YAML file, plus pinning a specific
output dir for re-analysis of an existing run's artifacts.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

if __package__ in (None, ""):                                                   # pragma: no cover - module-script invocation shim
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from preliminary.config import ExperimentConfig, load_config
else:
    from .config import ExperimentConfig, load_config


logger = logging.getLogger("preliminary.cli")


# --------------------------------------------------------------------------- #
# Output-dir resolution helper used across stages that share an output dir.
# --------------------------------------------------------------------------- #
def _resolve_output_dir(cfg: ExperimentConfig,
                        cli_output_dir_override: str | None) -> str:
    if cli_output_dir_override:
        return os.path.abspath(cli_output_dir_override)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    od_rel = cfg.output_dir_template.format(run_timestamp=ts)
    repo_root = cfg.resolve_repo_root()
    return os.path.join(repo_root, od_rel.replace("/", os.sep)) if not os.path.isabs(od_rel) else od_rel


def _most_recent_existing_run(repo_root: str, template_prefix: str = "rounds/_preliminary/") -> str | None:
    base = os.path.join(repo_root, "rounds", "_preliminary")
    if not os.path.isdir(base):
        return None
    subdirs = [d for d in os.listdir(base)
               if os.path.isfile(os.path.join(base, d, "raw_responses.jsonl"))]
    if not subdirs:
        return None
    subdirs.sort(key=lambda dn: os.path.getmtime(os.path.join(base, dn)), reverse=True)
    return os.path.join(base, subdirs[0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m preliminary.cli",
                                  description="Preliminary entropy-vs-Delta pipeline driver.")
    ap.add_argument("--config", default="configs/preliminary_entropy.yaml")
    ap.add_argument("--stage",
                    choices=["mine", "harness", "entropy", "stats", "plot", "all"],
                    default="all")
    ap.add_argument("--dry-run", dest="force_dry_run_true", action="store_true",
                    help="Force cfg.dry_run=True regardless of yaml setting. "
                         "Useful for offline plumbing validation before burning GPU/network cycles.")
    ap.add_argument("--output-dir-override",
                    default=os.environ.get("PRELIMINARY_OUTPUT_DIR_OVERRIDE"),
                    help="Pin a specific output dir for stages after 'mine'. Useful when re-analyzing an existing run.")
    ap.add_argument("--loglevel", default=None)

    args = ap.parse_args(argv)

    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    # Apply CLI overrides onto config copy where applicable.
    if getattr(args, "force_dry_run_true", False):
        from dataclasses import replace as dc_replace
        cfg = dc_replace(cfg, dry_run=True)

    lvl_str = (args.loglevel or cfg.logging_level).upper()
    numeric_level = getattr(logging, lvl_str.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level,
                        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s")

    logger.info("experiment=%s seed=%d dry_run_mode_active_now=%s",
                cfg.experiment_name, cfg.seed, bool(cfg.dry_run))

    requested_stage_lower = str(args.stage).lower()

    if requested_stage_lower == "all":
        stages_to_execute_sequentially = ["mine", "harness", "entropy", "stats", "plot"]
    else:
        stages_to_execute_sequentially = [requested_stage_lower]

    final_harness_output_dir_for_subsequent_stages: str | None = (
        os.path.abspath(args.output_dir_override) if args.output_dir_override else None
    )

    for stg_name in stages_to_execute_sequentially:
        print(f"\n========== STAGE [{stg_name}] BEGIN ==========", flush=True)
        if stg_name == "mine":
            from preliminary.miner import BenchMiner
            mn = BenchMiner(cfg)
            mn.mine()

        elif stg_name == "harness":
            from preliminary.harness import HarnessRunner
            runner_inst = HarnessRunner(cfg)
            out_dir_resolved_abs_path, summary_dict_from_runner = runner_inst.run()
            final_harness_output_dir_for_subsequent_stages = out_dir_resolved_abs_path

        elif stg_name in ("entropy", "stats", "plot"):
            if final_harness_output_dir_for_subsequent_stages is None or not os.path.isdir(final_harness_output_dir_for_subsequent_stages):
                resolved_recent_or_none = _most_recent_existing_run(cfg.resolve_repo_root())
                if resolved_recent_or_none is None:
                    raise SystemExit(
                        f"Stage '{stg_name}' requires either explicit --output-dir-override OR a previous "
                        "'harness'/'all' invocation having produced rounds/_preliminary/<ts>/. Neither found."
                    )
                final_harness_output_dir_for_subsequent_stages = resolved_recent_or_none
                logger.info("auto-selected most-recent run dir: %s",
                            final_harness_output_dir_for_subsequent_stages)

            raw_resp_jsonl_fp_target_stg_after_harness = os.path.join(
                final_harness_output_dir_for_subsequent_stages, "raw_responses.jsonl"
            )
            csv_out_fp_target_stg_after_harness = os.path.join(
                final_harness_output_dir_for_subsequent_stages, "entropy_per_scenario.csv"
            )
            json_summary_stats_fp_target_stg_after_harness = os.path.join(
                final_harness_output_dir_for_subsequent_stages, "summary_stats.json"
            )

            if stg_name == "entropy":
                from preliminary.entropy import run as entropy_module_run_fn_ref
                entropy_module_run_fn_ref(raw_responses_path=raw_resp_jsonl_fp_target_stg_after_harness,
                                          output_csv_path=csv_out_fp_target_stg_after_harness,
                                          cfg_K_floor_for_norm=cfg.entropy.effective_K_for_normalization_floor,
                                          first_k_window_size=cfg.entropy.first_k_window_size)
            elif stg_name == "stats":
                from preliminary.stats import analyze_and_emit_summary as stats_analyze_emit_fn_ref
                stats_analyze_emit_fn_ref(input_csv_path=csv_out_fp_target_stg_after_harness,
                                          output_json_path=json_summary_stats_fp_target_stg_after_harness,
                                          stats_cfg=cfg.stats)
            elif stg_name == "plot":
                from preliminary.plot import run as plot_module_run_fn_ref
                plot_module_run_fn_ref(input_csv_path=csv_out_fp_target_stg_after_harness,
                                       output_dir=final_harness_output_dir_for_subsequent_stages,
                                       summary_stats_json_path=json_summary_stats_fp_target_stg_after_harness)

        else:
            raise SystemExit(f"unknown stage {stg_name!r}")

        print(f"========== STAGE [{stg_name}] DONE ==========\n", flush=True)

    print("\n=== PIPELINE COMPLETE ===")
    if final_harness_output_dir_for_subsequent_stages:
        print(f"All outputs live at:\n  {final_harness_output_dir_for_subsequent_stages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
