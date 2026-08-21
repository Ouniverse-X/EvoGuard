"""Defender training glue (``training/intro.md``).

Two thin wrappers around the vendored frameworks:

* :mod:`evoguard.training.sft_runner` -- renders LLaMA-Factory LoRA-SFT
  datasets/configs/commands.
* :mod:`evoguard.training.grpo_runner` -- renders AEPO/verl GRPO prompts +
  Hydra overrides + launch commands.

Both are *dry-run friendly* (the default in :class:`TrainingConfig`) so the
pipeline can be exercised end-to-end without GPU resources. The
``prepare_and_run_*`` helpers return plans describing what was written and
whether training actually launched, which the pipeline uses to update the
defense agent's served-LoRA name for the next round.

The vendored framework trees under ``LLaMA-Factory/`` and ``AEPO/`` are treated
as external dependencies: nothing in this package imports them at runtime --
we only emit files they consume and shell commands that invoke them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from evoguard.config import TrainingConfig
from evoguard.process.dataset_builder import RLSample, SFTExample
from evoguard.utils.logging import get_logger

logger = get_logger("training")


@dataclass
class DefenderTrainingOutcome:
    """Result of one round's defender-training step."""

    method_used: str          # "none" | "sft" | "grpo" | "sft_then_grpo"
    sft_examples_written: int = 0
    rl_samples_written: int = 0
    adapter_dir: str = ""
    launched_sft: bool = False
    launched_grpo: bool = False
    # When the underlying server is a vLLM instance with adapters registered by
    # this symbolic name, the pipeline can flip the defense agent to use it on the
    # next round by setting ``DefenseConfig.llm.lora_adapter`` accordingly.
    new_lora_adapter_name: str = ""


def train_defender(
    records,
    *,
    exp_rounds_root: str,
    training_cfg: TrainingConfig,
    round_label: str,
    dataset_builder,
) -> DefenderTrainingOutcome:
    """Build SFT / GRPO artifacts from ``records``, optionally launching them.

    Parameters mirror what callers already have:

      - ``records`` -- list of :class:`~evoguard.core.types.TrajectoryRecord`.
      - ``dataset_builder`` -- an instantiated
        :class:`~evoguard.process.DefenderDatasetBuilder`; passed in so it shares
        task/tool registries built once per experiment rather than rebuilt here.
      - ``round_label`` -- short tag used for naming output subdirs + commands.
    """

    outcome = DefenderTrainingOutcome(method_used="none")

    if not training_cfg.enabled:
        return outcome

    method = (training_cfg.method or "").lower()

    # Native-trainer shortcut: bypasses vendored LF/verl entirely when the user
    # opts in via TrainingConfig.use_native_trainer=True OR sets method to one of:
    #
    #   * "native_sft"             -- explicit cold-start SFT (legacy behavior).
    #   * "native_grpo"            -- pure incremental RL, requires existing marker.
    method_lower = method.lower()
    is_explicit_new_method = (
        method_lower == "native_grpo"
        or method_lower == "sft_then_native_grpo"
        or method_lower == "sft_then_online_grpo"
    )
    use_legacy_native_sft_fallback = (
        bool(getattr(training_cfg, "use_native_trainer", False))
        and not is_explicit_new_method
    )
    use_native = (
        is_explicit_new_method
        or use_legacy_native_sft_fallback
        or method_lower == "native_sft"
    )
    if use_native:
        # Determine whether this round should run cold-start SFT vs incremental GRPO.
        do_coldstart_sft_this_round = (
            method_lower == "native_sft"
            or use_legacy_native_sft_fallback
            or (
                method_lower in ("sft_then_native_grpo",
                                  "sft_then_online_grpo")
                and str(round_label).endswith("r0")
                and not _has_existing_warm_start(exp_rounds_root)
            )
        )

        if do_coldstart_sft_this_round:
            from evoguard.training.native_runner import train_native_sft

            examples: list[SFTExample] = []
            if hasattr(dataset_builder, "build_sft"):
                try:
                    examples = list(dataset_builder.build_sft(records))
                except Exception as exc:                              # noqa: BLE001
                    logger.error("[train_defender][native-sft] build_sft raised: %s", exc)
                    examples = []
                else:
                    stats = getattr(dataset_builder, "last_sft_stats", None)
                    if stats:
                        logger.info(
                            "[train_defender][native-sft] dataset composition: %s", stats
                        )

            marker = os.path.join(exp_rounds_root, "latest_adapter_dir.txt")
            warm_start = _read_warm_start_dir(marker)

            result = train_native_sft(
                examples=examples,
                exp_rounds_root=exp_rounds_root,
                training_cfg=training_cfg,
                round_label=round_label,
                warm_start_from_dir=warm_start,
            )
            outcome.method_used = result.method_used
            outcome.sft_examples_written = result.sft_examples_written
            outcome.rl_samples_written = 0  # native runner doesn't do RL yet.
            outcome.adapter_dir = result.adapter_dir
            outcome.launched_sft = result.launched_sft
            outcome.launched_grpo = False

            new_name = _update_marker_if_saved(
                exp_rounds_root=exp_rounds_root,
                adapter_dir=result.adapter_dir,
                round_label=round_label,
                kind="sft",
            )
            outcome.new_lora_adapter_name = new_name

        else:                                                          # noqa: RET506 -- explicit branch reads cleaner here.
            # GRPO branch per spec §4.2. Requires existing warm-start dir as init point;
            # hard-error if absent so users get clear feedback instead of silent cold-start fallback.
            from evoguard.training.native_grpo_runner import train_native_grpo

            marker = os.path.join(exp_rounds_root, "latest_adapter_dir.txt")
            init_from_dir = _read_warm_start_dir(marker)
            if not init_from_dir:
                logger.error(
                    "[train_defender][native-grpo] %s requires prior trained adapter "
                    "marker at %s; none found. Run r0 with method=native_sft or "
                    "method=sft_then_native_grpo first.", round_label, marker,
                )
                return DefenderTrainingOutcome(method_used="error_no_init_from_dir")

            logger.info(
                "[train_defender][native-grpo] %s starting from %r",
                round_label, init_from_dir,
            )

            # ---- Online single-stage co-evolution trainer route ----------- #
            # Replaces TRL.GRPOTrainer._generate_and_score_completions internal
            # generate() sampler w/ external Controller-driven trio rollouts
            # producing G genuine siblings per prompt-row scored through live
            # judge_call closure threading real outcome verdicts back. See plan
            # partitioned-swimming-hartmanis.md §1 + module docstring atop
            # online/trio_controller.py. Factory sourcing policy:
            #   * dataset_builder may expose `_online_ctrl_factory` attr upstream;
            #   * dry_run mode never invokes factory so safe-stub guards None path;
            #   * production mode lacking factory -> informative error outcome.
            if method_lower == "sft_then_online_grpo":
                ctrl_factory_fn = getattr(dataset_builder,
                                          "_online_ctrl_factory", None)
                if ctrl_factory_fn is None and bool(getattr(training_cfg,
                                                            "dry_run",
                                                             False)):
                    def _stub_fac_dryrun():                                # noqa: ANN202
                        raise AssertionError(
                            "[dispatcher] dry-run must NOT invoke "
                            "controller_factory_fn")
                    ctrl_factory_fn = _stub_fac_dryrun

                if ctrl_factory_fn is None:
                    logger.error(
                        "[train_defender][online-grpo] %s requires "
                        "dataset_builder._online_ctrl_factory attribute when "
                        "training_cfg.dry_run=False; none found. Attach callable "
                        "returning fresh fully-wired LogpAgent-wrapped Controller.",
                        round_label,
                    )
                    return DefenderTrainingOutcome(method_used="error_no_ctrl_factory")

                from online.trio_controller import train_online_grpo

                ol_result = train_online_grpo(
                    exp_rounds_root=exp_rounds_root,
                    training_cfg=training_cfg,
                    round_label=round_label,
                    records=list(records),
                    dataset_builder=dataset_builder,
                    init_from_dir=init_from_dir,
                    controller_factory_fn=ctrl_factory_fn,
                )

                outcome.method_used = ol_result.method_used
                outcome.sft_examples_written = 0
                outcome.rl_samples_written = int(
                    getattr(ol_result, "grpo_samples_written", 0))
                og_adapter_dir_val = getattr(ol_result, "adapter_dir", "")
                outcome.adapter_dir = og_adapter_dir_val
                outcome.launched_sft = False
                outcome.launched_grpo = bool(getattr(ol_result, "launched_grpo", False))

                if (
                    bool(outcome.launched_grpo)
                    and os.path.isdir(og_adapter_dir_val or "")
                    and any(name.endswith("adapter_model.safetensors")
                            for name in os.listdir(og_adapter_dir_val))
                ):
                    abspath_og = os.path.abspath(og_adapter_dir_val)
                    try:
                        with open(marker, "w", encoding="utf-8") as fh_og_w:
                            fh_og_w.write(abspath_og.rstrip("/"))
                        logger.info("[train_defender][online-grpo] updated %s -> %s",
                                    marker, abspath_og)
                        outcome.new_lora_adapter_name = (
                            f"evoguard_{round_label}_online_weights")
                    except OSError as io_exc_og:                              # noqa: BLE001
                        logger.warning("[train_defender][online-grpo] failed writing "
                                       "%s (%s)", marker, io_exc_og)

                return outcome

            grpo_result = train_native_grpo(
                exp_rounds_root=exp_rounds_root,
                training_cfg=training_cfg,
                round_label=round_label,
                records=list(records),
                dataset_builder=dataset_builder,
                init_from_dir=init_from_dir,
            )

            # Translate NativeGrpoOutcome -> DefenderTrainingOutcome fields.
            outcome.method_used = grpo_result.method_used
            outcome.sft_examples_written = 0
            outcome.rl_samples_written = grpo_result.grpo_samples_written
            outcome.adapter_dir = grpo_result.adapter_dir
            # launched flags reflect whether actual fit() ran successfully AND saved weights.
            outcome.launched_sft = False
            outcome.launched_grpo = bool(grpo_result.launched_grpo)

            if (
                grpo_result.launched_grpo
                and os.path.isdir(grpo_result.adapter_dir or "")
                and any(name.endswith("adapter_model.safetensors")
                        for name in os.listdir(grpo_result.adapter_dir))
            ):
                abspath_g = os.path.abspath(grpo_result.adapter_dir)
                try:
                    with open(marker, "w", encoding="utf-8") as fh:
                        fh.write(abspath_g.rstrip("/"))
                    logger.info(
                        "[train_defender][native-grpo] updated %s -> %s",
                        marker, abspath_g,
                    )
                    outcome.new_lora_adapter_name = f"evoguard_{round_label}_grpo_weights"
                except OSError as io_exc:                                # noqa: BLE001
                    logger.warning(
                        "[train_defender][native-grpo] failed writing %s (%s)",
                        marker, io_exc,
                    )

        return outcome

    do_sft  = method in ("sft", "sft_then_grpo")
    do_rl   = method in ("grpo", "sft_then_grpo")
    if not (do_sft or do_rl):
        # Defensive default; treat unknown as no-op instead of crashing mid-run.
        do_sft = True

    label = round_label

    if do_sft:
        from evoguard.training.sft_runner import prepare_and_run_sft

        examples: list[SFTExample] = []
        if hasattr(dataset_builder, "build_sft"):
            examples = list(dataset_builder.build_sft(records))
            stats = getattr(dataset_builder, "last_sft_stats", None)
            if stats:
                logger.info("[train_defender][sft] dataset composition: %s", stats)
        plan = prepare_and_run_sft(
            examples=examples,
            exp_rounds_root=exp_rounds_root,
            training_cfg=training_cfg,
            round_id_for_label=label,
        )
        outcome.sft_examples_written = plan.n_examples
        outcome.launched_sft = plan.launched
        outcome.adapter_dir = plan.spec.adapter_dir
        outcome.new_lora_adapter_name = os.path.basename(plan.spec.adapter_dir.rstrip("/"))
        outcome.method_used = "sft"

    if do_rl:
        from evoguard.training.grpo_runner import prepare_and_run_grpo

        samples: list[RLSample] = []
        if hasattr(dataset_builder, "build_rl"):
            samples = list(dataset_builder.build_rl(records))
        grpo_plan = prepare_and_run_grpo(
            samples=samples,
            exp_rounds_root=exp_rounds_root,
            training_cfg=training_cfg,
            round_id_for_label=label,
        )
        outcome.rl_samples_written = grpo_plan.n_samples
        outcome.launched_grpo = grpo_plan.launched
        outcome.adapter_dir = grpo_plan.spec.adapter_dir
        suffix = "_rl"
        outcome.new_lora_adapter_name = (
            f"{outcome.new_lora_adapter_name}{suffix}"
            if outcome.new_lora_adapter_name else f"evoguard_{label}_grpo"
        )
        prev_method = outcome.method_used
        outcome.method_used = (
            f"{prev_method}+grpo" if prev_method != "none" else "grpo"
        )

    return outcome


# --------------------------------------------------------------------------- #
# Internal helpers shared by native_sft / native_grpo dispatch branches        #
# --------------------------------------------------------------------------- #
def _marker_path(exp_rounds_root: str) -> str:
    """Canonical location of the warm-start pointer file under an experiment dir."""
    return os.path.join(exp_rounds_root, "latest_adapter_dir.txt")


def _has_existing_warm_start(exp_rounds_root: str) -> bool:
    """Return True iff marker file exists AND points at a valid adapter directory."""
    return bool(_read_warm_start_dir(_marker_path(exp_rounds_root)))


def _read_warm_start_dir(marker_file: str):
    """Read+validate ``latest_adapter_dir.txt`` content.

    Returns absolute path string when both expected files are present inside,
    else None. Refuses garbage state so downstream code never loads a corrupt
    adapter.
    """
    if not os.path.isfile(marker_file):
        return None
    try:
        with open(marker_file, encoding="utf-8") as fh:
            cand = fh.read().strip()
    except OSError as io_exc:
        logger.warning("[train_defender] failed reading %s: %s", marker_file, io_exc)
        return None
    if not cand or not os.path.isdir(cand):
        return None
    has_files = any(
        name.endswith(("adapter_config.json", "adapter_model.safetensors"))
        for name in os.listdir(cand)
    )
    return cand if has_files else None


def _update_marker_if_saved(
    *,
    exp_rounds_root: str,
    adapter_dir: str,
    round_label: str,
    kind: str = "",
) -> str:
    """Write latest_adapter_dir.txt pointing at freshly-trained weights on success.

    Returns symbolic lora-adapter-name to register onto running vLLM server, or ""
    when no usable artifact was produced (caller should keep previous weights).
    """
    if (
        adapter_dir
        and os.path.isdir(adapter_dir)
        and any(name.endswith("adapter_model.safetensors")
                for name in os.listdir(adapter_dir))
    ):
        abspath = os.path.abspath(adapter_dir)
        marker = _marker_path(exp_rounds_root)
        try:
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(abspath.rstrip("/"))
            logger.info(
                "[train_defender][%s] updated %s -> %s",
                kind or "?", marker, abspath,
            )
            suffix_tag = "_grpo_weights" if kind == "grpo" else f"_{kind}_weights" \
                          if kind else "_weights"
            return f"evoguard_{round_label}{suffix_tag}"
        except OSError as io_exc:
            logger.warning(
                "[train_defender][%s] failed writing marker file -> next round "
                "won't pick up new adapter (%s)", kind, io_exc,
            )
            return ""
    else:
        logger.info(
            "[train_defender][%s] no usable adapter produced at %r; keeping prior.",
            kind or "?", adapter_dir,
        )
        return ""


# Public surface ----------------------------------------------------------- #
__all__ = ["train_defender", "DefenderTrainingOutcome"]
