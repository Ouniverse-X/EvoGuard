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
    if method not in {"sft", "grpo", "sft_then_grpo"}:
        raise ValueError(f"Unknown TrainingConfig.method={training_cfg.method!r}")

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


# Public surface ----------------------------------------------------------- #
__all__ = ["train_defender", "DefenderTrainingOutcome"]
