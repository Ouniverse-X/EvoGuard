"""Pre-training LoRA layer probe (one-shot static version).

Public entry point: :func:`run_lora_layer_probe(config)`.

Pipeline:
    1. pair_collector.collect_probe_pairs -- stratified sampling across the
       configured AgentDojo suites (banking/slack/travel/workspace), reusing
       :class:`evoguard.controller.Controller` to drive paired clean + injected
       trajectories on a freshly-loaded base model.
    2. sensitivity.attn_kl_score over each paired forward pass.
    3. ranking.aggregate_pair_scores -> rank_blocks_by_score ->
       select_top_k_blocks -> build_target_modules.
    4. ranking.emit_probe_artifact writes JSON consumed later by
       ``evoguard.training.native_runner.train_native_sft``.

Designed for single-GPU invocation via ``scripts/run_lora_probe.sh``; total wall
clock ~30-45 min for 80 pairs x 28 layers under Qwen2.5-7B bfloat16.
"""

from __future__ import annotations

from .pair_collector import (
    ProbePair,
    ProbePairCollection,
    collect_probe_pairs,
)
from .runner import run_lora_layer_probe

__all__ = [
    "ProbePair",
    "ProbePairCollection",
    "collect_probe_pairs",
    "run_lora_layer_probe",
]
