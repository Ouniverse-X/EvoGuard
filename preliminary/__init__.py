"""Preliminary sequence-probe: AS-vs-AF internal-representation divergence analysis.

Tier-1 (multi-domain x longer-thought joint expansion) + Tier-2 (temperature-
sweep deconfounder) pipeline for testing whether successful vs failed prompt-
injection attacks produce distinguishable internal-monologue representations
in a defender LLM.

Three core modules drive the pipeline:

1. ``alertness_v3_harness`` -- Phase-1 rollout collection. Produces per-scenario
   triplet (clean + N attacked) rollouts. Supports multi-domain suites via
   ``--suites`` and temperature-sweep via comma-separated ``--sampling-temperature``.

2. ``as_vs_af_sequence_probe`` -- Phase-2 feature extraction. Feeds each
   rollout's own recorded thought tokens past a shared scaffold, captures
   per-position softmax-derived scalars (entropy / top1 / refuse_mass / nll)
   and truncated top-K distribution snapshots along the decoded span.
   Default ``--max-thought-tokens=128`` (Tier-1 raise from v0's 64).

3. ``as_vs_af_sequence_analyze`` -- Phase-3 statistical analysis. Plan A
   runs segment-aggregated Wilcoxon tests across early/mid/late/extended bands
   with Bonferroni FWER control (alpha = 0.05 / (4 bands * 3 pairwise)).
   Plan B computes time-resolved paired-JS-divergence curves between AS and AF
   average distributions. Auto-discovers temperature strata and emits both
   pooled and nested per-T-stratum analyses.

Spec: ``docs/superpowers/specs/2026-08-04-sequence-probe-tier1-tier2-design.md``
"""

from __future__ import annotations

__version__ = "2.0.0"
__all__ = ["__version__"]
