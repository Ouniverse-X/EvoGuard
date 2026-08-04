"""Preliminary experiment: IPI alertness-entropy vs Delta.

A self-contained measurement harness that mines successful indirect-prompt-injection
attacks from prior EvoGuard rounds, buckets them by their historical ``Delta`` value
(imm / d1 / d2 / d3 / d4), replays each scenario against a bare Qwen2.5-7B-Instruct
served by vLLM, and computes per-token Shannon entropy of the agent's first response
after exposure to the injected tool observation.

Hypothesis under test (see ``docs/superpowers/specs/2026-08-02-preliminary-ipi-alertness-entropy-design.md``):
    As bucket label moves {imm -> d4}, mean entropy decreases monotonically.
    Equivalently Spearman rho(bucket_ordinal, mean_entropy) < 0 at alpha=0.05.

Public surface intentionally minimal; consumers should drive the pipeline via the CLI:
    python -m preliminary.cli --config configs/preliminary_entropy.yaml --stage all
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
