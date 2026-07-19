"""Processing: signal computation and dataset construction (``process/intro.md``)."""

from __future__ import annotations

from evoguard.process.dataset_builder import (
    DefenderDatasetBuilder,
    RLSample,
    SFTExample,
)
from evoguard.process.edit_distance import align, first_divergence_index_b
from evoguard.process.signals import compute_signals

__all__ = [
    "align",
    "first_divergence_index_b",
    "compute_signals",
    "DefenderDatasetBuilder",
    "SFTExample",
    "RLSample",
]
