"""Co-evolution pipeline package.

The driver lives in :mod:`evoguard.pipeline.driver` and round-artifact IO in
:mod:`evoguard.pipeline.io`. This package re-exports the public surface so
callers can simply do::

    from evoguard.pipeline import Pipeline, ExperimentSummary
"""

from __future__ import annotations

from evoguard.pipeline.driver import (
    Pipeline,
    RoundResult,
    ExperimentSummary,
)

__all__ = ["Pipeline", "RoundResult", "ExperimentSummary"]
