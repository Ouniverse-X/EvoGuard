"""Categorized trajectory exporter (``process/evo_data_exporter.py``).

Persists every per-round :class:`TrajectoryRecord` into a flat three-bucket
directory tree under ``<exp_dir>/evo_data/`` so downstream fine-tuning /
analysis pipelines can sample cleanly without re-parsing ``records.jsonl``:

* ``clean/<round_id>/<task_id>__<record_id>.jsonl``   -- trajectory A
  (normal tool calling under un-poisoned context).
* ``attack_success_B/<round_id>/<task_id>__<record_id>.jsonl`` -- injected
  attack that succeeded against the defense agent; used as negative /
  contrastive samples for defender SFT.
* ``attack_failure_C/<round_id>/<task_id>__<record_id>.jsonl`` -- injection
  present but the defense blocked it while still preserving legitimate task
  progress where possible; positive / safe-behavior examples.

Each JSONL file contains exactly one line carrying the full record dict as
emitted by :meth:`TrajectoryRecord.to_dict` so consumers get messages[],
actions[], signals, utility, judge_reason metadata etc. verbatim.

The export is idempotent within a round: writing for ``(exp_dir, round_id)``
twice in one session clears and recreates only that specific round's bucket
subdirectories. Other rounds stay untouched.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Iterable

from evoguard.core.types import AttackOutcome, TrajectoryKind, TrajectoryRecord
from evoguard.utils.logging import get_logger

logger = get_logger("evo_data")

# Bucket sub-directory names mirror plan.md A/B/C terminology.
BUCKET_CLEAN = "clean"
BUCKET_SUCCESS_B = "attack_success_B"
BUCKET_FAILURE_C = "attack_failure_C"

_ALL_BUCKETS: tuple[str, ...] = (
    BUCKET_CLEAN,
    BUCKET_SUCCESS_B,
    BUCKET_FAILURE_C,
)


def _classify(rec: TrajectoryRecord) -> str | None:
    """Map a single record onto its evo_data bucket name."""

    if rec.kind is TrajectoryKind.CLEAN:
        return BUCKET_CLEAN
    if rec.kind is not TrajectoryKind.ATTACKED:
        return None
    if rec.outcome is AttackOutcome.SUCCESS:
        return BUCKET_SUCCESS_B
    if rec.outcome is AttackOutcome.FAIL:
        return BUCKET_FAILURE_C
    return None


def _safe_filename_part(s: str) -> str:
    """Filesystem-safe slug preserving readability of long task_ids."""
    keep = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
    )
    return "".join(c if c in keep else "_" for c in (s or ""))[:160] or "x"


def export_round_trajectories(
    records: Iterable[TrajectoryRecord],
    *,
    exp_dir: str,
    round_id: int,
    bucket_parent_suffix: str = "",
) -> dict[str, int]:
    """Write categorized trajectories to ``<exp_dir>/evo_data[/suffix]/{bucket}/r{N}/``.

    Parameters
    ----------
    records
        Iterable of :class:`TrajectoryRecord` produced by the pipeline driver
        for this round. Each record is classified into exactly one of three
        buckets (clean / attack_success_B / attack_failure_C) and serialized
        as a single JSON line under its own file.
    exp_dir
        Experiment root directory, e.g. ``rounds/<exp_name>``.
    round_id
        Integer round index used to namespace per-round subdirectories so
        re-runs don't clobber earlier rounds' artifacts.
    bucket_parent_suffix
        Optional extra path component appended between ``evo_data`` and the
        bucket name to support parallel experiments sharing one root tree --
        e.g. passing ``"agentdojo"`` yields layout::

            <exp>/evo_data/agentdojo/clean/r0/<task>__<rec>.jsonl

        Empty string (default) preserves legacy flat structure unchanged.

    Returns a mapping of bucket-name -> number-of-files-written for telemetry;
    callers may log this however they like. Never raises on individual write
    failures -- logs warnings instead so the pipeline keeps making forward
    progress even if disk fills mid-export.
    """

    if bucket_parent_suffix:
        # Sanitize suffix defensively against path-traversal attempts while still
        # allowing reasonable identifiers such as 'agentdojo_v2'.
        safe = "".join(c if (c.isalnum() or c in "-_./") else "_"
                       for c in str(bucket_parent_suffix)).strip("/")
        root = os.path.join(exp_dir, "evo_data", *safe.split("/")) if safe \
               else os.path.join(exp_dir, "evo_data")
    else:
        root = os.path.join(exp_dir, "evo_data")
    counts: dict[str, int] = {b: 0 for b in _ALL_BUCKETS}

    # Pre-create / clear ONLY this round's buckets so prior rounds survive.
    round_subdirs_by_bucket: dict[str, str] = {}
    for b in _ALL_BUCKETS:
        d = os.path.join(root, b, f"r{int(round_id)}")
        try:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        except OSError as exc:
            logger.warning("[evo_data] could not reset %s: %s", d, exc)
            continue
        round_subdirs_by_bucket[b] = d

    n_total_attempted = 0
    n_unclassified = 0

    for rec in records:
        n_total_attempted += 1
        bucket = _classify(rec)
        if bucket is None:
            n_unclassified += 1
            continue
        target_dir = round_subdirs_by_bucket.get(bucket)
        if target_dir is None:
            # Earlier mkdir failed for this bucket; skip silently.
            continue
        fname = f"{_safe_filename_part(str(getattr(rec,'task_id','?')))}__{_safe_filename_part(str(getattr(rec,'record_id','?')))}.jsonl"
        path = os.path.join(target_dir, fname)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False))
                fh.write("\n")
        except OSError as exc:
            logger.warning("[evo_data] failed writing %s: %s", path, exc)
            continue
        counts[bucket] = counts.get(bucket, 0) + 1

    logger.info(
        "[evo_data] r%d exported %d/%d records "
        "(A=%d, B=%d, C=%d%s)",
        round_id,
        sum(counts.values()),
        n_total_attempted,
        counts[BUCKET_CLEAN],
        counts[BUCKET_SUCCESS_B],
        counts[BUCKET_FAILURE_C],
        f"; {n_unclassified} skipped" if n_unclassified else "",
    )
    return counts


__all__: list[str] = [
    "export_round_trajectories",
    "BUCKET_CLEAN",
    "BUCKET_SUCCESS_B",
    "BUCKET_FAILURE_C",
]
