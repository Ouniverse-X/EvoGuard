"""Persistence of per-round artifacts under ``rounds/`` (``CLAUDE.md``).

Layout for one experiment::

    rounds/<exp_name>/
        config.yaml                  # frozen experiment configuration
        metrics.jsonl                # one RoundMetrics snapshot per line
        metrics.csv                  # flat scalar view of the above
        curves.png                   # co-evolution plots
        summary.json                 # final termination state + best stats
        round_<id>/
            records.jsonl            # all TrajectoryRecord objects this round
            population.jsonl         # current attacker genome pool, per task_id
            evaluations.jsonl        # EvaluatedAttack list, per task_id
            metrics.json             # RoundMetrics snapshot alone

The schema is JSON Lines everywhere so partial writes / streaming reads are
easy and a crashed run can be inspected up to its last successful round.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from evoguard.core.types import AttackSpec, TrajectoryRecord


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.abspath(path), exist_ok=True)


def _dump_jsonl(objs: Iterable[Any], path: str) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(_to_serializable(o), ensure_ascii=False))
            f.write("\n")


def _to_serializable(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, dict):
        return {k: _to_serializable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_serializable(x) for x in o]
    if is_dataclass(o):
        return asdict(o)
    return o


def write_json(obj: Any, path: str) -> str:
    """Pretty-write ``obj`` to ``path`` after converting dataclasses/enums."""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(obj), f, ensure_ascii=False, indent=2)
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_records_jsonl(path: str) -> list[TrajectoryRecord]:
    out: list[TrajectoryRecord] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(TrajectoryRecord.from_dict(json.loads(line)))
    return out


def save_round(
    exp_dir: str,
    *,
    round_id: int,
    records: list[TrajectoryRecord],
    populations_by_task: dict[str, list[AttackSpec]],
    evaluations_by_task: dict[str, list],
) -> dict[str, str]:
    """Persist every artifact produced by one round.

    Returns a mapping describing where each file lives so callers can log it.
    """

    rd = os.path.join(exp_dir, f"round_{round_id}")
    _ensure_dir(rd)
    paths = {
        "records": os.path.join(rd, "records.jsonl"),
        "population": os.path.join(rd, "population.jsonl"),
        "evaluations": os.path.join(rd, "evaluations.jsonl"),
    }
    # Records.
    with open(paths["records"], "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False))
            f.write("\n")
    # Population by task_id; each row carries task_id + genomes[].
    pop_rows = [
        {"task_id": tid, "generation": getattr(specs[0], "generation", 0), "genomes": [g.to_dict() for g in specs]}
        for tid, specs in populations_by_task.items()
        if specs
    ]
    with open(paths["population"], "w", encoding="utf-8") as f:
        for row in pop_rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    # Evaluations: EvaluatedAttack carries spec+fitness+success+metadata. We do
    # not depend on its exact type at import time -- duck-typing keeps imports cheap.
    eval_path = paths["evaluations"]
    with open(eval_path, "w", encoding="utf-8") as f:
        for tid, evs in evaluations_by_task.items():
            payload = {
                "task_id": tid,
                "evaluated": [
                    {
                        "spec": e.spec.to_dict(),
                        "fitness": float(e.fitness),
                        "success": bool(e.success),
                        "metadata": dict(getattr(e, "metadata", {})),
                    }
                    for e in evs
                ],
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
    return {"round_dir": rd, **paths}
