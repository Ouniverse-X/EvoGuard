#!/usr/bin/env python3
"""Build cleaned ToolSafe/TS-Bench evaluation rollout files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.envs.toolsafe_env import ToolSafeEnv
from evoguard.evaluation.metrics import compute_metrics
from evoguard.types import TrajectoryRecord


DEFAULT_TSBENCH_ROOT = Path("/tmp/ToolSafe/TS-Bench")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build no-overlap ToolSafe and TS-Bench eval rollout files.")
    parser.add_argument("--train-rollouts", default="data/rollouts/toolsafe_train_tri_rollouts.jsonl")
    parser.add_argument("--heldout-rollouts", default="data/eval/toolsafe_heldout_tri_rollouts.jsonl")
    parser.add_argument("--tsbench-root", default=str(DEFAULT_TSBENCH_ROOT))
    parser.add_argument("--output-dir", default="data/eval")
    parser.add_argument("--round-id", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_rollouts(Path(args.train_rollouts))
    heldout_records = read_rollouts(Path(args.heldout_rollouts))
    train_signatures = {signature(record) for record in train_records}
    no_overlap = [record for record in heldout_records if signature(record) not in train_signatures]
    overlap = [record for record in heldout_records if signature(record) in train_signatures]

    outputs: dict[str, dict[str, Any]] = {}
    outputs["toolsafe_heldout_no_overlap"] = write_rollouts(
        output_dir / "toolsafe_heldout_no_overlap.jsonl",
        no_overlap,
        extra={"removed_exact_overlap": len(overlap), "source": args.heldout_rollouts},
    )

    tsbench_root = Path(args.tsbench_root)
    subsets = {
        "tsbench_agentharm_full": [
            tsbench_root / "agentharm-traj" / "benign_steps.json",
            tsbench_root / "agentharm-traj" / "harmful_steps.json",
        ],
        "tsbench_agentdojo": sorted((tsbench_root / "agentdojo-traj").glob("*.json")),
        "tsbench_asb": sorted((tsbench_root / "asb-traj" / "test").glob("*.json")),
    }
    all_records: list[TrajectoryRecord] = []
    for name, paths in subsets.items():
        records = build_records(paths, round_id=args.round_id, split=name)
        outputs[name] = write_rollouts(output_dir / f"{name}_rollouts.jsonl", records, extra={"inputs": [str(p) for p in paths]})
        all_records.extend(records)

    outputs["tsbench_all"] = write_rollouts(
        output_dir / "tsbench_all_rollouts.jsonl",
        all_records,
        extra={"inputs": [str(path) for paths in subsets.values() for path in paths]},
    )

    summary_path = output_dir / "toolsafe_tsbench_eval_splits_summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))


def build_records(paths: list[Path], *, round_id: int, split: str) -> list[TrajectoryRecord]:
    env = ToolSafeEnv.from_paths(paths)
    return env.to_records(round_id=round_id, split=split)


def read_rollouts(path: Path) -> list[TrajectoryRecord]:
    from evoguard.training.llm_rl import read_rollout_jsonl

    return read_rollout_jsonl(path)


def write_rollouts(path: Path, records: list[TrajectoryRecord], *, extra: dict[str, Any]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    payload = {
        "path": str(path),
        "records": len(records),
        "distribution": distribution(records),
        "metrics": compute_metrics(records),
        **extra,
    }
    return payload


def distribution(records: list[TrajectoryRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record.trajectory_type.value}:{record.gold_safety_action.value}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def signature(record: TrajectoryRecord) -> str:
    return json.dumps(
        {
            "user_task": record.user_task,
            "tool_name": record.tool_name,
            "tool_arguments": dict(record.tool_arguments),
            "gold": record.gold_safety_action.value,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


if __name__ == "__main__":
    main()
