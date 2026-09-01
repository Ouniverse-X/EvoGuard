"""Scan-vs-LLM turning-point agreement study (one-off, 2026-09-01).

``ProcessConfig.turning_point_method`` defaults to ``"scan"`` so every historical
run stays bit-reproducible. This script produces the evidence needed to decide
whether to flip that default to ``"llm"`` / ``"llm_then_scan"``: it replays the
attack judge over already-collected round records -- with the clean twin attached
as ``benign_reference`` -- and compares the judge's attributed ``turning_turn``
against the positional scan value stored in each record's ``signals``.

Only B (attack-success) records are compared: a turning point has no meaning on a
failure, which is the whole point of the 2026-09-01 change.

Usage::

    EVOGUARD_JUDGE_LLM_BASE_URL=http://localhost:8002/v1 \
    python scripts/study_turning_point_agreement.py \
        rounds/<exp>/round_0/records.jsonl [more.jsonl ...] \
        [--backend openai] [--model <name>] [--limit N] [--out study.json]

With no ``--backend`` it runs against the MockClient, which is only useful as a
plumbing check (the mock's attribution is the same post-injection-call rule the
scan approximates, so agreement there is near-tautological).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evoguard.config import LLMConfig
from evoguard.core.types import AttackOutcome, TrajectoryKind
from evoguard.judge import AttackJudge
from evoguard.pipeline.io import load_records_jsonl
from evoguard.process.signals import _validated_judged_turning_point


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("records", nargs="+", help="one or more round records.jsonl paths")
    p.add_argument("--backend", default="mock", help="LLMConfig.backend (mock|openai|qianfan)")
    p.add_argument("--model", default="", help="judge model name (backend-specific)")
    p.add_argument("--base-url", default=os.environ.get("EVOGUARD_JUDGE_LLM_BASE_URL", ""))
    p.add_argument("--limit", type=int, default=0, help="cap the number of B records judged")
    p.add_argument("--out", default="", help="write the per-record table here as JSON")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    cfg_kwargs: dict = {"backend": args.backend, "temperature": 0.0, "max_tokens": 512}
    if args.model:
        cfg_kwargs["model"] = args.model
    if args.base_url:
        cfg_kwargs["base_url"] = args.base_url
    judge = AttackJudge(LLMConfig(**cfg_kwargs))

    rows: list[dict] = []
    for path in args.records:
        records = load_records_jsonl(path)
        # Clean twins are keyed by task within the same file/round.
        cleans = {
            r.task_id: r.trajectory
            for r in records
            if r.kind is TrajectoryKind.CLEAN
        }
        for rec in records:
            if rec.kind is not TrajectoryKind.ATTACKED:
                continue
            if rec.outcome is not AttackOutcome.SUCCESS:
                continue          # no turning point exists on C -- nothing to compare
            if rec.attack is None or rec.signals is None:
                continue
            if args.limit and len(rows) >= args.limit:
                break

            clean = cleans.get(rec.task_id)
            verdict = judge.judge_verdict(rec.trajectory, rec.attack,
                                          benign_reference=clean)
            b_turns = rec.trajectory.tool_call_turns()
            judged = _validated_judged_turning_point(
                verdict.turning_turn, b_turns, rec.attack.target_turn
            )
            # The stored value is whatever method produced the record; on every
            # run to date that is the scan.
            scan = rec.signals.turning_point
            rows.append({
                "source":            path,
                "record_id":         rec.record_id,
                "task_id":           rec.task_id,
                "injection_point":   rec.attack.target_turn,
                "scan_turning_point": scan,
                "judge_raw_turn":    verdict.turning_turn,
                "judge_turning_point": judged,
                "judge_agrees_success": verdict.success,
                "reason":            verdict.reason[:200],
            })

    if not rows:
        print("no B records found in the given files -- nothing to compare")
        return 1

    n = len(rows)
    resolved = [r for r in rows if r["judge_turning_point"] is not None]
    both = [r for r in resolved if r["scan_turning_point"] is not None]
    exact = [r for r in both if r["judge_turning_point"] == r["scan_turning_point"]]
    diffs = [r["judge_turning_point"] - r["scan_turning_point"] for r in both]
    # A judged turn LATER than the scan's means the scan fired on a difference
    # that was not yet attributable to the injection -- the failure mode that
    # motivated the change.
    later = [d for d in diffs if d > 0]
    earlier = [d for d in diffs if d < 0]
    rejected = [
        r for r in rows
        if r["judge_turning_point"] is None and r["judge_raw_turn"] not in (None, -1)
    ]

    print(f"B records judged                : {n}")
    print(f"  judge re-confirmed success    : {sum(1 for r in rows if r['judge_agrees_success'])}")
    print(f"  judge attributed a valid turn : {len(resolved)} ({len(resolved)/n:.1%})")
    print(f"  invalid turn rejected by guard: {len(rejected)}")
    print(f"  comparable (both non-null)    : {len(both)}")
    if both:
        print(f"  exact agreement              : {len(exact)}/{len(both)} ({len(exact)/len(both):.1%})")
        print(f"  judge later than scan        : {len(later)}")
        print(f"  judge earlier than scan      : {len(earlier)}")
        print(f"  mean signed diff (judge-scan): {statistics.fmean(diffs):+.2f}")
        print(f"  diff histogram               : "
              f"{dict(sorted(Counter(diffs).items()))}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "n": n, "n_comparable": len(both),
                       "n_exact": len(exact)}, f, ensure_ascii=False, indent=2)
        print(f"\nper-record table -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
