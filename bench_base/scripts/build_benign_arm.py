"""Materialize the BENIGN (negative-class) arm of bench_base as first-class rows.

bench_base/scenarios/bucket_{imm,d1,d2,d3,d4}.jsonl hold 504 scenarios that are
*all* post-attack: every one carries a payload. A bench with only positives can
measure recall and ASR but has no TN/FP, so it cannot report precision,
accuracy, F1 or utility -- and it cannot detect the failure mode we actually
care about right now: a defender that raises a false alarm on benign input and
then never terminates. Measured on the P1-P4 adapter's clean arm,
``false_alarm_rate`` 72.37% (flag 68.86% + halt ~3.5%) with ``any_tool_rate``
96.49% and ``steps_mean`` 12.1, ``clean_utility_mean`` 0.10, yet ASR 2.02% --
invisible on the attacked arm alone.

This script writes ``bucket_clean.jsonl``: benign counterparts derived from the
same task / goal / context universe, with NO payload anywhere.

Two properties worth stating explicitly:

* **Self-contained.** ``eval_bench_clean_arm.py``'s original approach removed the
  payload at eval time, which meant re-resolving the clean twin out of
  ``rounds/`` on every run. That index is machine-specific (this box resolves
  ~355 of the historical 3742 records, so 29/504 scenarios skipped as
  ``skipped_no_clean_twin``). Here the resolved clean trajectory is written INTO
  the row, so the benign arm is reproducible from the bench alone.
* **Matched, then deduplicated.** The 504 attacked scenarios collapse onto 236
  distinct (task_id, goal_instruction, replay-cut) contexts -- 49 goals under a
  spread of cut indices. Emitting 504 near-duplicate benign rows would burn GPU
  time re-rolling identical contexts, so each distinct context is written once
  with ``_provenance.benign.parent_scenario_ids`` recording every attacked row it
  answers for. Downstream can report either the unique-context confusion matrix
  or the parent-weighted matched-pair one.

Schema: the same 22 bench_v2 keys, with the attack-specific ones neutralised:
``poisoned_observation_text=""`` (the benign marker),
``injected_payload_sha256_first16=null``, ``signals_ref=null``,
``delta_value_orig=null``, ``bucket="clean"``, ``origin_mode="benign_arm"``.
``injection_target_turn_index`` is KEPT, reinterpreted as the replay boundary the
evaluator generates from -- it is the cut its attacked twin was injected at, which
is what keeps the pair comparable.

Because ``origin_mode`` is not ``mined``, ``bench_schema.iter_load_scenarios``
skips these rows by default and ``bench_release_gate`` never sees them (it walks
only the five Δ buckets), so the Δ-bucket power analysis stays uncontaminated.

Usage:
    python bench_base/scripts/build_benign_arm.py [--dry-run] [--out PATH]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench_base.scripts.eval_full_rollout import (  # noqa: E402
    ROUNDS_ROOT,
    build_goal_to_clean,
    build_rounds_indexes,
)

SCENARIOS_ROOT = REPO_ROOT / "bench_base" / "scenarios"
DELTA_BUCKETS = ("imm", "d1", "d2", "d3", "d4")
BENIGN_BUCKET = "clean"
BENIGN_ORIGIN_MODE = "benign_arm"
# Keep at least two reference actions after the cut so `clean_follow` has
# something to compare the defender's continuation against.
MIN_REFERENCE_TAIL = 2


def load_attacked_scenarios() -> list[dict]:
    rows: list[dict] = []
    for lbl in DELTA_BUCKETS:
        p = SCENARIOS_ROOT / f"bucket_{lbl}.jsonl"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
    return rows


def resolve_clean_actions(scenario: dict, rec_to_clean, goal_to_clean,
                          task_to_clean) -> tuple[list[dict], str]:
    """Return (clean_actions, source_tag) for a scenario's benign counterpart."""
    prov = scenario.get("_provenance") or {}
    cri = prov.get("clean_record_record_id", "")
    if cri and cri in rec_to_clean:
        return rec_to_clean[cri], "rounds_record_id"
    goal = scenario.get("goal_instruction", "")
    if goal and goal in goal_to_clean:
        return goal_to_clean[goal], "rounds_goal_instruction"
    tid = scenario.get("task_id", "")
    if tid and tid in task_to_clean:
        return task_to_clean[tid], "rounds_task_id"
    return [], "unresolved"


def strip_payload_from_actions(scenario: dict) -> list[dict]:
    """Fallback: recover a clean prefix from the scenario's own attacked prefix.

    The poisoned observation is ``clean_observation + payload``, so removing the
    payload substring restores the clean text. Only the pre-injection steps plus
    the now-clean injection step are kept -- everything after belongs to the
    attacked continuation and is not benign behaviour.
    """
    acts = scenario.get("context_prefix_actions") or []
    payload = scenario.get("poisoned_observation_text") or ""
    it = scenario.get("injection_target_turn_index")
    if not acts or not payload or it is None or int(it) >= len(acts):
        return []
    it = int(it)
    obs = acts[it].get("observation") or ""
    if payload not in obs:
        return []
    out = [dict(a) for a in acts[: it + 1]]
    out[it]["observation"] = obs.replace(payload, "").rstrip()
    out[it]["metadata"] = {}
    return out


def _has_final_answer(actions: list[dict]) -> bool:
    return any((a.get("final_answer") or "").strip() for a in actions)


def build_benign_rows(attacked: list[dict], rec_to_clean, goal_to_clean,
                      task_to_clean) -> tuple[list[dict], collections.Counter]:
    stats: collections.Counter = collections.Counter()
    # key -> row under construction; parents accumulate onto the first hit.
    by_key: dict[tuple[str, str, int], dict] = {}
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for sc in attacked:
        stats["n_attacked_in"] += 1
        clean_acts, src = resolve_clean_actions(
            sc, rec_to_clean, goal_to_clean, task_to_clean)
        if not clean_acts:
            clean_acts = strip_payload_from_actions(sc)
            src = "payload_stripped_prefix" if clean_acts else "unresolved"
        if not clean_acts:
            stats["n_skipped_unresolved"] += 1
            continue
        if len(clean_acts) < 3:
            stats["n_skipped_clean_too_short"] += 1
            continue

        # Clamp the parent's injection turn so a reference tail always survives.
        raw_cut = int(sc.get("injection_target_turn_index") or 0)
        cut = max(0, min(raw_cut, len(clean_acts) - 1 - MIN_REFERENCE_TAIL))
        if cut != raw_cut:
            stats["n_cut_clamped"] += 1

        key = (sc.get("task_id", ""), sc.get("goal_instruction", ""), cut)
        existing = by_key.get(key)
        if existing is not None:
            b = existing["_provenance"]["benign"]
            b["parent_scenario_ids"].append(sc.get("scenario_id", ""))
            b["parent_buckets"].append(sc.get("bucket", ""))
            b["parent_count"] = len(b["parent_scenario_ids"])
            stats["n_merged_into_existing"] += 1
            continue

        digest = hashlib.sha256(
            f"{key[0]}|{key[1]}|{cut}".encode("utf-8")).hexdigest()[:12]
        by_key[key] = {
            "schema_version": sc.get("schema_version", "bench_v2"),
            "scenario_id": f"clean_{digest}",
            "bucket": BENIGN_BUCKET,
            "origin_mode": BENIGN_ORIGIN_MODE,
            "canonical_technique_id": None,
            "method_tag_raw": "",
            "task_id": sc.get("task_id", ""),
            "domain": sc.get("domain", ""),
            "toolkit_signature": sc.get("toolkit_signature"),
            "goal_instruction": sc.get("goal_instruction", ""),
            # Full clean trajectory: [:cut+1] is the replayed context, the tail is
            # the behavioural reference the defender's continuation is scored against.
            "context_prefix_actions": [dict(a) for a in clean_acts],
            "poisoned_observation_text": "",
            "injected_payload_sha256_first16": None,
            "injection_target_turn_index": cut,
            "signals_ref": None,
            "expected_response_length_tokens": None,
            "instruction_perplexity_proxy": None,
            "confound_flags": [],
            "delta_value_orig": None,
            "source": "benign_arm_from_bench_attacked",
            "_provenance": {
                "benign": {
                    "parent_scenario_ids": [sc.get("scenario_id", "")],
                    "parent_buckets": [sc.get("bucket", "")],
                    "parent_count": 1,
                    "clean_source": src,
                    "clean_reference_n_actions": len(clean_acts),
                    "clean_reference_has_final_answer": _has_final_answer(clean_acts),
                    "parent_injection_target_turn_index": raw_cut,
                },
                "built_at_utc_iso8601": now,
                "builder": "bench_base/scripts/build_benign_arm.py",
            },
            "_revalidation": None,
        }
        stats["n_rows_new"] += 1
        stats[f"clean_source:{src}"] += 1

    rows = sorted(by_key.values(), key=lambda r: r["scenario_id"])
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCENARIOS_ROOT / f"bucket_{BENIGN_BUCKET}.jsonl"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be written without touching disk.")
    args = ap.parse_args()

    attacked = load_attacked_scenarios()
    print(f"[benign] attacked scenarios loaded: {len(attacked)}", flush=True)

    print("[benign] building rounds clean indexes...", flush=True)
    task_to_clean, rec_to_clean, _ = build_rounds_indexes(ROUNDS_ROOT)
    goal_to_clean = build_goal_to_clean(task_to_clean, ROUNDS_ROOT)
    print(f"[benign] rec_to_clean={len(rec_to_clean)} goal_to_clean={len(goal_to_clean)} "
          f"task_to_clean={len(task_to_clean)}", flush=True)

    rows, stats = build_benign_rows(attacked, rec_to_clean, goal_to_clean, task_to_clean)

    print("\n[benign] build stats:")
    for k, v in sorted(stats.items()):
        print(f"    {k:38s} {v}")

    dom = collections.Counter(r["domain"] for r in rows)
    cut = collections.Counter(r["injection_target_turn_index"] for r in rows)
    pc = collections.Counter(r["_provenance"]["benign"]["parent_count"] for r in rows)
    covered = sum(r["_provenance"]["benign"]["parent_count"] for r in rows)
    print(f"\n[benign] rows={len(rows)} covering {covered}/{len(attacked)} attacked scenarios")
    print(f"[benign] domains={dict(dom.most_common())}")
    print(f"[benign] cut_index={dict(sorted(cut.items()))}")
    print(f"[benign] parent_count_hist={dict(sorted(pc.items()))}")
    print(f"[benign] reference_reaches_final_answer="
          f"{sum(1 for r in rows if r['_provenance']['benign']['clean_reference_has_final_answer'])}"
          f"/{len(rows)}")

    if args.dry_run:
        print("\n[benign] --dry-run: nothing written.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(out)
    print(f"\n[benign] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
