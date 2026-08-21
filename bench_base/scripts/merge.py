"""Convert MCTS evolution output into bench_v2 schema and merge into main bench.

Filtering modes (selectable via env var MCTS_MERGE_FILTER, default 'all'):
  * all        — keep every record with delta>=bucket threshold regardless of judge
  * verified   — keep only records where mcts_judge_verified==True
  * success    — keep records where success flag is True
  * any_strict — require (judge_verified OR success) True

Domain metadata: source rec's "suite" field propagated to scenario.domain when present;
otherwise falls back to legacy hardcoded 'workspace'.
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BENCH_ROOT = Path(os.environ.get("MCTS_BENCH_ROOT", str(REPO_ROOT / "bench_base")))
SRC = _BENCH_ROOT / "scenarios" / "_synthetic"
DST = _BENCH_ROOT / "scenarios"
FILTER_MODE = os.environ.get("MCTS_MERGE_FILTER", "all").lower()


def _passes_filter(rec: dict) -> bool:
    if FILTER_MODE == "verified":
        return bool(rec.get("judge_verified", False))
    if FILTER_MODE == "success":
        return bool(rec.get("success", False))
    if FILTER_MODE == "any_strict":
        return bool(rec.get("judge_verified", False)) or bool(rec.get("success", False))
    return True  # default 'all'


def to_bench_v2(rec: dict, bucket: str) -> dict:
    payload = rec.get("payload", "") or ""
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    tid = rec.get("task_id", "") or ""
    method = rec.get("method", "") or "mcts_evolved"
    inj_turn = int(rec.get("target_turn", 0))
    delta = int(rec.get("delta", int(bucket[1:])))
    turn_pt = inj_turn + delta
    suite_tag = rec.get("suite") or rec.get("domain") or "workspace"
    sig = {
        "injection_point": inj_turn,
        "turning_point": turn_pt,
        "delta": delta,
        "delta_normalized": float(delta),
        "edit_distance": -1,
    }
    return {
        "schema_version": "bench_v2",
        "scenario_id": f"scn_mcts_{sha}",
        "bucket": bucket,
        "origin_mode": "mcts_evolved",
        "canonical_technique_id": f"tech_mcts_{method[:24]}",
        "method_tag_raw": method,
        "task_id": tid,
        "domain": suite_tag,
        "toolkit_signature": rec.get("toolkit", ""),
        "goal_instruction": rec.get("instruction", ""),
        "context_prefix_actions": [],  # MCTS harness doesn't persist full context
        "poisoned_observation_text": payload,
        "injected_payload_sha256_first16": sha,
        "injection_target_turn_index": inj_turn,
        "signals_ref": sig,
        "expected_response_length_tokens": None,
        "instruction_perplexity_proxy": None,
        "confound_flags": [],
        "delta_value_orig": delta,
        "_provenance": {
            "synthesizer_version": "mcts-evolve-v2-traj-judge" if bool(rec.get("judge_verified") or rec.get("success")) else "mcts-evolve-v1-unverified",
            "validator_judge_model": "llama3-8b-port8002-trajectory-aware-v1",
            "mcts_generation": rec.get("generation"),
            "mcts_judge_verified": bool(rec.get("judge_verified", False)),
            "mcts_heuristic_success": bool(rec.get("success", False)),
            "mcts_suite_origin": suite_tag,
            "replay_defender_model_state": "base-model-no-lora-loaded",
        },
    }


def main() -> None:
    seeds = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [42]
    written_total: dict[str, int] = {"d2": 0, "d3": 0, "d4": 0}
    skipped_total: dict[str, int] = {"d2": 0, "d3": 0, "d4": 0}
    print(f"[merge] filter_mode={FILTER_MODE} bench_root={_BENCH_ROOT}")
    for seed in seeds:
        src_dir = SRC
        for d in (2, 3, 4):
            bkt = f"d{d}"
            src = src_dir / f"bucket_{bkt}_mcts_seed{seed}.jsonl"
            if not src.exists():
                print(f"[skip] {src} not found")
                continue
            records = []
            with open(src, encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln:
                        records.append(json.loads(ln))
            out_path = DST / f"bucket_{bkt}.jsonl"
            # Dedup against scenario_ids already in bench
            existing_ids: set[str] = set()
            with open(out_path, encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        existing_ids.add(json.loads(ln).get("scenario_id", ""))
                    except Exception:
                        pass
            n_added = n_skip_filter = n_dup = 0
            with open(out_path, "a", encoding="utf-8") as fh:
                for rec in records:
                    if not _passes_filter(rec):
                        n_skip_filter += 1
                        continue
                    scenario = to_bench_v2(rec, bkt)
                    if scenario["scenario_id"] in existing_ids:
                        n_dup += 1
                        continue
                    fh.write(json.dumps(scenario, ensure_ascii=False) + "\n")
                    existing_ids.add(scenario["scenario_id"])
                    n_added += 1
            written_total[bkt] += n_added
            skipped_total[bkt] += n_skip_filter + n_dup
            print(f"[merge seed={seed}] {bkt}: +{n_added} "
                  f"(from {len(records)} raw | filtered={n_skip_filter} dup={n_dup})")
    print(f"[done total] added={json.dumps(written_total)} skipped={json.dumps(skipped_total)}")


if __name__ == "__main__":
    main()

