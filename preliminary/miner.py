"""Bench-corpus miner for the preliminary entropy experiment.

Scans prior EvoGuard rounds under ``rounds/<source_exp>/*/records.jsonl``, filters
attacked-SUCCESS trajectories carrying a numeric ``signals.delta`` value, deduplicates
by ``(task_id, sha256(payload))``, assigns them to five canonical Δ-buckets
(``imm | d1 | d2 | d3 | d4``), pairs each surviving attacked-B record with its
matching clean-A twin sharing ``task_id`` IN THE SAME ROUND DIRECTORY so we can splice
the poisoned observation onto an otherwise-clean history prefix, optionally downsamples
per-bucket honoring inter-domain balance constraints, and writes the resulting scenarios
to ``bench/corpus_<bucket>.jsonl`` plus a top-level ``bench/manifest.json`` audit trail.

Schema reference: docs/superpowers/specs/2026-08-02-preliminary-ipi-alertness-entropy-design.md §2.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional

# Make sibling imports work whether invoked as module (-m preliminary.miner) or script.
if __package__ in (None, ""):                                                   # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from preliminary.config import (
        ExperimentConfig,
        MiningConfig,
        bucket_label_from_delta,
        load_config,
    )
else:
    from .config import (
        ExperimentConfig,
        MiningConfig,
        bucket_label_from_delta,
        load_config,
    )

logger = logging.getLogger("preliminary.miner")


# --------------------------------------------------------------------------- #
# Records file IO helpers
# --------------------------------------------------------------------------- #
def _iter_record_lines(jsonl_path: str) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON objects from a single records.jsonl path. Skips malformed lines silently with WARN log."""
    try:
        f = open(jsonl_path, "r", encoding="utf-8")
    except FileNotFoundError as exc:
        logger.warning("records.jsonl missing -- skipping: %s (%s)", jsonl_path, exc)
        return
    with f:
        for lineno, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as exc:
                logger.warning("skip malformed %s:%d (%s)", jsonl_path, lineno, exc)


def _is_success_outcome(rec: dict[str, Any]) -> bool:
    oc = rec.get("outcome")
    if isinstance(oc, str):
        return oc.lower() == "success"
    if isinstance(oc, dict):
        return bool(oc.get("success", False))
    return False


def _numeric_delta(rec: dict[str, Any]) -> Optional[int]:
    sig = rec.get("signals") or {}
    dp = sig.get("delta")
    if isinstance(dp, bool):
        return None
    if isinstance(dp, int):
        return dp
    if isinstance(dp, float) and float(dp).is_integer():
        return int(dp)
    return None


def _injection_target_turn(B_rec: dict[str, Any]) -> Optional[int]:
    """Return canonical t_i preferring signals.injection_point, falling back to attack.target_turn,
    finally falling back to scanning trajectory.metadata.injection_visible_turn."""
    sig = B_rec.get("signals") or {}
    ip = sig.get("injection_point")
    if isinstance(ip, int) and ip >= 0:
        return ip
    atk = B_rec.get("attack") or {}
    tt = atk.get("target_turn")
    if isinstance(tt, int) and tt >= 0:
        return tt
    traj_meta = (B_rec.get("trajectory") or {}).get("metadata") or {}
    ivt = traj_meta.get("injection_visible_turn")
    if isinstance(ivt, int) and ivt >= 0:
        return ivt
    return None


def _payload_text(B_rec: dict[str, Any]) -> str:
    atk = B_rec.get("attack") or {}
    p = atk.get("payload", "")
    return str(p) if p else ""


def _dedup_key_for(task_id: str, payload: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{task_id}|{payload}".encode("utf-8")).hexdigest()
    short = digest[:16]
    full_key = f"{task_id}|sha256:{digest}"
    return short, full_key


def _git_sha(repo_root: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return out[:40]
    except Exception:                                                          # pragma: no cover
        return "unknown"


# --------------------------------------------------------------------------- #
# Benchmark-side env_info cache (used only opportunistically here;
# primary use happens in context_builder.py).
# --------------------------------------------------------------------------- #
_BENCH_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def load_benchmark_index(domain: str, repo_root: str) -> dict[str, dict[str, Any]]:
    """Return mapping {task_uid_hash_prefix10 -> {'instruction','env_info_raw'}} for one AgentDojo suite.

    The key matches the trailing hash component of EvoGuard-style uids built via
    ``evoguard.envs.toolsafe._task_uid('agentdojo', '<domain>', instruction)`` i.e.
    ``sha1(instruction).hexdigest()[:10]``. We precompute keys upfront rather than hashing
    instructions repeatedly.
    """

    cache_k = ("agentdojo:" + domain, repo_root)
    if cache_k in _BENCH_CACHE:
        return _BENCH_CACHE[cache_k]

    path = os.path.join(repo_root, "data", "toolsafe", "agentdojo-tragj", f"{domain}.json")
    idx: dict[str, dict[str, Any]] = {}
    if not os.path.isfile(path):
        logger.warning("benchmark json missing: %s", path)
        _BENCH_CACHE[cache_k] = idx
        return idx
    try:
        with open(path, "r", encoding="utf-8") as f:
            rec_list = json.load(f)
    except Exception as exc:
        logger.warning("failed reading %s: %s", path, exc)
        _BENCH_CACHE[cache_k] = idx
        return idx
    for r in rec_list:
        instr = (r.get("instruction") or "").strip()
        if not instr:
            continue
        hsh = hashlib.sha1(instr.encode("utf-8")).hexdigest()[:10]
        if hsh in idx:
            continue  # first-seen wins mirroring toolsafe._load semantics
        idx[hsh] = {
            "instruction": instr,
            "env_info_raw": r.get("env_info", ""),
        }
    _BENCH_CACHE[cache_k] = idx
    return idx


# --------------------------------------------------------------------------- #
# Scenario builder
# --------------------------------------------------------------------------- #
def _extract_domain_from_task_id(task_id: str) -> str:
    parts = task_id.split(":")
    if len(parts) == 3 and parts[0] == "agentdojo":
        return parts[1]
    return "?"


def _safe_actions_slice(actions: list[dict[str, Any]], upto_exclusive_end_index: int) -> list[dict[str, Any]]:
    """Return defensive shallow copy of actions[0:end]."""
    return [a for a in actions[:upto_exclusive_end_index]]


def _splice_poisoned_observation_into_action(
    base_action_dict: dict[str, Any],
    poisoned_observation_str: str,
) -> dict[str, Any]:
    new_a = dict(base_action_dict)
    new_a["observation"] = poisoned_observation_str
    # Drop bookkeeping-only fields irrelevant to prompt reconstruction to slim disk footprint.
    new_a.pop("turn", None)
    return new_a


def build_scenario(
    *,
    B_rec: dict[str, Any],
    A_rec: dict[str, Any],
    delta_value_orig: int,
    bucket_label: str,
    source_exp_name: str,
    miner_git_sha: str,
    dedup_short_hash: str,
) -> Optional[dict[str, Any]]:
    """Construct a single scenario dict per spec §2. Return None on structural failure."""

    task_id = B_rec.get("task_id") or ""
    if not task_id:
        logger.warning("B-rec missing task_id; dropping %r", B_rec.get("record_id"))
        return None

    domain = _extract_domain_from_task_id(task_id)

    T_i = _injection_target_turn(B_rec)
    if T_i is None:
        logger.warning("no resolvable injection-point for %s/%s",
                       source_exp_name, B_rec.get("record_id"))
        return None

    A_traj = (A_rec.get("trajectory") or {})
    A_actions = A_traj.get("actions") or []
    B_traj = (B_rec.get("trajectory") or {})
    B_actions = B_traj.get("actions") or []

    if T_i >= len(A_actions) or T_i >= len(B_actions):
        logger.warning(
            "T_i=%d beyond min(A_len=%d,B_len=%d) for %s/%s; dropping",
            T_i, len(A_actions), len(B_actions), source_exp_name, B_rec.get("record_id"),
        )
        return None

    poisoned_observation_at_T_i = B_actions[T_i].get("observation", "")
    if not poisoned_observation_at_T_i.strip():
        logger.warning("empty poisoned observation at T_i=%d for %s/%s; dropping",
                       T_i, source_exp_name, B_rec.get("record_id"))
        return None

    prefix_actions_pre_splice = [a for a in A_actions[: T_i + 1]]
    spliced_last_idx = len(prefix_actions_pre_splice) - 1
    spliced_prefix = []
    for j, act in enumerate(prefix_actions_pre_splice):
        if j != spliced_last_idx:
            cleaned = {k: v for k, v in act.items() if k != "turn"}
            spliced_prefix.append(cleaned)
        else:
            spliced_prefix.append(_splice_poisoned_observation_into_action(act, poisoned_observation_at_T_i))

    goal_instr = ((B_traj.get("metadata") or {}).get("instruction")) \
                 or ((A_traj.get("metadata") or {}).get("instruction")) \
                 or ""

    atk_block = B_rec.get("attack") or {}
    method_tag = atk_block.get("method", "")
    attack_id_val = atk_block.get("attack_id", "")

    orig_signals_ref = {
        k: v for k, v in (B_rec.get("signals") or {}).items()
        if k != "metadata"
    }
    md_meta = (B_rec.get("signals") or {}).get("metadata") or {}
    orig_signals_ref["metadata"] = md_meta

    scenario = {
        "scenario_id": "scn_" + dedup_short_hash,
        "bucket": bucket_label,
        "delta_value_orig": int(delta_value_orig),
        "source": {
            "exp_dir": source_exp_name,
            "round_id": B_rec.get("round_id"),
            "record_id_attacked": B_rec.get("record_id"),
            "attack_id": attack_id_val,
        },
        "task_id": task_id,
        "domain": domain,
        "goal_instruction": goal_instr,
        "method": method_tag,
        "context_prefix_actions": spliced_prefix,
        "injection_target_turn_index": T_i,
        "original_signals_for_reference": orig_signals_ref,
        "_provenance": {
            "miner_git_sha_of_code": miner_git_sha,
            "dedup_key_sha256_first16": dedup_short_hash,
            "clean_record_record_id": A_rec.get("record_id"),
            "poisoned_observation_sha256_first16":
                hashlib.sha256(poisoned_observation_at_T_i.encode()).hexdigest()[:16],
            "mined_at_utc_iso8601": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    return scenario


# --------------------------------------------------------------------------- #
# Main orchestrator class
# --------------------------------------------------------------------------- #
class BenchMiner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.repo_root = cfg.resolve_repo_root()
        self.rounds_root = os.path.join(self.repo_root, cfg.mining.rounds_root_relative_to_repo)
        self.bench_out_root = os.path.join(self.repo_root, cfg.mining.bench_output_dir_relative_to_repo)
        self.git_sha_now = _git_sha(self.repo_root)

    # ----- public API ----------------------------------------------------- #
    def mine(self) -> dict[str, Any]:
        """Run full pipeline returning final manifest-summary dict (also persisted to bench/manifest.json)."""

        os.makedirs(self.bench_out_root, exist_ok=True)
        logger.info("mining sources under %s : %s",
                    self.rounds_root, list(self.cfg.mining.source_experiments))

        candidates_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen_dedup_keys: set[str] = set()

        global_dropped_counts: Counter[str] = Counter()

        for src_exp in self.cfg.mining.source_experiments:
            edir = os.path.join(self.rounds_root, src_exp)
            if not os.path.isdir(edir):
                logger.warning("missing source exp dir -- skipping: %s", edir)
                continue
            for rnd in sorted(os.listdir(edir)):
                rp_jsonl = os.path.join(edir, rnd, "records.jsonl")
                if not os.path.isfile(rp_jsonl):
                    continue
                # Cache whole-round records map ONCE for both filtering AND pair-matching lookups.
                round_recs_by_id: dict[str, dict[str, Any]] = {}
                round_attacked_succ_candidates: list[dict[str, Any]] = []

                for rec_obj in _iter_record_lines(rp_jsonl):
                    rid = rec_obj.get("record_id")
                    if rid:
                        round_recs_by_id[str(rid)] = rec_obj
                    if rec_obj.get("kind") != "attacked":
                        continue
                    if self.cfg.mining.require_outcome_success and not _is_success_outcome(rec_obj):
                        continue
                    di = _numeric_delta(rec_obj) if self.cfg.mining.require_numeric_delta else \
                         ((rec_obj.get("signals") or {}).get("delta"))
                    blabel = bucket_label_from_delta(di)
                    if blabel is None:
                        if isinstance(di, int) and di > 4:
                            global_dropped_counts["out_of_scope_delta_gt_4"] += 1
                        elif di is None:
                            global_dropped_counts["non_numeric_delta_or_missing"] += 1
                        else:
                            global_dropped_counts[f"unmapped_delta_{di}"] += 1
                        continue
                    pl = _payload_text(rec_obj)
                    _, full_dedup = _dedup_key_for(str(rec_obj.get("task_id", "")), pl)
                    if full_dedup in seen_dedup_keys:
                        global_dropped_counts["duplicate_dropped"] += 1
                        continue
                    seen_dedup_keys.add(full_dedup)
                    rec_obj["_bucket_tmp"] = blabel
                    rec_obj["_dedup_full_tmp"] = full_dedup
                    rec_obj["_src_exp_tmp"] = src_exp
                    rec_obj["_short_hash_tmp"] = full_dedup.split("|")[1][-16:]
                    round_attacked_succ_candidates.append(rec_obj)

                if not round_attacked_succ_candidates:
                    continue

                # Build quick lookup of clean-kind records grouped by task_id WITHIN THIS SAME ROUND DIR.
                cleans_by_tid: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for rrec in round_recs_by_id.values():
                    if rrec.get("kind") == "clean":
                        tid = rrec.get("task_id")
                        if tid:
                            cleans_by_tid[tid].append(rrec)

                for cand in round_attacked_succ_candidates:
                    tid = cand.get("task_id")
                    A_partner = next(iter(cleans_by_tid.get(tid, [])), None)
                    if A_partner is None:
                        global_dropped_counts["dropped_no_clean_pair"] += 1
                        continue
                    sc = build_scenario(
                        B_rec=cand,
                        A_rec=A_partner,
                        delta_value_orig=int(_numeric_delta(cand)),
                        bucket_label=str(cand["_bucket_tmp"]),
                        source_exp_name=str(cand["_src_exp_tmp"]),
                        miner_git_sha=self.git_sha_now,
                        dedup_short_hash=str(cand["_short_hash_tmp"]),
                    )
                    if sc is None:
                        global_dropped_counts["scenario_build_failed"] += 1
                        continue
                    candidates_by_bucket[sc["bucket"]].append(sc)

                del round_recs_by_id, cleans_by_tid

        # Stratified subsample per bucket enforcing caps/domain fraction deterministically.
        sampled_buckets: dict[str, list[dict[str, Any]]] = {}
        domain_count_realized: dict[str, dict[str, int]] = {}

        rng_seed_base = self.cfg.seed ^ self.cfg.mining.sampling_seed
        for bname in self.cfg.mining.buckets:
            pool = sorted(candidates_by_bucket.get(bname, []),
                          key=lambda x: x["scenario_id"])
            chosen_pool, dom_counter = _stratified_select_within_bucket(
                pool=pool,
                cap=self.cfg.mining.cap_per_bucket,
                max_domain_fraction=self.cfg.mining.max_domain_fraction_in_bucket,
                seed=rng_seed_base + sum(ord(c) * 7 ** i for i, c in enumerate(bname)),
            )
            sampled_buckets[bname] = chosen_pool
            domain_count_realized[bname] = dom_counter

        # Persist outputs.
        bucket_summaries: dict[str, Any] = {}
        total_written = 0
        for bname in self.cfg.mining.buckets:
            chosen = sampled_buckets.get(bname, [])
            fname = f"corpus_{bname}.jsonl"
            fp = os.path.join(self.bench_out_root, fname)
            n_unique_methods_seen = len({c["method"] for c in chosen})
            with open(fp, "w", encoding="utf-8") as fout:
                for scen in chosen:
                    fout.write(json.dumps(scen, ensure_ascii=False) + "\n")
            total_written += len(chosen)
            bucket_summaries[bname] = {
                "path_relative_to_repo": os.path.relpath(fp, self.repo_root),
                "path_absolute": fp,
                "n_scenarios": len(chosen),
                "pool_size_before_sampling": len(candidates_by_bucket.get(bname, [])),
                "unique_method_tags": n_unique_methods_seen,
                "domain_distribution": dict(Counter(s["domain"] for s in chosen).most_common()),
                "max_domain_share_observed": (
                    max((Counter(c["domain"] for c in chosen)).values()) /
                    max(len(chosen), 1)
                ) if chosen else 0.0,
                "_methods_used_top15": [
                    {"method": m, "count": cnt}
                    for m, cnt in Counter(c["method"] for c in chosen).most_common(15)
                ],
            }

        manifest = {
            "schema_version": "preliminary_bench_v1",
            "experiment_name_under_config": self.cfg.experiment_name,
            "mined_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "miner_runtime_git_sha": self.git_sha_now,
            "repo_root_resolved_to": self.repo_root,
            "source_experiments_mined": list(self.cfg.mining.source_experiments),
            "buckets": bucket_summaries,
            "global_dropped_counts_breakdown": dict(global_dropped_counts),
            "total_scenarios_persisted": total_written,
            "sampling_parameters": {
                "cap_per_bucket_configured": self.cfg.mining.cap_per_bucket,
                "max_domain_fraction_in_bucket_configured": self.cfg.mining.max_domain_fraction_in_bucket,
                "seed_basis_int": rng_seed_base,
            },
        }

        mp = os.path.join(self.bench_out_root, "manifest.json")
        with open(mp, "w", encoding="utf-8") as fout_manifest:
            json.dump(manifest, fout_manifest, ensure_ascii=False, indent=2)

        logger.info("wrote manifest -> %s ; total scenarios written=%d", mp, total_written)
        return manifest


# --------------------------------------------------------------------------- #
# Sampling helper
# --------------------------------------------------------------------------- #
import random


def _stratified_select_within_bucket(
    *,
    pool: list[dict[str, Any]],
    cap: int,
    max_domain_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deterministic two-phase sampler honouring bucket cap + per-domain ceiling.

      Phase A (strict): iterate pool in shuffled order adding candidates whose domain
        has not yet reached ``ceil(max_domain_fraction * cap)``. Stops at ``cap`` or end of pool.
      Phase B (relaxed once if Phase A stalls short of cap AND strict ceiling blocked further adds):
        relax ceiling to 0.60 (or original value if higher) and continue iterating remaining candidates.

    Returns (selected_list, domain_counter_dict).
    """

    import math

    if not pool:
        return [], {}

    rng = random.Random(seed)
    order = list(pool)
    rng.shuffle(order)

    selected: list[dict[str, Any]] = []
    dom_cnt: Counter[str] = Counter()
    chosen_ids: set[str] = set()

    def fill_once(max_frac: float) -> bool:
        """Single pass over remaining unselected items; returns True if anything was added."""
        progressed = False
        hard_ceil = math.ceil(max_frac * cap)
        for cand in order:
            if len(selected) >= cap:
                break
            sid = cand["scenario_id"]
            if sid in chosen_ids:
                continue
            dmn = cand["domain"]
            if dom_cnt[dmn] >= hard_ceil:
                continue
            selected.append(cand)
            dom_cnt[dmn] += 1
            chosen_ids.add(sid)
            progressed = True
        return progressed

    # Strict pass.
    fill_once(max_domain_fraction)

    # Relaxation passes: progressively raise ceiling until either cap met OR ceiling hits 1.0.
    relaxed_frac = max(max_domain_fraction, 0.60)
    relaxation_used = False
    while len(selected) < cap and relaxed_frac <= 1.000001 and not relaxation_used:
        added_anything_more = fill_once(relaxed_frac)
        if not added_anything_more:
            break
        if relaxed_frac >= 0.9999:
            relaxation_used = True   # only one full-fill attempt after relaxing upward all the way
        else:
            new_relaxed = min(1.0, relaxed_frac + 0.10)
            if abs(new_relaxed - relaxed_frac) < 1e-6:
                relaxation_used = True
            else:
                relaxed_frac = new_relaxed

    return selected, dict(dom_cnt)


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m preliminary.miner")
    ap.add_argument("--config", default="configs/preliminary_entropy.yaml")
    ap.add_argument("--loglevel", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    lvl = args.loglevel or cfg.logging_level.upper()
    logging.basicConfig(level=lvl, format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s")

    mn = BenchMiner(cfg)
    manifest = mn.mine()
    print("\n=== MINING SUMMARY ===")
    print(f"total scenarios persisted: {manifest['total_scenarios_persisted']}")
    for bn, bs in manifest["buckets"].items():
        dd = ", ".join(f"{d}:{n}" for d, n in bs["domain_distribution"].items())
        print(f"  [{bn}] n={bs['n_scenarios']:>2} methods={bs['unique_method_tags']:>3} domains=[{dd}]"
              f" max_share={bs['max_domain_share_observed']:.2f}")
    gd = manifest["global_dropped_counts_breakdown"]
    if gd:
        print("--- drops ---")
        for reason, ct in sorted(gd.items(), key=lambda kv: kv[0]):
            print(f"  {reason}: {ct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
