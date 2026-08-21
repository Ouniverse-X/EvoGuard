"""Overnight launch harness for the Strategy-① Turn-Shift Replant synthesizer.

Custom orchestrator wrapping :mod:`evoguard.process.synth_replant` so that each
seed gets a *per-seed* ``defender_generate_fn`` closure that faithfully
reconstructs (task, tools, history) from the workspace benchmark index. This is
necessary because ``synth_replant.run_pipeline`` exposes only a context-action
list to the defender callback, but the real :class:`LLMDefenseAgent.decide`
needs ``Task`` + ``ToolSpec`` list to build its system prompt.

Reconstruction contract:
  * ``goal_instruction``  ← bench scenarios (loaded by ``scenario_id``)
  * ``env_info`` (tools)  ← ``data/toolsafe/agentdojo-tragj/workspace.json``
                             matched by exact instruction text
  * ``history``           ← augmented_ctx actions passed by validate_forward_replay

Output: ``bench/scenarios/_synthetic/bucket_d{3,4}_synth.jsonl``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evoguard.core.types import Action, Task
from evoguard.agents.llm_agent import LLMDefenseAgent
from evoguard.config import DefenseConfig, LLMConfig
from evoguard.envs.tool_parsing import parse_env_info
from evoguard.process import synth_replant as sr
from evoguard.process.bench_schema import iter_load_scenarios
from evoguard.process.bench_constants import DOMAIN_SCOPE_RESTRICTION_V2

DEFENDER_URL = os.environ.get("SYNTH_DEFENDER_URL", "http://localhost:8000/v1")
DEFENDER_MODEL = os.environ.get("SYNTH_DEFENDER_MODEL", "qwen2.5-7b-it")
JUDGE_URL = os.environ.get("SYNTH_JUDGE_URL", "http://localhost:8002/v1")
JUDGE_MODEL = os.environ.get("SYNTH_JUDGE_MODEL", "llama3-8b-judge")
PER_BUCKET_QUOTA = int(os.environ.get("SYNTH_PER_BUCKET_QUOTA", "40"))
SOURCE_BUCKETS = ("imm", "d1", "d2")
TARGET_BUCKETS = ("d3", "d4")


def build_rounds_actions_index(rounds_root: Path) -> dict[str, list[dict]]:
    """Single-pass scan of rounds/ building {record_id: actions} index.

    Replaces the catastrophically slow per-seed re-rglob inside
    ``synth_replant._load_rounds_safe`` (O(seeds × files × lines)).
    """
    idx: dict[str, list[dict]] = {}
    if not rounds_root.is_dir():
        return idx
    files = list(rounds_root.rglob("*.jsonl"))
    files.sort()
    for jf in files:
        try:
            with open(jf, encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    obj = json.loads(ln)
                    rid = obj.get("record_id") or obj.get("id") or ""
                    if not rid:
                        continue
                    srid = str(rid)
                    if srid in idx:
                        continue
                    traj = obj.get("trajectory") or {}
                    acts = traj.get("actions") or []
                    if isinstance(acts, list) and acts:
                        idx[srid] = acts
        except Exception:
            continue
    return idx


def load_seeds_fast(
    bench_root: Path,
    rounds_root: Path,
    source_buckets: tuple[str, ...],
) -> list[sr.SynthSeed]:
    """Build SynthSeeds in bulk using a pre-indexed clean-traj lookup.

    Functionally equivalent to ``synth_replant.extract_seeds`` but runs in a
    single O(files × lines) pass instead of re-rglobbing per seed.
    """
    acts_idx = build_rounds_actions_index(rounds_root)
    seeds: list[sr.SynthSeed] = []
    for bkt in source_buckets:
        scen_path = bench_root / "scenarios" / f"bucket_{bkt}.jsonl"
        if not scen_path.exists():
            continue
        for row in iter_load_scenarios(str(scen_path), exclude_synthetic=True):
            if row.get("domain") != DOMAIN_SCOPE_RESTRICTION_V2:
                continue
            ctx = row.get("context_prefix_actions") or []
            iti = row.get("injection_target_turn_index", 0)
            pot = row.get("poisoned_observation_text") or ""
            prov_blk = row.get("_provenance", {}) or {}
            cri_id = prov_blk.get("clean_record_record_id", "")
            clean_acts = acts_idx.get(cri_id, [])
            seeds.append(
                sr.SynthSeed(
                    origin_scenario_id=row.get("scenario_id", "unknown_seed"),
                    bucket_origin=bkt,
                    delta_value_orig=int(
                        row.get("delta_value_orig", row.get("signals_ref", {}).get("delta", 0))
                    ),
                    domain=row.get("domain", DOMAIN_SCOPE_RESTRICTION_V2),
                    task_id=row.get("task_id", ""),
                    toolkit_signature=row.get("toolkit_signature", ""),
                    channel_class_hint="",
                    canonical_technique_id=row.get("canonical_technique_id", ""),
                    poisoned_observation_text=pot,
                    context_prefix_actions_verbatim=ctx,
                    injection_target_turn_index_original=int(iti),
                    clean_trajectory_actions_full=clean_acts,
                    metadata_notes={
                        "clean_twin_status": "loaded" if clean_acts else "missing",
                        "extraction_source_path": str(scen_path),
                    },
                )
            )
    return seeds


def load_workspace_env_info_map() -> dict[str, str]:
    ws_path = REPO_ROOT / "data" / "toolsafe" / "agentdojo-tragj" / "workspace.json"
    with open(ws_path, encoding="utf-8") as f:
        ws = json.load(f)
    out: dict[str, str] = {}
    for r in ws:
        instr = (r.get("instruction") or "").strip()
        if instr and instr not in out:
            out[instr] = r.get("env_info", "") or ""
    return out


def load_goal_by_scenario_id() -> dict[str, str]:
    out: dict[str, str] = {}
    bench_root = REPO_ROOT / "bench" / "scenarios"
    for bkt in SOURCE_BUCKETS + TARGET_BUCKETS:
        p = bench_root / f"bucket_{bkt}.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["scenario_id"]] = (rec.get("goal_instruction") or "").strip()
    return out


def build_defense_agent() -> LLMDefenseAgent:
    llm = LLMConfig(
        backend="openai",
        base_url=DEFENDER_URL,
        api_key="EMPTY",
        model=DEFENDER_MODEL,
        temperature=0.0,
        max_tokens=256,
    )
    return LLMDefenseAgent(DefenseConfig(llm=llm, system_prompt=None))


def make_defender_fn(agent: LLMDefenseAgent, goal_instruction: str, env_info: str):
    """Build a per-seed defender callback matching validate_forward_replay contract.

    Returns ``def(ctx_actions: list[dict]) -> dict`` whose dict carries
    ``thought`` + ``tool_call`` (or ``None`` on terminal/error).
    """
    tools = parse_env_info(env_info) if env_info else []
    task = Task(
        task_id="synth-replant",
        instruction=goal_instruction,
        tool_names=[t.name for t in tools],
    )

    def fn(ctx_actions: list[dict]) -> dict:
        history = [Action.from_dict(a) for a in ctx_actions]
        try:
            act = agent.decide(task, tools, history)
        except Exception as exc:  # keep synthesizer resilient to transient LLM errors
            return {"thought": f"defender-error: {exc}", "tool_call": None}
        return {
            "thought": act.thought,
            "tool_call": act.tool_call.to_dict() if act.tool_call else None,
        }

    return fn


def main() -> None:
    t0 = time.time()
    env_info_map = load_workspace_env_info_map()
    goal_map = load_goal_by_scenario_id()
    print(f"[synth] workspace instructions indexed: {len(env_info_map)}", flush=True)
    print(f"[synth] bench scenarios indexed: {len(goal_map)}", flush=True)

    seeds = load_seeds_fast(
        bench_root=(REPO_ROOT / "bench"),
        rounds_root=(REPO_ROOT / "rounds"),
        source_buckets=SOURCE_BUCKETS,
    )
    seeds_with_clean = [s for s in seeds if s.clean_trajectory_actions_full]
    print(
        f"[synth] seeds loaded: {len(seeds)} | with clean-traj twin: {len(seeds_with_clean)}",
        flush=True,
    )

    agent = build_defense_agent()
    judge_fn = None
    try:
        judge_fn = sr.make_live_judge_closure(
            endpoint_url=JUDGE_URL,
            model_id=JUDGE_MODEL,
            api_key="EMPTY",
        )
        print(f"[synth] live judge wired: {JUDGE_URL} ({JUDGE_MODEL})", flush=True)
    except Exception as exc:
        judge_fn = None
        print(f"[synth] WARN judge wiring failed ({exc}); proceeding WITHOUT live judge", flush=True)

    accepted: dict[str, list] = {bkt: [] for bkt in TARGET_BUCKETS}
    near_misses: list = []
    stats = {"seeds_skipped_no_goal": 0, "seeds_skipped_no_envinfo": 0, "defender_calls": 0}

    quota_full = {bkt: False for bkt in TARGET_BUCKETS}
    for si, seed in enumerate(seeds_with_clean):
        goal = goal_map.get(seed.origin_scenario_id, "")
        if not goal:
            stats["seeds_skipped_no_goal"] += 1
            continue
        env_info = env_info_map.get(goal, "")
        if not env_info:
            stats["seeds_skipped_no_envinfo"] += 1
            continue
        dfn = make_defender_fn(agent, goal, env_info)

        for bkt in TARGET_BUCKETS:
            if quota_full[bkt]:
                continue
            tgt = int(bkt.lstrip("d"))
            for cand in sr.enumerate_candidates(
                seed=seed, target_deltas=(tgt,), near_miss_sink=near_misses
            ):
                res = sr.validate_forward_replay(
                    candidate=cand,
                    defender_generate_fn=dfn,
                    judge_fn=judge_fn,
                    near_miss_sink=near_misses,
                )
                stats["defender_calls"] += 1
                if res.accepted:
                    accepted[bkt].append(res)
                    if len(accepted[bkt]) >= PER_BUCKET_QUOTA:
                        quota_full[bkt] = True
                    break  # one acceptance per (seed, bucket) — diversity
            if quota_full[bkt]:
                break

        if all(quota_full.values()):
            print("[synth] all target bucket quotas full; stopping early", flush=True)
            break

        if (si + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[synth] progress seed {si + 1}/{len(seeds_with_clean)} | "
                f"accepted d3={len(accepted['d3'])} d4={len(accepted['d4'])} | "
                f"near_misses={len(near_misses)} | defender_calls={stats['defender_calls']} | "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    # Diversify + write
    out_root = REPO_ROOT / "bench" / "scenarios" / "_synthetic"
    out_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for bkt, results in accepted.items():
        curated = sr.assemble_diversified_batch(
            results=results,
            family_ceiling=0.35,
            per_bucket_quota=PER_BUCKET_QUOTA,
            rng_random_state=0,
        )
        out_path = out_root / f"bucket_{bkt}_synth.jsonl"
        with open(out_path, "w", encoding="utf-8") as wh:
            for cr in curated:
                wh.write(sr._serialize_accepted_result(cr) + "\n")
        written[bkt] = len(curated)

    elapsed = time.time() - t0
    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "seeds_total": len(seeds),
        "seeds_with_clean_traj": len(seeds_with_clean),
        "accepted_before_diversification": {bkt: len(lst) for bkt, lst in accepted.items()},
        "written_after_diversification": written,
        "near_miss_logged": len(near_misses),
        "stats": stats,
    }
    print("[synth] DONE summary:", json.dumps(summary, indent=2), flush=True)

    summary_path = out_root / "_synth_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[synth] summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
