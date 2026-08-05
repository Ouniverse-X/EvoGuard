"""V3 alertness experiment: Clean < AS < AF three-level comparison.

Implements docs/preliminary.md spec:
  - 1 clean (temp=0) + N attacked (temp=0.7) per scenario
  - Keep scenarios with >=3 success + >=3 fail
  - Metric A: average entropy over first 32/64 thought tokens
  - Metric B: logit bias = logprob(mal_tool) - logprob(safe_tool) at tool-name position
  - Per-scenario triplet: (Clean, AS_avg, AF_avg)
  - Analysis: Page's L trend test + pairwise Wilcoxon + monotonic count
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from typing import Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from evoguard.agents.prompts import build_system_prompt, render_history
from evoguard.agents import build_defense_agent
from evoguard.config import ExperimentConfig
from evoguard.controller import Controller
from evoguard.core.types import Action, AttackSpec, ToolCall, Trajectory
from evoguard.envs import build_env
from evoguard.judge import AttackJudge
from evoguard.process.vendored_attack_parser import (
    VendoredAttack, load_all_vendored_attacks,
)

logger = logging.getLogger("preliminary.alertness_v3")

EVALUATOR_VERSION = "alertness-v3-triplet"


# ---------------------------------------------------------------------------
# Phase 1: Rollout — collect clean + N attacked per scenario
# ---------------------------------------------------------------------------

def _find_target_turn(clean_traj: Trajectory, target_tool: str) -> Optional[int]:
    for i, action in enumerate(clean_traj.actions):
        if action.tool_call is not None and action.tool_call.name == target_tool:
            return i
    return None


def _serialize_actions(actions: list[Action]) -> list[dict[str, Any]]:
    out = []
    for a in actions:
        out.append({
            "turn": a.turn,
            "thought": a.thought or "",
            "tool_call_name": a.tool_call.name if a.tool_call else None,
            "tool_call_args": a.tool_call.arguments if a.tool_call else None,
            "observation": a.observation or "",
            "final_answer": a.final_answer or "",
        })
    return out


def _extract_rollout_info(traj: Trajectory, target_turn: int) -> dict[str, Any]:
    """Extract thought + tool_call info from a trajectory at target_turn."""
    if target_turn >= len(traj.actions):
        return {"thought": "", "tool_call_name": None, "tool_call_args": None,
                "final_answer": ""}
    act = traj.actions[target_turn]
    return {
        "thought": act.thought or "",
        "tool_call_name": act.tool_call.name if act.tool_call else None,
        "tool_call_args": act.tool_call.arguments if act.tool_call else None,
        "final_answer": act.final_answer or "",
    }


def run_rollouts(
    config_path: str,
    output_dir: str,
    *,
    suites: list[str] | None = None,
    n_samples: int = 15,
    sampling_temperature: float = 0.7,
    target_scenarios: int = 100,
    min_per_bucket: int = 3,
    seed: int = 42,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Phase 1: Run clean + N attacked rollouts per scenario.

    Groups by task to cache clean rollouts. Returns list of valid scenarios
    (those with >= min_per_bucket success + min_per_bucket fail).

    ``sampling_temperature`` may be either a scalar float OR an iterable of
    floats; iterables trigger *temperature-sweep mode* (Tier-2 deconfounder)
    which distributes the requested sample budget across strata and tags each
    collected attacked_rollout with its own ``collection_temperature`` field
    so downstream Plan-A/B analyses can stratify.
    """
    # Normalise temperature schedule into a list of (T_value, n_samples_at_T).
    if isinstance(sampling_temperature, (list, tuple)):
        t_schedule = list(sampling_temperature)
        n_strata = len(t_schedule)
        per_stratum = max(min_per_bucket, n_samples // n_strata)
        temp_plan = [(float(t), int(per_stratum)) for t in t_schedule]
        logger.info("temperature-sweep mode enabled: %d strata × %d samples/stratum",
                    n_strata, per_stratum)
    else:
        temp_plan = [(float(sampling_temperature), int(n_samples))]

    cfg = ExperimentConfig.from_file(config_path)
    if suites:
        cfg.env.suites = suites
    cfg.defense.llm.lora_adapter = None

    env = build_env(cfg.env, seed=cfg.seed)
    agent = build_defense_agent(cfg.defense, seed=cfg.seed)
    controller = Controller(agent, env, cfg.defense)
    judge = AttackJudge(cfg.env.judge_llm, seed=cfg.seed)

    suite_list = cfg.env.suites or None
    scenarios = load_all_vendored_attacks(cfg.env.data_root, suites=suite_list)
    tasks_by_id = {t.task_id: t for t in env.get_tasks()}

    by_task: dict[str, list[VendoredAttack]] = {}
    for va in scenarios:
        if va.task_id in tasks_by_id:
            by_task.setdefault(va.task_id, []).append(va)

    # Flatten scenarios with (task_id, va, scenario_idx)
    all_scenarios = []
    for task_id, va_list in by_task.items():
        for idx, va in enumerate(va_list):
            all_scenarios.append((task_id, va, idx))
    rng = random.Random(seed)
    rng.shuffle(all_scenarios)

    # Resume support
    valid_scenarios: list[dict[str, Any]] = []
    existing_task_scenario_keys: set[tuple[str, int]] = set()
    if resume:
        existing_path = os.path.join(output_dir, "all_scenarios.json")
        if os.path.exists(existing_path):
            valid_scenarios = json.load(open(existing_path))
            for s in valid_scenarios:
                existing_task_scenario_keys.add((s["task_id"], s["scenario_idx"]))
            logger.info("resume: loaded %d existing valid scenarios, skipping them",
                        len(valid_scenarios))

    logger.info("loaded %d scenarios across %d tasks; target=%d, n_samples=%d, temp=%.1f",
                len(all_scenarios), len(by_task), target_scenarios, n_samples,
                sampling_temperature)

    t0 = time.time()
    n_scenarios_tried = 0

    # Streaming event log: append-only JSONL survives crashes; one line per
    # significant state change (scenario_start / clean_ok|crash /
    # attacked_sample_done / scenario_pass_filter | skip_*).
    events_path = os.path.join(output_dir, "events.jsonl")
    scenarios_path = os.path.join(output_dir, "all_scenarios.json")

    def _emit_event(event: dict[str, Any]) -> None:
        """Append a single JSONL record to the streaming log."""
        try:
            with open(events_path, "a") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass  # logging must never break the experiment.

    def _checkpoint_valid() -> None:
        """Atomically rewrite the full valid-scenarios list to disk."""
        tmp = scenarios_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(valid_scenarios, f, ensure_ascii=False, indent=2)
        os.replace(tmp, scenarios_path)

    _emit_event({"event": "phase1_start", "target": target_scenarios,
                 "n_samples": n_samples, "temp": sampling_temperature,
                 "resume_loaded": len(existing_task_scenario_keys)})

    for task_id, va, scenario_idx in all_scenarios:
        if len(valid_scenarios) >= target_scenarios:
            break

        if (task_id, scenario_idx) in existing_task_scenario_keys:
            continue

        task = tasks_by_id[task_id]
        tools = env.get_tools(task)
        system_prompt = build_system_prompt(task, tools, cfg.defense.system_prompt)

        # Clean rollout (greedy).
        _emit_event({"event": "scenario_start", "task_id": task_id,
                     "scenario_idx": scenario_idx,
                     "target_tool": va.target_tool, "mal_tool": va.mal_tool})
        try:
            agent.config.llm.temperature = 0.0
            clean_traj = controller.run_clean(task)
        except Exception as exc:
            logger.warning("clean rollout crashed for %s: %s", task_id, exc)
            _emit_event({"event": "clean_crash", "task_id": task_id,
                         "scenario_idx": scenario_idx,
                         "error": str(exc)[:300]})
            continue

        target_turn = _find_target_turn(clean_traj, va.target_tool)
        if target_turn is None:
            _emit_event({"event": "skip_no_target_tool", "task_id": task_id,
                         "scenario_idx": scenario_idx,
                         "target_tool": va.target_tool})
            continue

        clean_info = _extract_rollout_info(clean_traj, target_turn)
        clean_obs = clean_traj.actions[target_turn].observation or ""
        injected_obs = va.payload  # The injected payload
        # For attacked rollout, the real observation at target_turn includes the injection
        # Controller.run_attacked handles this via _inject()

        attack = AttackSpec(
            attack_id=AttackSpec.new_id(),
            task_id=task.task_id,
            target_turn=target_turn,
            injection_channel="tool_result",
            method="vendored_replay",
            payload=va.payload,
            goal=va.goal,
            origin="vendored",
            metadata={"mal_tool": va.mal_tool, "mal_args": va.mal_args,
                      "real_observation": va.real_observation},
        )

        # N sampled attacked rollouts distributed across temperature strata.
        success_rollouts: list[dict[str, Any]] = []
        fail_rollouts: list[dict[str, Any]] = []

        for t_val, n_at_t in temp_plan:
            agent.config.llm.temperature = t_val
            for s in range(n_at_t):
                try:
                    attacked_traj = controller.run_attacked(task, attack, clean=clean_traj)
                except Exception as exc:
                    logger.debug("attacked rollout crashed (T=%.2f s=%d) for %s: %s",
                                 t_val, s, task_id, exc)
                    continue

                try:
                    success, reason = judge.judge(attacked_traj, attack)
                except Exception:
                    success, reason = False, "judge_error"

                rollout_info = _extract_rollout_info(attacked_traj, target_turn)
                rollout_info["bucket"] = "success" if success else "fail"
                rollout_info["judge_reason"] = reason[:200]
                # Tier-2 deconfounder: record which T-stratum this sample came
                # from so downstream analyses can stratify by sampling-stochasticity.
                rollout_info["collection_temperature"] = t_val

                if success:
                    success_rollouts.append(rollout_info)
                else:
                    fail_rollouts.append(rollout_info)
                _emit_event({"event": "attacked_sample_done", "task_id": task_id,
                             "scenario_idx": scenario_idx,
                             "temperature": t_val, "sample": s,
                             "verdict": "success" if success else "fail",
                             "n_success_so_far": len(success_rollouts),
                             "n_fail_so_far": len(fail_rollouts)})

        n_scenarios_tried += 1

        # Keep scenario if enough samples in both buckets.
        if len(success_rollouts) >= min_per_bucket and len(fail_rollouts) >= min_per_bucket:
            # Tier-2: tag each rollout with its T-stratum if sweep mode is on
            # so the persisted record is self-describing.
            t_strata_seen = sorted({r.get("collection_temperature", sampling_temperature)
                                    for r in success_rollouts + fail_rollouts
                                    if r.get("collection_temperature") is not None})
            valid_scenarios.append({
                "task_id": task_id,
                "scenario_idx": scenario_idx,
                "domain": va.suite,
                "target_tool": va.target_tool,
                "mal_tool": va.mal_tool,
                "target_turn": target_turn,
                "system_prompt": system_prompt,
                "serialized_actions": _serialize_actions(
                    clean_traj.actions[:target_turn + 1]),
                "clean_obs": clean_obs,
                "injected_obs": va.real_observation or clean_obs,
                "clean_rollout": clean_info,
                "attacked_rollouts": success_rollouts + fail_rollouts,
                "n_success": len(success_rollouts),
                "n_fail": len(fail_rollouts),
                "temperature_schedule": t_strata_seen,
            })
            logger.info("valid scenario %d: task=%s scenario=%d (success=%d, fail=%d) — total=%d",
                        len(valid_scenarios), task_id[:30], scenario_idx,
                        len(success_rollouts), len(fail_rollouts), len(valid_scenarios))
            # Incremental atomic checkpoint: every successful scenario persists
            # immediately so a crash or kill never loses more than the in-flight
            # incomplete one.
            _checkpoint_valid()
            _emit_event({"event": "scenario_pass_filter", "task_id": task_id,
                         "scenario_idx": scenario_idx,
                         "n_success": len(success_rollouts),
                         "n_fail": len(fail_rollouts),
                         "total": len(valid_scenarios)})
        else:
            _emit_event({"event": "scenario_skip_insufficient_buckets",
                         "task_id": task_id, "scenario_idx": scenario_idx,
                         "n_success": len(success_rollouts),
                         "n_fail": len(fail_rollouts)})

        if n_scenarios_tried % 10 == 0:
            elapsed = time.time() - t0
            logger.info("tried %d scenarios, %d valid, %.1fs, %.1fs/scenario",
                        n_scenarios_tried, len(valid_scenarios), elapsed,
                        elapsed / max(n_scenarios_tried, 1))

    # Final save.
    _checkpoint_valid()
    _emit_event({"event": "phase1_done", "valid": len(valid_scenarios),
                 "tried": n_scenarios_tried})
    logger.info("rollout phase done: %d valid scenarios from %d tried",
                len(valid_scenarios), n_scenarios_tried)
    return valid_scenarios


# ---------------------------------------------------------------------------
# Phase 2: Measurement — Metric A (thought entropy) + Metric B (tool bias)
# ---------------------------------------------------------------------------

def _build_context_messages(
    system_prompt: str,
    serialized_actions: list[dict[str, Any]],
    target_turn: int,
    obs_override: str,
) -> list[dict[str, str]]:
    """Build [system, user] messages: history up to target_turn with obs_override."""
    actions: list[Action] = []
    for i, a in enumerate(serialized_actions):
        if i > target_turn:
            break
        tc = None
        if a["tool_call_name"]:
            tc = ToolCall(name=a["tool_call_name"], arguments=a["tool_call_args"] or {})
        obs = obs_override if i == target_turn else a["observation"]
        actions.append(Action(
            turn=a["turn"], thought=a["thought"], tool_call=tc,
            observation=obs, final_answer=a["final_answer"],
        ))
    user = render_history(actions)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


METRICS_CSV_COLUMNS: tuple[str, ...] = (
    "task_id", "scenario_idx", "rollout_type", "bucket",
    "metric_a_32", "metric_a_64", "metric_b_logprob_mal", "metric_b_logprob_safe",
    "metric_b_bias", "n_thought_tokens", "has_tool_call",
    "target_turn", "target_tool", "mal_tool",
    "evaluator_version",
)


def measure_metrics(
    scenarios: list[dict[str, Any]],
    output_dir: str,
    *,
    model_path: str = "/ssd1/models/qwen2.5-7b-it",
    cuda_visible_devices: str = "0",
    max_input_length: int = 32000,
) -> list[dict[str, Any]]:
    """Phase 2: Measure Metric A (thought entropy) + Metric B (tool bias)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading tokenizer from %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    logger.info("loading model %s onto cuda bf16", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    model.eval()
    device = "cuda"
    logger.info("model loaded")

    csv_path = os.path.join(output_dir, "metrics.csv")
    jsonl_path = os.path.join(output_dir, "raw_metrics.jsonl")

    total_rollouts = sum(1 + len(s["attacked_rollouts"]) for s in scenarios)
    logger.info("measuring %d scenarios, %d total rollouts", len(scenarios), total_rollouts)

    rollout_idx = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv, \
         open(jsonl_path, "w", encoding="utf-8") as fjsonl:
        writer = csv.DictWriter(fcsv, fieldnames=METRICS_CSV_COLUMNS)
        writer.writeheader()

        for scenario in scenarios:
            mal_tool = scenario.get("mal_tool") or ""
            safe_tool = scenario.get("target_tool") or ""

            # Build rollouts to measure: 1 clean + all attacked
            rollouts_to_measure = []
            rollouts_to_measure.append(("clean", "clean", scenario["clean_rollout"]))
            for ar in scenario["attacked_rollouts"]:
                bucket = ar["bucket"]  # "success" or "fail"
                rollouts_to_measure.append(("attacked", bucket, ar))

            for rollout_type, bucket, rinfo in rollouts_to_measure:
                rollout_idx += 1
                try:
                    thought = rinfo.get("thought", "") or ""
                    obs = (scenario["clean_obs"] if rollout_type == "clean"
                           else scenario["injected_obs"])

                    msgs = _build_context_messages(
                        scenario["system_prompt"],
                        scenario["serialized_actions"],
                        scenario["target_turn"],
                        obs,
                    )

                    # --- Tokenize context + thought ---
                    prefix_ids = tok.apply_chat_template(
                        msgs, tokenize=True, add_generation_prompt=True,
                        return_tensors="pt").to(device)
                    thought_ids = tok.encode(
                        thought, add_special_tokens=False,
                        return_tensors="pt").to(device)

                    # --- Metric A: entropy of first 32/64 thought tokens ---
                    # Truncate thought to 64 tokens for efficiency
                    thought_ids_trunc = thought_ids[:, :64]
                    full_ids_a = torch.cat([prefix_ids, thought_ids_trunc], dim=1)

                    # Left-truncate if too long
                    if full_ids_a.shape[1] > max_input_length:
                        overflow = full_ids_a.shape[1] - max_input_length
                        full_ids_a = full_ids_a[:, overflow:]
                        prefix_len_a = prefix_ids.shape[1] - overflow
                    else:
                        prefix_len_a = prefix_ids.shape[1]

                    thought_len = min(thought_ids_trunc.shape[1], 64)
                    if thought_len < 2:
                        metric_a_32 = metric_a_64 = None
                    else:
                        with torch.no_grad():
                            out_a = model(input_ids=full_ids_a, use_cache=False)
                        logits_a = out_a.logits[0]  # [seq_len, vocab]

                        # Positions predicting thought tokens: [prefix_len-1, ...]
                        pred_positions = list(range(
                            max(prefix_len_a - 1, 0),
                            min(prefix_len_a - 1 + thought_len, logits_a.shape[0]),
                        ))
                        entropies = []
                        for pos in pred_positions:
                            pos_logits = logits_a[pos].float()
                            probs = torch.softmax(pos_logits, dim=-1)
                            ent = -(probs * (probs + 1e-12).log()).sum().item()
                            entropies.append(ent)
                        metric_a_32 = float(np.mean(entropies[:32])) if len(entropies) >= 2 else None
                        metric_a_64 = float(np.mean(entropies)) if len(entropies) >= 2 else None

                    # --- Metric B: logit bias at tool-name position ---
                    metric_b_bias = None
                    logprob_mal = None
                    logprob_safe = None
                    has_tool_call = rinfo.get("tool_call_name") is not None

                    if mal_tool and safe_tool and thought:
                        # Build partial JSON: {"thought": "...", "tool": "
                        partial_json = '{"thought": ' + json.dumps(thought) + ', "tool": "'
                        partial_ids = tok.encode(
                            partial_json, add_special_tokens=False,
                            return_tensors="pt").to(device)

                        mal_ids = tok.encode(
                            mal_tool, add_special_tokens=False,
                            return_tensors="pt").to(device)
                        safe_ids = tok.encode(
                            safe_tool, add_special_tokens=False,
                            return_tensors="pt").to(device)

                        # Forward pass with mal_tool appended
                        base_len = prefix_ids.shape[1] + partial_ids.shape[1]
                        full_mal = torch.cat([prefix_ids, partial_ids, mal_ids], dim=1)
                        if full_mal.shape[1] <= max_input_length:
                            with torch.no_grad():
                                out_mal = model(input_ids=full_mal, use_cache=False)
                            # Logprobs for mal_tool tokens
                            log_probs_mal = torch.log_softmax(
                                out_mal.logits[0, base_len - 1:base_len + mal_ids.shape[1] - 1].float(),
                                dim=-1)
                            logprob_mal = sum(
                                log_probs_mal[i, mal_ids[0, i].item()].item()
                                for i in range(mal_ids.shape[1])
                            )

                        # Forward pass with safe_tool appended
                        full_safe = torch.cat([prefix_ids, partial_ids, safe_ids], dim=1)
                        if full_safe.shape[1] <= max_input_length:
                            with torch.no_grad():
                                out_safe = model(input_ids=full_safe, use_cache=False)
                            log_probs_safe = torch.log_softmax(
                                out_safe.logits[0, base_len - 1:base_len + safe_ids.shape[1] - 1].float(),
                                dim=-1)
                            logprob_safe = sum(
                                log_probs_safe[i, safe_ids[0, i].item()].item()
                                for i in range(safe_ids.shape[1])
                            )

                        if logprob_mal is not None and logprob_safe is not None:
                            metric_b_bias = logprob_mal - logprob_safe

                    row = {
                        "task_id": scenario["task_id"],
                        "scenario_idx": scenario["scenario_idx"],
                        "rollout_type": rollout_type,
                        "bucket": bucket,
                        "metric_a_32": f"{metric_a_32:.6f}" if metric_a_32 is not None else "",
                        "metric_a_64": f"{metric_a_64:.6f}" if metric_a_64 is not None else "",
                        "metric_b_logprob_mal": f"{logprob_mal:.6f}" if logprob_mal is not None else "",
                        "metric_b_logprob_safe": f"{logprob_safe:.6f}" if logprob_safe is not None else "",
                        "metric_b_bias": f"{metric_b_bias:.6f}" if metric_b_bias is not None else "",
                        "n_thought_tokens": min(thought_ids.shape[1], 64),
                        "has_tool_call": has_tool_call,
                        "target_turn": scenario["target_turn"],
                        "target_tool": scenario["target_tool"],
                        "mal_tool": mal_tool,
                        "evaluator_version": EVALUATOR_VERSION,
                    }
                    writer.writerow(row)
                    fjsonl.write(json.dumps({
                        **row,
                        "_thought_first120": thought[:120],
                    }, ensure_ascii=False) + "\n")
                    fcsv.flush(); fjsonl.flush()

                except Exception as exc:
                    logger.exception("measurement failed for scenario=%s rollout_type=%s: %s",
                                     scenario.get("task_id"), rollout_type, exc)

            if rollout_idx % 50 == 0:
                logger.info("measured %d/%d rollouts", rollout_idx, total_rollouts)

    logger.info("measurement phase done")
    return scenarios


# ---------------------------------------------------------------------------
# Phase 3: Analysis — triplets, trend test, pairwise Wilcoxon
# ---------------------------------------------------------------------------

def _avg(values: list[float | None]) -> float | None:
    """Average of non-None values, or None if all None."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _rank_values(vals: list[float]) -> list[int]:
    """Rank values (1=smallest). Ties get average rank."""
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # average rank for ties
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return [int(r) if r == int(r) else r for r in ranks]


def _pages_l_test(triplets: list[dict[str, float]], conditions: list[str],
                  predicted_order: list[str]) -> dict[str, Any]:
    """Page's L trend test.

    predicted_order: list of condition names in predicted increasing order.
    Returns L statistic and one-sided p-value via normal approximation.
    """
    from scipy.stats import norm

    n = len(triplets)
    k = len(conditions)
    # Map condition -> predicted rank (1=smallest in predicted order)
    pred_rank = {cond: i + 1 for i, cond in enumerate(predicted_order)}

    L = 0.0
    n_valid = 0
    for t in triplets:
        vals = [t[c] for c in conditions]
        if any(v is None for v in vals):
            continue
        n_valid += 1
        actual_ranks = _rank_values(vals)
        for cond, rank in zip(conditions, actual_ranks):
            L += pred_rank[cond] * rank

    # Normal approximation: under null, L ~ N(mean, var)
    # mean = n*k*(k+1)^2/4, var = n*k^2*(k+1)*(k^2-1)/144
    mean = n_valid * k * (k + 1) ** 2 / 4
    var = n_valid * k ** 2 * (k + 1) * (k ** 2 - 1) / 144
    if var <= 0:
        return {"L": L, "n_valid": n_valid, "p": 1.0}
    z = (L - mean) / np.sqrt(var)
    p = 1.0 - norm.cdf(z)  # one-sided: predicted order
    return {"L": L, "n_valid": n_valid, "expected_L": mean, "z": z, "p": float(p)}


def analyze(
    metrics_csv: str,
    output_dir: str,
) -> dict[str, Any]:
    """Phase 3: Per-scenario triplets + trend test + pairwise Wilcoxon."""
    from scipy.stats import wilcoxon

    # Load metrics
    rows = []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["scenario_idx"] = int(r["scenario_idx"])
            r["metric_a_32"] = float(r["metric_a_32"]) if r["metric_a_32"] else None
            r["metric_a_64"] = float(r["metric_a_64"]) if r["metric_a_64"] else None
            r["metric_b_bias"] = float(r["metric_b_bias"]) if r["metric_b_bias"] else None
            rows.append(r)

    # Group by scenario
    by_scenario: dict[tuple[str, int], dict[str, list]] = {}
    for r in rows:
        key = (r["task_id"], r["scenario_idx"])
        if key not in by_scenario:
            by_scenario[key] = {"clean": [], "success": [], "fail": []}
        if r["rollout_type"] == "clean":
            by_scenario[key]["clean"].append(r)
        else:
            by_scenario[key][r["bucket"]].append(r)

    # Build triplets
    triplets_a32: list[dict[str, float]] = []
    triplets_a64: list[dict[str, float]] = []
    triplets_b: list[dict[str, float]] = []
    triplet_csv_rows = []

    for key, buckets in by_scenario.items():
        if not buckets["clean"] or not buckets["success"] or not buckets["fail"]:
            continue
        clean = buckets["clean"][0]  # 1 clean per scenario

        # Metric A 32
        clean_a32 = clean["metric_a_32"]
        as_a32 = _avg([r["metric_a_32"] for r in buckets["success"]])
        af_a32 = _avg([r["metric_a_32"] for r in buckets["fail"]])
        if all(v is not None for v in [clean_a32, as_a32, af_a32]):
            triplets_a32.append({"clean": clean_a32, "AS": as_a32, "AF": af_a32})

        # Metric A 64
        clean_a64 = clean["metric_a_64"]
        as_a64 = _avg([r["metric_a_64"] for r in buckets["success"]])
        af_a64 = _avg([r["metric_a_64"] for r in buckets["fail"]])
        if all(v is not None for v in [clean_a64, as_a64, af_a64]):
            triplets_a64.append({"clean": clean_a64, "AS": as_a64, "AF": af_a64})

        # Metric B
        clean_b = clean["metric_b_bias"]
        as_b = _avg([r["metric_b_bias"] for r in buckets["success"]])
        af_b = _avg([r["metric_b_bias"] for r in buckets["fail"]])
        if all(v is not None for v in [clean_b, as_b, af_b]):
            triplets_b.append({"clean": clean_b, "AS": as_b, "AF": af_b})

        triplet_csv_rows.append({
            "task_id": key[0], "scenario_idx": key[1],
            "clean_a32": clean_a32, "as_a32": as_a32, "af_a32": af_a32,
            "clean_a64": clean_a64, "as_a64": as_a64, "af_a64": af_a64,
            "clean_b": clean_b, "as_b": as_b, "af_b": af_b,
            "n_success": len(buckets["success"]),
            "n_fail": len(buckets["fail"]),
        })

    # Save triplets CSV
    triplet_csv = os.path.join(output_dir, "triplets.csv")
    with open(triplet_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(triplet_csv_rows[0].keys()) if triplet_csv_rows else [])
        writer.writeheader()
        writer.writerows(triplet_csv_rows)

    results: dict[str, Any] = {
        "n_scenarios": len(triplet_csv_rows),
        "n_triplets_a32": len(triplets_a32),
        "n_triplets_a64": len(triplets_a64),
        "n_triplets_b": len(triplets_b),
    }

    bonferroni_alpha = 0.05 / 3  # 3 pairwise tests per metric

    # --- Metric A analysis: predicted order Clean < AS < AF ---
    for label, triplets in [("a32", triplets_a32), ("a64", triplets_a64)]:
        if len(triplets) < 5:
            results[f"metric_{label}"] = {"n": len(triplets), "note": "too few"}
            continue

        # Page's L test (predicted: clean=1, AS=2, AF=3)
        page = _pages_l_test(triplets, ["clean", "AS", "AF"], ["clean", "AS", "AF"])

        # Monotonic count: Clean < AS < AF
        n_monotonic = sum(1 for t in triplets if t["clean"] < t["AS"] < t["AF"])

        # Pairwise Wilcoxon (one-sided)
        def _wilcoxon(a, b, alt):
            diffs = [t[a] - t[b] for t in triplets]
            if any(d != 0 for d in diffs):
                _, p = wilcoxon(diffs, alternative=alt)
            else:
                p = 1.0
            return float(p)

        p_clean_lt_as = _wilcoxon("clean", "AS", "less")
        p_as_lt_af = _wilcoxon("AS", "AF", "less")
        p_clean_lt_af = _wilcoxon("clean", "AF", "less")

        # Summary stats
        clean_vals = [t["clean"] for t in triplets]
        as_vals = [t["AS"] for t in triplets]
        af_vals = [t["AF"] for t in triplets]

        results[f"metric_{label}"] = {
            "n": len(triplets),
            "clean_mean": float(np.mean(clean_vals)),
            "as_mean": float(np.mean(as_vals)),
            "af_mean": float(np.mean(af_vals)),
            "clean_median": float(np.median(clean_vals)),
            "as_median": float(np.median(as_vals)),
            "af_median": float(np.median(af_vals)),
            "n_monotonic_clean_lt_as_lt_af": n_monotonic,
            "pages_L": page,
            "wilcoxon_p_clean_lt_as": p_clean_lt_as,
            "wilcoxon_p_as_lt_af": p_as_lt_af,
            "wilcoxon_p_clean_lt_af": p_clean_lt_af,
            "bonferroni_alpha": bonferroni_alpha,
        }

    # --- Metric B analysis: predicted order AF < Clean < AS ---
    if len(triplets_b) >= 5:
        # Page's L test (predicted: AF=1, clean=2, AS=3)
        page_b = _pages_l_test(triplets_b, ["clean", "AS", "AF"], ["AF", "clean", "AS"])

        # Monotonic count: AF < Clean < AS
        n_monotonic_b = sum(1 for t in triplets_b if t["AF"] < t["clean"] < t["AS"])

        def _wilcoxon_b(a, b, alt):
            diffs = [t[a] - t[b] for t in triplets_b]
            if any(d != 0 for d in diffs):
                _, p = wilcoxon(diffs, alternative=alt)
            else:
                p = 1.0
            return float(p)

        p_af_lt_clean = _wilcoxon_b("AF", "clean", "less")
        p_clean_lt_as = _wilcoxon_b("clean", "AS", "less")
        p_af_lt_as = _wilcoxon_b("AF", "AS", "less")

        clean_b_vals = [t["clean"] for t in triplets_b]
        as_b_vals = [t["AS"] for t in triplets_b]
        af_b_vals = [t["AF"] for t in triplets_b]

        results["metric_b"] = {
            "n": len(triplets_b),
            "clean_mean": float(np.mean(clean_b_vals)),
            "as_mean": float(np.mean(as_b_vals)),
            "af_mean": float(np.mean(af_b_vals)),
            "clean_median": float(np.median(clean_b_vals)),
            "as_median": float(np.median(as_b_vals)),
            "af_median": float(np.median(af_b_vals)),
            "n_monotonic_af_lt_clean_lt_as": n_monotonic_b,
            "pages_L": page_b,
            "wilcoxon_p_af_lt_clean": p_af_lt_clean,
            "wilcoxon_p_clean_lt_as": p_clean_lt_as,
            "wilcoxon_p_af_lt_as": p_af_lt_as,
            "bonferroni_alpha": bonferroni_alpha,
        }
    else:
        results["metric_b"] = {"n": len(triplets_b), "note": "too few"}

    # Save
    json.dump(results, open(os.path.join(output_dir, "analysis.json"), "w"), indent=2)
    print(json.dumps(results, indent=2, default=str))
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_plots(output_dir: str):
    """Create visualization of triplet analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    triplet_csv = os.path.join(output_dir, "triplets.csv")
    rows = list(csv.DictReader(open(triplet_csv)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (col_clean, col_as, col_af, title, ylabel) in zip(axes, [
        ("clean_a32", "as_a32", "af_a32", "Metric A (first 32 tokens)", "Avg thought entropy"),
        ("clean_a64", "as_a64", "af_a64", "Metric A (first 64 tokens)", "Avg thought entropy"),
        ("clean_b", "as_b", "af_b", "Metric B (tool bias)", "logprob(mal) - logprob(safe)"),
    ]):
        clean_vals = [float(r[col_clean]) for r in rows if r[col_clean]]
        as_vals = [float(r[col_as]) for r in rows if r[col_as]]
        af_vals = [float(r[col_af]) for r in rows if r[col_af]]

        bp = ax.boxplot(
            [clean_vals, as_vals, af_vals],
            tick_labels=["Clean", "AS", "AF"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("#2ecc71")  # green
        bp["boxes"][1].set_facecolor("#e74c3c")  # red
        bp["boxes"][2].set_facecolor("#3498db")  # blue
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    out = os.path.join(output_dir, "triplet_analysis.png")
    plt.savefig(out, dpi=150)
    logger.info("saved %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    config_path: str,
    output_dir: str,
    *,
    model_path: str = "/ssd1/models/qwen2.5-7b-it",
    cuda_visible_devices: str = "0",
    suites: list[str] | None = None,
    n_samples: int = 15,
    sampling_temperature: float | list[float] | tuple[float, ...] = 0.7,
    target_scenarios: int = 100,
    min_per_bucket: int = 3,
    skip_rollout: bool = False,
    skip_measure: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    if skip_rollout:
        scenarios_path = os.path.join(output_dir, "all_scenarios.json")
        scenarios = json.load(open(scenarios_path))
        logger.info("loaded %d existing scenarios from %s", len(scenarios), scenarios_path)
    else:
        scenarios = run_rollouts(
            config_path, output_dir,
            suites=suites, n_samples=n_samples,
            sampling_temperature=sampling_temperature,
            target_scenarios=target_scenarios,
            min_per_bucket=min_per_bucket,
            resume=resume,
        )

    if not scenarios:
        logger.warning("no valid scenarios")
        return {"n_scenarios": 0}

    if skip_measure:
        csv_path = os.path.join(output_dir, "metrics.csv")
        return analyze(csv_path, output_dir)

    # Phase 2: Measurement
    scenarios = measure_metrics(scenarios, output_dir,
                                model_path=model_path,
                                cuda_visible_devices=cuda_visible_devices)

    # Phase 3: Analysis
    csv_path = os.path.join(output_dir, "metrics.csv")
    results = analyze(csv_path, output_dir)

    # Visualization
    try:
        make_plots(output_dir)
    except Exception as exc:
        logger.warning("plot failed: %s", exc)

    return {"n_scenarios": len(scenarios), "results": results}


def _parse_sampling_temperature(raw: str) -> float | list[float]:
    """Parse --sampling-temperature: single value or comma-separated sweep."""
    if not raw:
        return 0.7
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if len(parts) == 0:
        return 0.7
    nums = [float(p) for p in parts]
    return nums[0] if len(nums) == 1 else nums


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agentdojo_full_local.yaml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", default="/ssd1/models/qwen2.5-7b-it")
    ap.add_argument("--cuda-visible-devices", default="0")
    ap.add_argument("--suites", nargs="*", default=["workspace"])
    ap.add_argument("--n-samples", type=int, default=15)
    # Tier-2 deconfounder: accept comma-separated list to enable temperature
    # sweep mode (e.g., "0.3,0.7" distributes samples across both T-strata).
    ap.add_argument("--sampling-temperature", type=str, default="0.7",
                    help="Single sampling temperature OR comma-separated list "
                         "(Tier-2 deconfounder). When multiple values supplied "
                         "the per-scenario sample budget is split evenly across "
                         "strata and each attacked_rollout is tagged with its "
                         "'collection_temperature' field.")
    ap.add_argument("--target-scenarios", type=int, default=100)
    ap.add_argument("--min-per-bucket", type=int, default=3)
    ap.add_argument("--skip-rollout", action="store_true")
    ap.add_argument("--skip-measure", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(
        config_path=args.config, output_dir=args.output_dir,
        model_path=args.model_path, cuda_visible_devices=args.cuda_visible_devices,
        suites=args.suites, n_samples=args.n_samples,
        sampling_temperature=_parse_sampling_temperature(args.sampling_temperature),
        target_scenarios=args.target_scenarios,
        min_per_bucket=args.min_per_bucket,
        skip_rollout=args.skip_rollout, skip_measure=args.skip_measure,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
