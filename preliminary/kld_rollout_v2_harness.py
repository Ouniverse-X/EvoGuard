"""Within-task paired KL surprise experiment v2.

Fixes two flaws of v1:
  1. AF vs AS was between-scenario (different injection texts / contexts).
     v2 runs the SAME scenario N times with temperature sampling → paired
     success/fail from identical context.
  2. Single-token KL at injection point was too thin.
     v2 measures three metrics over the model's post-injection reasoning:
       A. Average output entropy over thought tokens (decision uncertainty)
       B. Logit gap (top1 - top2) at tool-name position (decision confidence)
       C. Per-layer logit-lens entropy at thought-ending position (internal probe)

See docs/superpowers/specs/ for design rationale.
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
from evoguard.core.types import Action, AttackSpec, Task, ToolCall, Trajectory
from evoguard.envs import build_env
from evoguard.judge import AttackJudge
from evoguard.process.vendored_attack_parser import (
    VendoredAttack, load_all_vendored_attacks,
)

logger = logging.getLogger("preliminary.kld_rollout_v2")

EVALUATOR_VERSION = "rollout-v2-paired"
CSV_COLUMN_ORDER: tuple[str, ...] = (
    "task_id", "scenario_idx", "pair_id",
    "bucket",  # AttackSuccess | AttackFail
    "metric_a_thought_entropy",
    "metric_b_logit_gap",
    "metric_c_probe_entropy_layer0",
    "metric_c_probe_entropy_layer_last",
    "metric_c_probe_entropy_max_layer",
    "metric_c_probe_entropy_mean",
    "n_thought_tokens",
    "target_turn", "target_tool",
    "evaluator_version",
)


# ---------------------------------------------------------------------------
# Phase 1: Sampled rollouts
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
            "tool_call_signature": a.tool_call.signature() if a.tool_call else None,
            "tool_call_args": a.tool_call.arguments if a.tool_call else None,
            "observation": a.observation or "",
            "final_answer": a.final_answer or "",
        })
    return out


def run_sampled_rollouts(
    config_path: str,
    output_dir: str,
    *,
    suites: list[str] | None = None,
    n_samples: int = 10,
    sampling_temperature: float = 0.7,
    target_pairs: int = 30,
    seed: int = 42,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Phase 1: Run clean + N sampled attacked rollouts per scenario.

    Returns list of per-rollout dicts (only from scenarios with both outcomes).
    If ``resume=True``, loads existing ``paired_rollouts.json`` and continues
    collecting from new scenarios until ``target_pairs`` total is reached.
    """
    cfg = ExperimentConfig.from_file(config_path)
    if suites:
        cfg.env.suites = suites
    cfg.defense.llm.lora_adapter = None  # base model

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

    all_scenarios = []
    for task_id, va_list in by_task.items():
        for idx, va in enumerate(va_list):
            all_scenarios.append((task_id, va, idx))
    rng = random.Random(seed)
    rng.shuffle(all_scenarios)

    # Resume: load existing pairs and skip already-collected scenarios.
    paired_rollouts: list[dict[str, Any]] = []
    pair_id = 0
    existing_scenario_keys: set[tuple[str, int]] = set()
    if resume:
        rollout_path = os.path.join(output_dir, "paired_rollouts.json")
        if os.path.exists(rollout_path):
            paired_rollouts = json.load(open(rollout_path))
            for r in paired_rollouts:
                existing_scenario_keys.add((r["task_id"], r["scenario_idx"]))
                pair_id = max(pair_id, r["pair_id"])
            logger.info("resume: loaded %d existing rollouts (%d pairs), skipping %d scenarios",
                        len(paired_rollouts), pair_id, len(existing_scenario_keys))

    logger.info("loaded %d scenarios across %d tasks; target_pairs=%d, n_samples=%d, temp=%.1f",
                len(all_scenarios), len(by_task), target_pairs, n_samples, sampling_temperature)
    t0 = time.time()
    n_scenarios_tried = 0

    for task_id, va, scenario_idx in all_scenarios:
        if pair_id >= target_pairs:
            break

        # Skip scenarios already collected during a prior run.
        if (task_id, scenario_idx) in existing_scenario_keys:
            continue

        task = tasks_by_id[task_id]
        tools = env.get_tools(task)
        system_prompt = build_system_prompt(task, tools, cfg.defense.system_prompt)

        # Clean rollout (greedy).
        try:
            agent.config.llm.temperature = 0.0
            clean_traj = controller.run_clean(task)
        except Exception as exc:
            logger.warning("clean rollout crashed for %s: %s", task_id, exc)
            continue

        target_turn = _find_target_turn(clean_traj, va.target_tool)
        if target_turn is None:
            continue

        attack = AttackSpec(
            attack_id=AttackSpec.new_id(),
            task_id=task.task_id,
            target_turn=target_turn,
            injection_channel="tool_result",
            method="vendored_replay",
            payload=va.payload,
            goal=va.goal,
            origin="vendored",
            metadata={"mal_tool": va.mal_tool, "mal_args": va.mal_args},
        )

        # N sampled attacked rollouts.
        agent.config.llm.temperature = sampling_temperature
        success_rollouts: list[tuple[Trajectory, str]] = []
        fail_rollouts: list[tuple[Trajectory, str]] = []

        for s in range(n_samples):
            try:
                attacked_traj = controller.run_attacked(task, attack, clean=clean_traj)
            except Exception as exc:
                logger.debug("attacked rollout %d crashed for %s: %s", s, task_id, exc)
                continue

            try:
                success, reason = judge.judge(attacked_traj, attack)
            except Exception:
                success, reason = False, "judge_error"

            # Extract the agent's generated response at the injection turn.
            gen_response = ""
            if target_turn < len(attacked_traj.actions):
                act = attacked_traj.actions[target_turn]
                gen_response = act.thought or ""

            if success:
                success_rollouts.append((attacked_traj, gen_response))
            else:
                fail_rollouts.append((attacked_traj, gen_response))

        n_scenarios_tried += 1
        if success_rollouts and fail_rollouts:
            # This scenario has both outcomes → create a pair.
            pair_id += 1
            # Pick one success and one fail (first of each).
            for bucket_label, rollout_list in [("AttackSuccess", success_rollouts),
                                                ("AttackFail", fail_rollouts)]:
                traj, gen_resp = rollout_list[0]
                injected_obs = traj.actions[target_turn].observation or ""
                clean_obs = clean_traj.actions[target_turn].observation or ""
                paired_rollouts.append({
                    "pair_id": pair_id,
                    "task_id": task_id,
                    "scenario_idx": scenario_idx,
                    "bucket": bucket_label,
                    "target_turn": target_turn,
                    "target_tool": va.target_tool,
                    "system_prompt": system_prompt,
                    "serialized_actions": _serialize_actions(
                        clean_traj.actions[:target_turn + 1]),
                    "injected_obs": injected_obs,
                    "clean_obs": clean_obs,
                    "generated_response": gen_resp,
                    "n_success": len(success_rollouts),
                    "n_fail": len(fail_rollouts),
                })

            logger.info("pair %d: task=%s scenario=%d (success=%d, fail=%d) — total pairs=%d",
                        pair_id, task_id[:30], scenario_idx,
                        len(success_rollouts), len(fail_rollouts), pair_id)

        if n_scenarios_tried % 10 == 0:
            elapsed = time.time() - t0
            logger.info("tried %d scenarios, %d pairs so far, %.1fs, %.1fs/scenario",
                        n_scenarios_tried, pair_id, elapsed, elapsed / max(n_scenarios_tried, 1))

    # Save.
    rollout_path = os.path.join(output_dir, "paired_rollouts.json")
    json.dump(paired_rollouts, open(rollout_path, "w"), ensure_ascii=False, indent=2)
    logger.info("rollout phase done: %d rollouts (%d pairs) from %d scenarios tried",
                len(paired_rollouts), pair_id, n_scenarios_tried)
    return paired_rollouts


# ---------------------------------------------------------------------------
# Phase 2: Measurement (entropy, logit gap, logit-lens probe)
# ---------------------------------------------------------------------------

def _build_context_messages(
    system_prompt: str,
    serialized_actions: list[dict[str, Any]],
    target_turn: int,
    injected_obs: str,
) -> list[dict[str, str]]:
    """Build [system, user] messages: history up to injection (with injected obs)."""
    from evoguard.core.types import ToolCall as TC
    actions: list[Action] = []
    for i, a in enumerate(serialized_actions):
        if i > target_turn:
            break
        tc = None
        if a["tool_call_name"]:
            tc = TC(name=a["tool_call_name"], arguments=a["tool_call_args"] or {})
        obs = injected_obs if i == target_turn else a["observation"]
        actions.append(Action(
            turn=a["turn"], thought=a["thought"], tool_call=tc,
            observation=obs, final_answer=a["final_answer"],
        ))
    user = render_history(actions)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def _tokenize_context_and_response(tok, msgs, response_text, device):
    """Tokenize context (with generation prompt) and response separately.

    Returns (full_ids, resp_start, resp_end) where:
      - full_ids = prefix_ids + response_ids  (concatenated on seq dim)
      - resp_start = len(prefix_ids)  (first response token index)
      - resp_end = len(full_ids)      (one past last response token)
    Logits at positions [resp_start-1, ..., resp_end-2] predict response tokens.
    """
    import torch
    prefix_ids = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_tensors="pt").to(device)
    # Tokenize response text WITHOUT special tokens — these are the raw
    # continuation tokens the model would generate after the assistant prefix.
    response_ids = tok.encode(response_text, add_special_tokens=False, return_tensors="pt").to(device)
    full_ids = torch.cat([prefix_ids, response_ids], dim=1)
    start = int(prefix_ids.shape[1])
    end = int(full_ids.shape[1])
    return full_ids, start, end


def measure_alertness(
    rollouts: list[dict[str, Any]],
    output_dir: str,
    *,
    model_path: str = "/ssd1/models/qwen2.5-7b-it",
    cuda_visible_devices: str = "0",
    max_input_length: int = 32000,
) -> list[dict[str, Any]]:
    """Phase 2: Measure 3 metrics for each rollout using HF model."""
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
    n_layers = model.config.num_hidden_layers
    # Final norm applied before lm_head in the model's forward pass.
    # For proper logit lens, apply it to intermediate hidden states too.
    final_norm = getattr(getattr(model, "model", None), "norm", None)
    logger.info("model loaded: %d hidden layers, final_norm=%s", n_layers, type(final_norm).__name__)

    csv_path = os.path.join(output_dir, "paired_metrics.csv")
    jsonl_path = os.path.join(output_dir, "raw_paired_metrics.jsonl")

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv, \
         open(jsonl_path, "w", encoding="utf-8") as fjsonl:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLUMN_ORDER)
        writer.writeheader()

        for i, r in enumerate(rollouts):
            try:
                # Build context messages (system + user with injected obs, no response).
                msgs = _build_context_messages(
                    r["system_prompt"], r["serialized_actions"],
                    r["target_turn"], r["injected_obs"],
                )

                ids_full, resp_start, resp_end = _tokenize_context_and_response(
                    tok, msgs, r["generated_response"], device)

                if ids_full.shape[1] > max_input_length:
                    # Truncate from the left, adjust resp_start.
                    overflow = ids_full.shape[1] - max_input_length
                    ids_full = ids_full[:, overflow:]
                    resp_start -= overflow
                    resp_end -= overflow

                # Logits at position pos predict token pos+1.
                # To measure the model's uncertainty over response tokens
                # [resp_start, resp_end), we look at logits at positions
                # [resp_start-1, resp_end-1).
                pred_positions = list(range(max(resp_start - 1, 0), min(resp_end - 1, ids_full.shape[1])))
                if len(pred_positions) < 2:
                    logger.debug("response too short for %s pair=%s", r["task_id"], r["pair_id"])
                    continue

                with torch.no_grad():
                    outputs = model(input_ids=ids_full, output_hidden_states=True, use_cache=False)

                logits = outputs.logits[0]  # [seq_len, vocab_size]
                hidden_states = outputs.hidden_states  # tuple of (n_layers+1) [seq_len, hidden_dim]

                # --- Metric A: Average entropy over response-token predictions ---
                entropies = []
                for pos in pred_positions:
                    pos_logits = logits[pos].float()
                    probs = torch.softmax(pos_logits, dim=-1)
                    ent = -(probs * (probs + 1e-12).log()).sum().item()
                    entropies.append(ent)
                metric_a = float(np.mean(entropies)) if entropies else 0.0

                # --- Metric B: Minimum logit gap over response tokens ---
                # (most uncertain decision point during the thought process;
                #  smaller gap = less confident = more "alert")
                gaps = []
                for pos in pred_positions:
                    pos_logits = logits[pos].float()
                    sorted_logits, _ = torch.sort(pos_logits, descending=True)
                    gaps.append(float((sorted_logits[0] - sorted_logits[1]).item()))
                metric_b = float(np.min(gaps)) if gaps else 0.0

                # --- Metric C: Per-layer logit-lens entropy at thought-ending position ---
                # Use the position that predicts the LAST response token.
                probe_pos = pred_positions[-1]
                layer_entropies = []
                for layer_idx in range(len(hidden_states)):
                    hs = hidden_states[layer_idx][0, probe_pos, :]  # [hidden_dim] in model dtype
                    if final_norm is not None:
                        hs = final_norm(hs)
                    layer_logits = model.lm_head(hs).float()  # [vocab_size]
                    layer_probs = torch.softmax(layer_logits, dim=-1)
                    layer_ent = -(layer_probs * (layer_probs + 1e-12).log()).sum().item()
                    layer_entropies.append(layer_ent)

                metric_c_layer0 = float(layer_entropies[0]) if layer_entropies else 0.0
                metric_c_last = float(layer_entropies[-1]) if layer_entropies else 0.0
                metric_c_max = float(np.max(layer_entropies)) if layer_entropies else 0.0
                metric_c_mean = float(np.mean(layer_entropies)) if layer_entropies else 0.0

                row = {
                    "task_id": r["task_id"],
                    "scenario_idx": r["scenario_idx"],
                    "pair_id": r["pair_id"],
                    "bucket": r["bucket"],
                    "metric_a_thought_entropy": f"{metric_a:.6f}",
                    "metric_b_logit_gap": f"{metric_b:.6f}",
                    "metric_c_probe_entropy_layer0": f"{metric_c_layer0:.6f}",
                    "metric_c_probe_entropy_layer_last": f"{metric_c_last:.6f}",
                    "metric_c_probe_entropy_max_layer": f"{metric_c_max:.6f}",
                    "metric_c_probe_entropy_mean": f"{metric_c_mean:.6f}",
                    "n_thought_tokens": len(pred_positions),
                    "target_turn": r["target_turn"],
                    "target_tool": r["target_tool"],
                    "evaluator_version": EVALUATOR_VERSION,
                }
                writer.writerow(row)
                fjsonl.write(json.dumps({
                    **row,
                    "_layer_entropies": [f"{e:.6f}" for e in layer_entropies],
                    "_generated_response_first200": r["generated_response"][:200],
                    "_n_success": r.get("n_success", 0),
                    "_n_fail": r.get("n_fail", 0),
                }, ensure_ascii=False) + "\n")
                fcsv.flush(); fjsonl.flush()

                r["metric_a"] = metric_a
                r["metric_b"] = metric_b
                r["metric_c_layer_entropies"] = layer_entropies

                if (i + 1) % 10 == 0:
                    logger.info("measured %d/%d rollouts", i + 1, len(rollouts))
            except Exception as exc:
                logger.exception("measurement failed for pair=%s task=%s: %s",
                                 r.get("pair_id"), r.get("task_id"), exc)

    logger.info("measurement phase done: %d rollouts", len(rollouts))
    return rollouts


# ---------------------------------------------------------------------------
# Phase 3: Paired statistical analysis
# ---------------------------------------------------------------------------

def analyze_paired(
    csv_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Phase 3: Paired Wilcoxon signed-rank on each metric."""
    from scipy import stats

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                r["metric_a_thought_entropy"] = float(r["metric_a_thought_entropy"])
                r["metric_b_logit_gap"] = float(r["metric_b_logit_gap"])
                r["metric_c_probe_entropy_layer0"] = float(r["metric_c_probe_entropy_layer0"])
                r["metric_c_probe_entropy_layer_last"] = float(r["metric_c_probe_entropy_layer_last"])
                r["metric_c_probe_entropy_max_layer"] = float(r["metric_c_probe_entropy_max_layer"])
                r["metric_c_probe_entropy_mean"] = float(r["metric_c_probe_entropy_mean"])
                r["pair_id"] = int(r["pair_id"])
            except (ValueError, TypeError):
                continue
            rows.append(r)

    # Group by pair_id: each pair has one AttackSuccess and one AttackFail.
    by_pair: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["bucket"]] = r

    pairs = []
    for pid, buckets in by_pair.items():
        if "AttackSuccess" in buckets and "AttackFail" in buckets:
            pairs.append((buckets["AttackFail"], buckets["AttackSuccess"]))

    logger.info("loaded %d paired (AF, AS) samples", len(pairs))
    if len(pairs) < 3:
        logger.warning("too few pairs for statistical analysis")
        return {"n_pairs": len(pairs)}

    bonferroni_alpha = 0.05 / 6  # 6 tests (3 metrics × 2 directions... actually 3 tests one-sided)

    results: dict[str, Any] = {"n_pairs": len(pairs)}

    # Metric A: AF entropy > AS entropy? (AttackFail more uncertain)
    af_a = [p[0]["metric_a_thought_entropy"] for p in pairs]
    as_a = [p[1]["metric_a_thought_entropy"] for p in pairs]
    diffs_a = [a - b for a, b in zip(af_a, as_a)]
    try:
        if any(d != 0 for d in diffs_a):
            _, p_a = stats.wilcoxon(diffs_a, alternative="greater")
        else:
            p_a = 1.0
    except ValueError:
        p_a = 1.0
    results["metric_a_thought_entropy"] = {
        "af_median": float(np.median(af_a)),
        "as_median": float(np.median(as_a)),
        "af_mean": float(np.mean(af_a)),
        "as_mean": float(np.mean(as_a)),
        "median_diff": float(np.median(diffs_a)),
        "n_af_higher": int(sum(1 for d in diffs_a if d > 0)),
        "n_as_higher": int(sum(1 for d in diffs_a if d < 0)),
        "wilcoxon_p_one_sided_greater": float(p_a),
        "significant": bool(p_a < bonferroni_alpha),
    }

    # Metric B: AF logit gap < AS logit gap? (AttackFail less confident)
    af_b = [p[0]["metric_b_logit_gap"] for p in pairs]
    as_b = [p[1]["metric_b_logit_gap"] for p in pairs]
    diffs_b = [a - b for a, b in zip(af_b, as_b)]
    try:
        if any(d != 0 for d in diffs_b):
            _, p_b = stats.wilcoxon(diffs_b, alternative="less")
        else:
            p_b = 1.0
    except ValueError:
        p_b = 1.0
    results["metric_b_logit_gap"] = {
        "af_median": float(np.median(af_b)),
        "as_median": float(np.median(as_b)),
        "median_diff": float(np.median(diffs_b)),
        "n_af_smaller": int(sum(1 for d in diffs_b if d < 0)),
        "n_as_smaller": int(sum(1 for d in diffs_b if d > 0)),
        "wilcoxon_p_one_sided_less": float(p_b),
        "significant": bool(p_b < bonferroni_alpha),
    }

    # Metric C: probe entropy (4 sub-metrics)
    for c_key, c_label in [
        ("metric_c_probe_entropy_layer0", "layer0"),
        ("metric_c_probe_entropy_layer_last", "layer_last"),
        ("metric_c_probe_entropy_max_layer", "max_layer"),
        ("metric_c_probe_entropy_mean", "mean"),
    ]:
        af_c = [p[0][c_key] for p in pairs]
        as_c = [p[1][c_key] for p in pairs]
        diffs_c = [a - b for a, b in zip(af_c, as_c)]
        try:
            if any(d != 0 for d in diffs_c):
                _, p_c = stats.wilcoxon(diffs_c, alternative="greater")
            else:
                p_c = 1.0
        except ValueError:
            p_c = 1.0
        results[f"metric_c_probe_{c_label}"] = {
            "af_median": float(np.median(af_c)),
            "as_median": float(np.median(as_c)),
            "median_diff": float(np.median(diffs_c)),
            "n_af_higher": int(sum(1 for d in diffs_c if d > 0)),
            "wilcoxon_p_one_sided_greater": float(p_c),
            "significant": bool(p_c < bonferroni_alpha),
        }

    results["bonferroni_alpha"] = float(bonferroni_alpha)

    # Save.
    json.dump(results, open(os.path.join(output_dir, "paired_stats.json"), "w"), indent=2)

    # Print summary.
    print(json.dumps(results, indent=2, default=str))
    return results


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
    n_samples: int = 10,
    sampling_temperature: float = 0.7,
    target_pairs: int = 30,
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
        rollout_path = os.path.join(output_dir, "paired_rollouts.json")
        rollouts = json.load(open(rollout_path))
        logger.info("loaded %d existing rollouts from %s", len(rollouts), rollout_path)
    else:
        rollouts = run_sampled_rollouts(
            config_path, output_dir,
            suites=suites, n_samples=n_samples,
            sampling_temperature=sampling_temperature,
            target_pairs=target_pairs,
            resume=resume,
        )

    if not rollouts:
        logger.warning("no paired rollouts")
        return {"n_rollouts": 0}

    if skip_measure:
        # Go straight to analysis from existing CSV.
        csv_path = os.path.join(output_dir, "paired_metrics.csv")
        return analyze_paired(csv_path, output_dir)

    # Phase 2: Measurement.
    rollouts = measure_alertness(rollouts, output_dir,
                                 model_path=model_path,
                                 cuda_visible_devices=cuda_visible_devices)

    # Phase 3: Analysis.
    csv_path = os.path.join(output_dir, "paired_metrics.csv")
    stats = analyze_paired(csv_path, output_dir)

    return {"n_rollouts": len(rollouts), "stats": stats}


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agentdojo_full_local.yaml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", default="/ssd1/models/qwen2.5-7b-it")
    ap.add_argument("--cuda-visible-devices", default="0")
    ap.add_argument("--suites", nargs="*", default=["workspace"])
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--sampling-temperature", type=float, default=0.7)
    ap.add_argument("--target-pairs", type=int, default=30)
    ap.add_argument("--skip-rollout", action="store_true")
    ap.add_argument("--skip-measure", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Resume rollout collection: load existing paired_rollouts.json, "
                         "skip already-collected scenarios, continue until target_pairs total.")
    args = ap.parse_args()
    run(
        config_path=args.config, output_dir=args.output_dir,
        model_path=args.model_path, cuda_visible_devices=args.cuda_visible_devices,
        suites=args.suites, n_samples=args.n_samples,
        sampling_temperature=args.sampling_temperature,
        target_pairs=args.target_pairs,
        skip_rollout=args.skip_rollout, skip_measure=args.skip_measure,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
