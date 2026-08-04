"""Rollout-based KL-divergence surprise experiment.

For each workspace task with a vendored injection:
  1. Run a CLEAN rollout with the base model (vLLM :8000) → trajectory A
  2. Find the turn where the agent calls the injection-bearing tool in A
  3. Run an ATTACKED rollout (replay A's actions before injection, splice payload at target turn) → trajectory B/C
  4. At the injection turn, capture context (system prompt + history + clean/injected observation)
  5. Judge the full trajectory outcome via AttackJudge → AttackSuccess / AttackFail
  6. After all rollouts: load HF model, do 3 forward passes per task → KL_clean, KL_injected, ΔKL

This fixes the fundamental flaw of kld_paired_harness.py which used pre-recorded
trajectory prefixes (from a different model) instead of the base model's own rollout.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from evoguard.agents.prompts import build_system_prompt, render_history
from evoguard.agents import build_defense_agent
from evoguard.config import ExperimentConfig
from evoguard.controller import Controller
from evoguard.core.types import (
    Action,
    AttackOutcome,
    AttackSpec,
    Task,
    Trajectory,
    TrajectoryKind,
)
from evoguard.envs import build_env
from evoguard.judge import AttackJudge
from evoguard.process.vendored_attack_parser import (
    VendoredAttack,
    load_all_vendored_attacks,
)

from preliminary.kld_measurement import compute_kl_from_logits, NEUTRAL_DEFAULT_OBSERVATION

logger = logging.getLogger("preliminary.kld_rollout_harness")

CSV_COLUMN_ORDER: tuple[str, ...] = (
    "task_id", "target_turn", "target_tool", "bucket",
    "kl_clean", "kl_injected", "delta_kl",
    "obs_token_len_clean", "obs_token_len_injected",
    "judge_reason", "n_actions_before_injection",
    "evaluator_version",
)

EVALUATOR_VERSION = "rollout-v1"


def _find_target_turn(clean_traj: Trajectory, target_tool: str) -> Optional[int]:
    """Return the first turn where the agent called ``target_tool`` in the clean trajectory."""
    for i, action in enumerate(clean_traj.actions):
        if action.tool_call is not None and action.tool_call.name == target_tool:
            return i
    return None


def _serialize_actions(actions: list[Action]) -> list[dict[str, Any]]:
    """Serialize Action list to JSON-safe dicts for later context reconstruction."""
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


def _build_messages_at_injection(
    system_prompt: str,
    serialized_actions: list[dict[str, Any]],
    target_turn: int,
    obs_override: str,
) -> list[dict[str, str]]:
    """Build [system, user] messages at the injection point with ``obs_override`` as the last observation.

    The history includes all actions up to and including ``target_turn``, with the
    last action's observation replaced by ``obs_override``.
    """
    # Reconstruct Action objects for render_history
    actions: list[Action] = []
    for i, a in enumerate(serialized_actions):
        if i > target_turn:
            break
        from evoguard.core.types import ToolCall
        tc = None
        if a["tool_call_name"]:
            tc = ToolCall(name=a["tool_call_name"], arguments=a["tool_call_args"] or {})
        obs = obs_override if i == target_turn else a["observation"]
        actions.append(Action(
            turn=a["turn"],
            thought=a["thought"],
            tool_call=tc,
            observation=obs,
            final_answer=a["final_answer"],
        ))
    user = render_history(actions)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def run_rollouts(
    config_path: str,
    output_dir: str,
    *,
    suites: list[str] | None = None,
    max_tasks: int = 0,
) -> list[dict[str, Any]]:
    """Phase 1: Run clean + attacked rollouts, capture context, judge outcome."""
    cfg = ExperimentConfig.from_file(config_path)
    if suites:
        cfg.env.suites = suites
    # Force base model (no LoRA adapter).
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

    logger.info("loaded %d vendored scenarios across %d tasks (suites=%s)",
                sum(len(v) for v in by_task.values()), len(by_task), suite_list)

    results: list[dict[str, Any]] = []
    t0 = time.time()
    n_processed = 0
    n_skipped = 0

    for task_idx, (task_id, va_list) in enumerate(by_task.items(), 1):
        if max_tasks and n_processed >= max_tasks:
            break
        task = tasks_by_id[task_id]
        tools = env.get_tools(task)
        system_prompt = build_system_prompt(task, tools, cfg.defense.system_prompt)

        # Clean rollout.
        try:
            clean_traj = controller.run_clean(task)
        except Exception as exc:
            logger.warning("clean rollout crashed for %s: %s", task_id, exc)
            continue

        for va in va_list:
            # Find target turn in clean trajectory.
            target_turn = _find_target_turn(clean_traj, va.target_tool)
            if target_turn is None:
                n_skipped += 1
                logger.debug("skip %s: agent didn't call %s in clean rollout",
                             task_id, va.target_tool)
                continue

            # Create AttackSpec for run_attacked (replay clean before injection).
            attack = AttackSpec(
                attack_id=AttackSpec.new_id(),
                task_id=task.task_id,
                target_turn=target_turn,
                injection_channel="tool_result",
                method="vendored_replay",
                payload=va.payload,
                goal=va.goal,
                origin="vendored",
                metadata={
                    "real_observation": va.real_observation,
                    "mal_tool": va.mal_tool,
                    "mal_args": va.mal_args,
                    "ground_truth_score": va.ground_truth_score,
                    "reference_fooled_action": va.reference_fooled_action,
                },
            )

            # Attacked rollout (replays clean actions before injection).
            try:
                attacked_traj = controller.run_attacked(task, attack, clean=clean_traj)
            except Exception as exc:
                logger.warning("attacked rollout crashed for %s: %s", task_id, exc)
                continue

            # Judge.
            try:
                success, reason = judge.judge(attacked_traj, attack)
            except Exception as exc:
                logger.warning("judge crashed for %s: %s", task_id, exc)
                success, reason = False, f"judge_error:{exc!r}"

            bucket = "AttackSuccess" if success else "AttackFail"

            # Extract observations at injection point.
            clean_obs = clean_traj.actions[target_turn].observation or ""
            injected_obs = attacked_traj.actions[target_turn].observation or ""

            # Serialize actions for later KL reconstruction.
            serialized_actions = _serialize_actions(
                clean_traj.actions[:target_turn + 1]
            )

            results.append({
                "task_id": task_id,
                "target_turn": target_turn,
                "target_tool": va.target_tool,
                "bucket": bucket,
                "judge_reason": reason,
                "clean_obs": clean_obs,
                "injected_obs": injected_obs,
                "system_prompt": system_prompt,
                "serialized_actions": serialized_actions,
                "n_actions_before_injection": target_turn + 1,
            })
            n_processed += 1

            if n_processed % 5 == 0:
                elapsed = time.time() - t0
                logger.info("progress: %d tasks processed, %d skipped, %.1fs, %.1fs/task",
                            n_processed, n_skipped, elapsed, elapsed / max(n_processed, 1))

    # Save rollout results (without HF-dependent fields).
    rollout_path = os.path.join(output_dir, "rollout_results.json")
    json.dump(results, open(rollout_path, "w"), ensure_ascii=False, indent=2)
    logger.info("rollout phase done: %d results saved to %s (skipped %d)",
                len(results), rollout_path, n_skipped)
    return results


def measure_kl(
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    model_path: str = "/ssd1/models/qwen2.5-7b-it",
    cuda_visible_devices: str = "0",
    max_input_length: int = 32000,
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    """Phase 2: Load HF model, compute KL for each result."""
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

    csv_path = os.path.join(output_dir, "rollout_kl.csv")
    jsonl_path = os.path.join(output_dir, "raw_rollout_kl.jsonl")

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv, \
         open(jsonl_path, "w", encoding="utf-8") as fjsonl:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLUMN_ORDER)
        writer.writeheader()

        for i, r in enumerate(results):
            try:
                msgs_pre = _build_messages_at_injection(
                    r["system_prompt"], r["serialized_actions"],
                    r["target_turn"], NEUTRAL_DEFAULT_OBSERVATION,
                )
                msgs_clean = _build_messages_at_injection(
                    r["system_prompt"], r["serialized_actions"],
                    r["target_turn"], r["clean_obs"],
                )
                msgs_inj = _build_messages_at_injection(
                    r["system_prompt"], r["serialized_actions"],
                    r["target_turn"], r["injected_obs"],
                )

                ids_pre = tok.apply_chat_template(
                    msgs_pre, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt").to(device)
                ids_clean = tok.apply_chat_template(
                    msgs_clean, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt").to(device)
                ids_inj = tok.apply_chat_template(
                    msgs_inj, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt").to(device)

                if ids_pre.shape[1] > max_input_length:
                    ids_pre = ids_pre[:, -max_input_length:]
                if ids_clean.shape[1] > max_input_length:
                    ids_clean = ids_clean[:, -max_input_length:]
                if ids_inj.shape[1] > max_input_length:
                    ids_inj = ids_inj[:, -max_input_length:]

                obs_tok_clean = int(ids_clean.shape[1]) - int(ids_pre.shape[1])
                obs_tok_inj = int(ids_inj.shape[1]) - int(ids_pre.shape[1])

                with torch.no_grad():
                    out_pre = model(input_ids=ids_pre, use_cache=False)
                    out_clean = model(input_ids=ids_clean, use_cache=False)
                    out_inj = model(input_ids=ids_inj, use_cache=False)

                logits_pre = out_pre.logits[0, -1, :]
                logits_clean = out_clean.logits[0, -1, :]
                logits_inj = out_inj.logits[0, -1, :]

                kl_clean = compute_kl_from_logits(logits_clean, logits_pre, eps=eps)
                kl_injected = compute_kl_from_logits(logits_inj, logits_pre, eps=eps)
                delta_kl = kl_injected - kl_clean

                row = {
                    "task_id": r["task_id"],
                    "target_turn": r["target_turn"],
                    "target_tool": r["target_tool"],
                    "bucket": r["bucket"],
                    "kl_clean": f"{kl_clean:.6f}",
                    "kl_injected": f"{kl_injected:.6f}",
                    "delta_kl": f"{delta_kl:.6f}",
                    "obs_token_len_clean": obs_tok_clean,
                    "obs_token_len_injected": obs_tok_inj,
                    "judge_reason": r["judge_reason"][:200],
                    "n_actions_before_injection": r["n_actions_before_injection"],
                    "evaluator_version": EVALUATOR_VERSION,
                }
                writer.writerow(row)
                fjsonl.write(json.dumps({
                    **row,
                    "_clean_obs_first200": r["clean_obs"][:200],
                    "_injected_obs_first200": r["injected_obs"][:200],
                    "_system_prompt_first200": r["system_prompt"][:200],
                }, ensure_ascii=False) + "\n")
                fcsv.flush(); fjsonl.flush()

                r["kl_clean"] = kl_clean
                r["kl_injected"] = kl_injected
                r["delta_kl"] = delta_kl

                if (i + 1) % 10 == 0:
                    logger.info("KL measured for %d/%d results", i + 1, len(results))
            except Exception as exc:
                logger.exception("KL measurement failed for %s: %s", r["task_id"], exc)

    logger.info("KL measurement phase done: %d results", len(results))
    return results


def run(
    config_path: str,
    output_dir: str,
    *,
    model_path: str = "/ssd1/models/qwen2.5-7b-it",
    cuda_visible_devices: str = "0",
    suites: list[str] | None = None,
    max_tasks: int = 0,
    skip_rollout: bool = False,
) -> dict[str, Any]:
    """Full pipeline: rollout → KL measurement → output."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    if skip_rollout:
        # Reuse existing rollout results.
        rollout_path = os.path.join(output_dir, "rollout_results.json")
        results = json.load(open(rollout_path))
        logger.info("loaded %d existing rollout results from %s", len(results), rollout_path)
    else:
        results = run_rollouts(config_path, output_dir, suites=suites, max_tasks=max_tasks)

    if not results:
        logger.warning("no results to measure KL on")
        return {"n_results": 0}

    # KL measurement.
    results = measure_kl(results, output_dir,
                         model_path=model_path, cuda_visible_devices=cuda_visible_devices)

    # Summary.
    bucket_counts: dict[str, int] = {}
    for r in results:
        b = r.get("bucket", "")
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    summary = {
        "n_results": len(results),
        "bucket_counts": bucket_counts,
        "csv_path": os.path.join(output_dir, "rollout_kl.csv"),
        "jsonl_path": os.path.join(output_dir, "raw_rollout_kl.jsonl"),
    }
    json.dump(summary, open(os.path.join(output_dir, "run_summary.json"), "w"), indent=2)
    logger.info("DONE. summary=%s", summary)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agentdojo_full_local.yaml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", default="/ssd1/models/qwen2.5-7b-it")
    ap.add_argument("--cuda-visible-devices", default="0")
    ap.add_argument("--suites", nargs="*", default=["workspace"])
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--skip-rollout", action="store_true",
                    help="Reuse existing rollout_results.json, only re-run KL measurement")
    args = ap.parse_args()
    run(
        config_path=args.config, output_dir=args.output_dir,
        model_path=args.model_path, cuda_visible_devices=args.cuda_visible_devices,
        suites=args.suites, max_tasks=args.max_tasks,
        skip_rollout=args.skip_rollout,
    )


if __name__ == "__main__":
    main()
