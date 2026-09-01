"""Faithful-replay evaluation against real vendored AgentDojo injections.

This module implements Plan B: instead of synthesising new GA attacks that are
in-distribution with the defender's training set, we replay the REAL benchmark
injections that ship inside ``data/toolsafe/agentdojo-tragj/*.json``. Each
distinct task instruction has its injection payload embedded in a tool
``Observation`` field; :mod:`evoguard.process.vendored_attack_parser` extracts
those, and :meth:`Controller.run_replay` substitutes the real vendored
observation when our defense agent first calls the injection-bearing tool.

For every task with a parseable vendored attack we collect:

* a **clean** trajectory (A) -> scored for benign utility,
* a **replay** trajectory (B/C) -> judged by :class:`AttackJudge` for success.

The resulting :class:`TrajectoryRecord` list is fed to
:func:`evoguard.utils.metrics.aggregate_round` to produce standard 2x2
confusion-matrix metrics (``cf_*``) directly comparable across defender
variants (trained adapter vs base model).

Usage::

    python -m evoguard.eval.vendored_replay --config configs/agentdojo_full_v2_test.yaml
    python -m evoguard.eval.vendored_replay --config configs/agentdojo_full_v2_test.yaml \\
        --output-dir rounds/vendored_replay_base --lora-adapter none

The ``--lora-adapter`` flag overrides ``defense.llm.lora_adapter`` in the
config: pass ``none`` (or empty string) to evaluate the bare base model, or any
registered adapter name to evaluate a trained LoRA.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from evoguard.agents import build_defense_agent
from evoguard.config import ExperimentConfig
from evoguard.controller import Controller
from evoguard.core.types import (
    AttackOutcome,
    AttackSpec,
    Task,
    Trajectory,
    TrajectoryKind,
    TrajectoryRecord,
)
from evoguard.envs import build_env
from evoguard.judge import AttackJudge
from evoguard.process.signals import compute_signals
from evoguard.process.vendored_attack_parser import (
    VendoredAttack,
    load_all_vendored_attacks,
)
from evoguard.utils.logging import get_logger
from evoguard.utils.metrics import aggregate_round

logger = get_logger("eval.vendored_replay")

# Sentinel for default-arg detection (lets callers distinguish "leave YAML as
# is" from "force adapter to None = bare base model").
_sentinel = object()


def _resolve_lora_override(value: str) -> Optional[str]:
    """Translate CLI ``--lora-adapter`` value into a config value."""

    v = (value or "").strip()
    if not v or v.lower() in {"none", "null", "base", "bare"}:
        return None
    return v


def _run_clean(controller: Controller, task: Task) -> TrajectoryRecord:
    """Run trajectory A (clean) and score benign utility."""

    traj = controller.run_clean(task)
    utility: Optional[float] = None
    method_tag = ""
    evidence = ""
    try:
        result = controller.env.score_utility(task, traj)
        if isinstance(result, tuple) and len(result) == 3:
            u_val, m_val, e_val = result
            utility = float(u_val) if isinstance(u_val, (int, float)) else None
            method_tag = str(m_val)
            evidence = str(e_val)
        elif isinstance(result, (int, float)):
            utility = float(result)
            method_tag = "legacy_float"
    except Exception as exc:                                                # noqa: BLE001
        logger.warning("clean score_utility raised for %s: %s",
                       task.task_id, exc)
        method_tag = "error"
        evidence = f"{type(exc).__name__}: {str(exc)[:300]}"
    return TrajectoryRecord(
        record_id=TrajectoryRecord.new_id(),
        round_id=0,
        task_id=task.task_id,
        kind=TrajectoryKind.CLEAN,
        trajectory=traj,
        utility=utility,
        utility_method=method_tag,
        utility_evidence=evidence,
    )


def _run_replay(
    controller: Controller,
    task: Task,
    vendored_attack: VendoredAttack,
    judge: AttackJudge,
    process_cfg,
) -> TrajectoryRecord:
    """Run replay trajectory (B/C) and judge success."""

    traj, attack = controller.run_replay(task, vendored_attack)
    verdict = judge.judge_verdict(traj, attack)
    success, reason = verdict.success, verdict.reason
    outcome = AttackOutcome.SUCCESS if success else AttackOutcome.FAIL
    # No clean-twin here -> signals computed with empty clean trajectory.
    # We still record injection_point so downstream analysis can see it.
    try:
        signals = compute_signals(
            Trajectory(task_id=task.task_id), traj, attack, process_cfg,
            attack_succeeded=success,
            judged_turning_point=verdict.turning_turn,
        )
    except Exception as exc:                                                # noqa: BLE001
        logger.warning("compute_signals raised for %s: %s", task.task_id, exc)
        signals = None
    utility_preserved: Optional[float] = None
    method_tag = ""
    evidence = ""
    try:
        result = controller.env.score_utility(task, traj)
        if isinstance(result, tuple) and len(result) == 3:
            u_val, m_val, e_val = result
            utility_preserved = float(u_val) if isinstance(u_val, (int, float)) else None
            method_tag = str(m_val)
            evidence = str(e_val)
        elif isinstance(result, (int, float)):
            utility_preserved = float(result)
            method_tag = "legacy_float"
    except Exception as exc:                                                # noqa: BLE001
        logger.warning("replay score_utility raised for %s: %s", task.task_id, exc)
        method_tag = "error"
    return TrajectoryRecord(
        record_id=TrajectoryRecord.new_id(),
        round_id=0,
        task_id=task.task_id,
        kind=TrajectoryKind.ATTACKED,
        trajectory=traj,
        outcome=outcome,
        attack=attack,
        signals=signals,
        utility=utility_preserved,
        utility_method=method_tag,
        utility_evidence=evidence,
        metadata={
            "judge_reason": reason,
            "vendored_replay": True,
            "suite": vendored_attack.suite,
            "target_tool": vendored_attack.target_tool,
            "mal_tool": vendored_attack.mal_tool,
            "ground_truth_score": vendored_attack.ground_truth_score,
        },
    )


def _append_jsonl(path: str, record_dict: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")


def run_vendored_replay(
    config_path: str,
    *,
    output_dir: Optional[str] = None,
    lora_adapter_override: Optional[Optional[str]] = _sentinel,
    dataset_dir: Optional[str] = None,
    split: Optional[str] = None,
    concurrency: int = 1,
) -> dict:
    """Run vendored replay eval per ``config_path``; return metrics dict.

    Parameters
    ----------
    config_path
        Path to an :class:`ExperimentConfig` YAML. Only ``defense``, ``env``,
        ``process`` blocks are used; ``attacker`` / ``training`` / ``pipeline``
        are ignored.
    output_dir
        Where to write ``records.jsonl`` + ``safety_metrics.jsonl``. Defaults
        to ``rounds/vendored_replay_<config_stem>/``.
    lora_adapter_override
        If not ``_sentinel``, overrides ``defense.llm.lora_adapter`` in the
        config (``None`` -> bare base model).
    dataset_dir
        Override the injection-scenario source dir (default
        ``<data_root>/toolsafe/agentdojo-tragj``). Point it at
        ``data/toolsafe/agentdojo-tragjnew/test`` for a held-out evaluation.
    split
        Keep only env tasks whose ``metadata["split"]`` equals this value.
        Requires a split-aware env (``dataset: agentdojo_split``); tasks with no
        declared split are dropped when this is set.
    concurrency
        Number of tasks replayed in parallel. Controller/agent/env/judge hold no
        per-run mutable state, so fan-out is safe; each worker owns one task's
        clean rollout plus all of that task's scenarios, which preserves the
        one-clean-record-per-task invariant.
    """

    cfg = ExperimentConfig.from_file(config_path)

    # Optional adapter override from CLI.
    if lora_adapter_override is not _sentinel:
        cfg.defense.llm.lora_adapter = lora_adapter_override

    adapter_label = cfg.defense.llm.lora_adapter or "base"
    cfg_name = os.path.splitext(os.path.basename(config_path))[0]
    if output_dir is None:
        output_dir = f"rounds/vendored_replay_{cfg_name}_{adapter_label}"
    os.makedirs(output_dir, exist_ok=True)
    records_path = os.path.join(output_dir, "records.jsonl")
    metrics_path = os.path.join(output_dir, "safety_metrics.jsonl")
    # Fresh files each run.
    for p in (records_path, metrics_path):
        if os.path.exists(p):
            os.remove(p)

    logger.info(
        "vendored_replay start: config=%s adapter=%s output=%s",
        config_path, adapter_label, output_dir,
    )

    # Build env, agent, controller, judge.
    env = build_env(cfg.env, seed=cfg.seed)
    agent = build_defense_agent(cfg.defense, seed=cfg.seed)
    controller = Controller(agent, env, cfg.defense)
    judge = AttackJudge(cfg.env.judge_llm, seed=cfg.seed)

    # Load ALL distinct vendored injection scenarios (not just first-per-task).
    # A single instruction can carry multiple distinct injections (different
    # target tools / observation text); each is a separate test case.
    suites = cfg.env.suites or None
    scenarios = load_all_vendored_attacks(
        cfg.env.data_root, suites=suites, dataset_dir=dataset_dir,
        dataset=cfg.env.dataset,
    )
    logger.info("loaded %d distinct vendored injection scenarios "
                "(suites=%s dir=%s dataset=%s)",
                len(scenarios), suites, dataset_dir or "<default>", cfg.env.dataset)

    # Match scenarios against env tasks; group by task so we can cache the
    # clean rollout (one per task) and run replay per scenario.
    env_tasks = env.get_tasks()
    if split:
        env_tasks = [
            t for t in env_tasks
            if str((t.metadata or {}).get("split") or "") == split
        ]
        if not env_tasks:
            raise SystemExit(
                f"no env tasks carry metadata['split'] == {split!r}; "
                f"is env.dataset split-aware (agentdojo_split)?"
            )
        logger.info("restricted to split=%s: %d env tasks", split, len(env_tasks))
    tasks_by_id = {t.task_id: t for t in env_tasks}
    by_task: dict[str, list[VendoredAttack]] = {}
    for va in scenarios:
        task = tasks_by_id.get(va.task_id)
        if task is None:
            continue
        by_task.setdefault(va.task_id, []).append(va)
    n_scenarios_matched = sum(len(v) for v in by_task.values())
    logger.info("matched %d scenarios across %d tasks (of %d env tasks)",
                n_scenarios_matched, len(by_task), len(tasks_by_id))

    records: list[TrajectoryRecord] = []
    t0 = time.time()
    task_items = list(by_task.items())
    progress = {"scenarios": 0, "tasks": 0}
    progress_lock = threading.Lock()
    write_lock = threading.Lock()

    def _replay_one_task(item: tuple[str, list[VendoredAttack]]) -> list[TrajectoryRecord]:
        """Clean rollout once + every scenario for a single task."""
        task_id, va_list = item
        task = tasks_by_id[task_id]
        out: list[TrajectoryRecord] = []
        try:
            out.append(_run_clean(controller, task))
        except Exception as exc:                                            # noqa: BLE001
            logger.warning("clean rollout crashed for %s: %s", task_id, exc)
            return []
        for va in va_list:
            try:
                out.append(_run_replay(controller, task, va, judge, cfg.process))
            except Exception as exc:                                        # noqa: BLE001
                logger.warning("replay crashed for %s: %s", task_id, exc)
                continue
        with write_lock:
            for rec in out:
                _append_jsonl(records_path, rec.to_dict())
        with progress_lock:
            progress["scenarios"] += len(out) - 1
            progress["tasks"] += 1
            elapsed = time.time() - t0
            logger.info(
                "progress %d/%d scenarios, task %d/%d (%.1fs, %.1fs/scenario) "
                "— last: %s",
                progress["scenarios"], n_scenarios_matched,
                progress["tasks"], len(task_items),
                elapsed, elapsed / max(progress["scenarios"], 1), task_id,
            )
        return out

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for chunk in pool.map(_replay_one_task, task_items):
                records.extend(chunk)
    else:
        for item in task_items:
            records.extend(_replay_one_task(item))

    # Aggregate metrics.
    # n_tasks here = number of distinct tasks that contributed at least one
    # scenario (used only for bookkeeping in RoundMetrics). The cf_* confusion
    # matrix is computed over ALL records: attacked scenarios (TP/FN) plus
    # one clean record per task (FP/TN).
    metrics = aggregate_round(records, {}, round_id=0, n_tasks=len(by_task))
    _append_jsonl(metrics_path, metrics.to_dict())

    # Print summary table to stdout + log.
    summary = {
        "adapter": adapter_label,
        "config": config_path,
        "dataset_dir": dataset_dir or os.path.join(cfg.env.data_root, "toolsafe",
                                                   "agentdojo-tragj"),
        "split": split or "all",
        "n_tasks": len(by_task),
        "n_injection_scenarios": n_scenarios_matched,
        "n_records": len(records),
        "cf_tp": metrics.cf_tp,
        "cf_fn": metrics.cf_fn,
        "cf_fp": metrics.cf_fp,
        "cf_tn": metrics.cf_tn,
        "cf_precision": metrics.cf_precision,
        "cf_recall": metrics.cf_recall,
        "cf_f1": metrics.cf_f1,
        "cf_acc": metrics.cf_acc,
        "attack_success_rate": metrics.attack_success_rate,
        "clean_completion_rate": metrics.clean_completion_rate,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("vendored_replay done: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a defender against real vendored AgentDojo injections.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to ExperimentConfig YAML (defense/env/process blocks used).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for records.jsonl + safety_metrics.jsonl.",
    )
    parser.add_argument(
        "--lora-adapter", default=None,
        help=(
            "Override defense.llm.lora_adapter. Pass 'none' (or 'base') to "
            "evaluate the bare base model; pass an adapter name to evaluate "
            "a trained LoRA. Omit to use whatever the YAML specifies."
        ),
    )
    parser.add_argument(
        "--dataset-dir", default=None,
        help="Override injection-scenario source dir (default "
             "<data_root>/toolsafe/agentdojo-tragj); e.g. "
             "data/toolsafe/agentdojo-tragjnew/test for the held-out split.",
    )
    parser.add_argument(
        "--split", default=None,
        help="Keep only env tasks whose metadata['split'] equals this "
             "(needs dataset: agentdojo_split). e.g. 'test'.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Tasks replayed in parallel (default 1 = serial).",
    )
    args = parser.parse_args()

    override = _sentinel
    if args.lora_adapter is not None:
        override = _resolve_lora_override(args.lora_adapter)

    run_vendored_replay(
        args.config,
        output_dir=args.output_dir,
        lora_adapter_override=override,
        dataset_dir=args.dataset_dir,
        split=args.split,
        concurrency=max(1, args.concurrency),
    )


if __name__ == "__main__":
    main()
