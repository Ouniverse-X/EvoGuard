"""Top-level co-evolution loop (``docs/plan.md``, ``CLAUDE.md``).

Ties every component together:

1. Build env + defense agent + attacker generator from :class:`ExperimentConfig`.
2. Split tasks into train / validation subsets.
3. Maintain one per-task :class:`~evoguard.attacks.GeneticAttacker` whose
   population is rolled out, evaluated and evolved each round.
4. For each round:
   a. Collect the (A, B/C...) pair for every task via
      :func:`evoguard.rollouts.collect_tri_rollouts`.
   b. Aggregate metrics and persist artifacts under ``rounds/<exp>/round_<id>``.
   c. Evolve each task's population (elitism + tournament selection with
      diversity penalty + LLM crossover/mutation).
   d. Optionally run defender training (SFT cold-start then GRPO on LoRA) using
      this round's records; if it produced a new adapter name, point subsequent
      rounds' defense agent at it.
   e. Check termination criteria (zero successful attacks OR K consecutive
      rounds below ASR threshold ε).
5. Render final co-evolution curves.

The pipeline never crashes the whole experiment when an individual rollout or
training step fails: those errors are logged and that task/round's results are
skipped so partial progress survives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from evoguard.agents import build_defense_agent
from evoguard.attacks import GeneticAttacker, build_attack_generator
from evoguard.config import ExperimentConfig
from evoguard.controller import Controller
from evoguard.envs import build_env
from evoguard.judge import AttackJudge
from evoguard.llm.base import LLMClient
from evoguard.process.dataset_builder import DefenderDatasetBuilder
from evoguard.rollouts import collect_tri_rollouts
from evoguard.utils.logging import get_logger
from evoguard.utils.metrics import (
    RoundMetrics,
    aggregate_round,
    append_safety_metrics_jsonl,
    update_termination_state,
)
# Note: plots imported lazily inside run() to keep matplotlib optional at import.
# Note: training.train_defender imported lazily for same reason (it pulls in
# dataclass-only deps but we want to defer any side effects).

logger = get_logger("pipeline")


@dataclass
class RoundResult:
    """Per-round outcome returned by :meth:`Pipeline.run_one`."""

    round_id: int
    rollouts: object  # RoundRollouts; opaque to avoid circular imports here
    metrics: RoundMetrics


@dataclass
class ExperimentSummary:
    """Final summary persisted to ``summary.json``."""

    config_path: str = ""
    exp_dir: str = ""
    n_rounds_completed: int = 0
    terminated: bool = False
    terminate_reason: str = ""
    best_metrics_by_metric: dict[str, float] = field(default_factory=dict)
    last_metrics_jsonl_path: str = ""
    curves_png_path: str | None = None


def _split_train_val(tasks: list, val_fraction: float):
    """Deterministic split preserving dataset order; val takes tail fraction."""
    if not tasks:
        return [], []
    n_val = max(0, min(len(tasks), int(round(len(tasks) * val_fraction))))
    if n_val == 0:
        return list(tasks), []
    return list(tasks[:-n_val]), list(tasks[-n_val:])


def _build_defense_agent(cfg: ExperimentConfig):
    """Construct the current-round defense agent honoring LoRA adapter choice."""
    return build_defense_agent(
        cfg.defense,
        seed=cfg.seed,
        client=_build_shared_client_for_role("defense", cfg),
    )


def _build_shared_client_for_role(role: str, cfg: ExperimentConfig):
    """Return ``None`` so each agent gets its own client built lazily.

    Kept as a hook so future experiments can share a single OpenAIClient across
    roles if they all hit the same endpoint -- but currently leaving role-specific
    clients lets us tune temperature/max_tokens independently per role without
    re-issuing requests through shared state.
    """
    return None  # noqa: ARG001 - intentional placeholder hook


class Pipeline:
    """Orchestrates EvoGuard end-to-end against one configuration."""

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.exp_dir = os.path.join(config.rounds_dir, _safe_name(config.name))
        self._tasks_cache: dict[str, tuple] = {}      # task_id -> (task, tools)
        self._dataset_builder: DefenderDatasetBuilder | None = None
        self._attacker_clients_built = False
        # Held-out / training-task bookkeeping populated by :meth:`setup`.
        self._train_tasks: list = []
        self._val_tasks: list = []
        # Latest training-side EvaluatedAttack list per task -- captured *before*
        # ``evolve_after_round`` mutates populations so the validation phase can
        # still inspect this round's measured elites ("the attacker generator's
        # current best individual" per docs/plan.md line 9).
        self._latest_train_evals_by_task: dict[str, list] = {}

    # ------------------------------------------------------------------ #
    # Public entry points                                                #
    # ------------------------------------------------------------------ #
    def setup(self):
        """Pre-build heavy objects once before running rounds.

        Returns ``(env, judge, attackers_init_dict, attack_generator)`` where
        ``attackers_init_dict`` maps **training** task_ids -> ready-to-seed GA
        instances. Validation tasks do NOT get their own GAs -- they are probed
        later (:meth:`run_validation`) using elites copied from the training
        side.
        """

        # Build environment first so we know which tasks exist.
        env = build_env(self.cfg.env)

        # Cache tasks & tool lists so downstream builders stay cheap.
        tasks_all = env.get_tasks()
        train_tasks, val_tasks = _split_train_val(
            tasks_all, self.cfg.pipeline.validation_fraction
        )
        self._train_tasks = list(train_tasks)
        self._val_tasks = list(val_tasks)
        logger.info(
            "Setup complete: %d total tasks (%d train, %d val)",
            len(tasks_all), len(train_tasks), len(val_tasks),
        )

        # Judge uses its own client configured separately in EnvConfig.judge_llm.
        judge = AttackJudge(self.cfg.env.judge_llm)
        gen = build_attack_generator(self.cfg.attacker, seed=self.cfg.seed)

        # Pre-instantiate GAs ONLY for training tasks. Validation will reuse
        # elite genomes copied verbatim from training side rather than running
        # its own evolutionary search -- otherwise val would leak signal back
        # into stopping criteria defeating independence.
        attackers: dict[str, GeneticAttacker] = {}
        for t in train_tasks:
            tools = env.get_tools(t)
            ga = GeneticAttacker(
                t, tools, gen, self.cfg.attacker,
                defense_max_turns=self.cfg.defense.max_turns,
            )
            attackers[t.task_id] = ga
            self._tasks_cache[t.task_id] = (t, tools)

        # Dataset builder needs registries spanning BOTH splits (corrective SFT
        # examples reference paired-clean trajectories regardless of split).
        self._build_dataset_builder(env, tasks_all)

        return env, judge, attackers, gen

    def save_config_snapshot(self) -> str:
        path = os.path.join(self.exp_dir, "config.yaml")
        self.cfg.save(path)
        return path

    # ------------------------------------------------------------------ #
    # Per-round execution                                               #
    # ------------------------------------------------------------------ #
    def run_one(
        self,
        *,
        env,
        judge,
        attackers: dict[str, "GeneticAttacker"],
        round_id: int,
        tasks_subset=None,
    ) -> RoundResult:
        """Run exactly one co-evolution round over the given task subset."""
        controller_agent = _build_defense_agent(self.cfg)
        controller = Controller(controller_agent, env, self.cfg.defense)

        tasks_to_run = [
            t for t in env.get_tasks()
            if (attackers.get(t.task_id) is not None)
            and (tasks_subset is None or t.task_id in {x.task_id for x in tasks_subset})
        ]

        result_obj = collect_tri_rollouts(
            controller=controller,
            tasks=tasks_to_run,
            attackers={tid: attackers[tid] for tid in [t.task_id for t in tasks_to_run]},
            judge=judge,
            process_config=self.cfg.process,
            round_id=round_id,
            task_concurrency=getattr(self.cfg.pipeline, "task_concurrency", 1),
            attack_concurrency=getattr(self.cfg.pipeline, "attack_concurrency", 1),
        )

        n_total_tasks_in_env = sum(1 for t in env.get_tasks())
        metrics = aggregate_round(
            result_obj.records,
            result_obj.evaluations,
            round_id=round_id,
            n_tasks=n_total_tasks_in_env,
        )

        # Persist round artifacts under <exp_dir>/round_<id>/
        populations_by_task = {
            tid: attackers[tid].current_population()
            for tid in result_obj.evaluations.keys()
        }
        try:
            from .io import save_round
            paths = save_round(
                exp_dir=self.exp_dir,
                round_id=round_id,
                records=result_obj.records,
                populations_by_task=populations_by_task,
                evaluations_by_task=result_obj.evaluations,
            )
            write_metrics_file(os.path.join(paths["round_dir"], "metrics.json"), metrics.to_json())

            append_line(
                os.path.join(self.exp_dir, "metrics.jsonl"),
                json_dump(metrics.to_dict()),
            )
            # Per docs/todo.md item #4: stream safety metrics (acc / f1 /
            # recall + existing ASR/delta signals) into <exp>/results/.
            try:
                append_safety_metrics_jsonl(self.exp_dir, metrics)
            except Exception as exc:
                logger.warning(
                    "[round %d] results/ JSONL write failed: %s", round_id, exc,
                )
            logger.info("[round %d] saved %d records -> %s", round_id,
                        len(result_obj.records), paths["records"])
        except Exception as exc:
            logger.error("[round %d] artifact persistence failed: %s", round_id, exc)

        return RoundResult(round_id=round_id, rollouts=result_obj, metrics=metrics)

    def evolve_after_round(self, attackers, result_obj):
        """Advance every task's population based on this round's evals."""
        for tid, evals in getattr(result_obj, "evaluations", {}).items():
            ga = attackers.get(tid)
            if ga is None:
                continue
            try:
                ga.evolve(evals)
            except Exception as exc:
                logger.warning("Evolution failed for task=%s: %s", tid, exc)

    def maybe_train_defender(self, *, round_label: str, records, round_id: int = 0):
        """Build datasets/configs and optionally launch SFT+GRPO training.

        Honors ``TrainingConfig.sft_coldstart_only_round_zero``: when set (and an
        adapter from a prior round is already loaded), rounds after r0 skip the
        SFT cold-start phase entirely and only run GRPO incrementally on new
        successful B-trajectories. This eliminates ~30-40 min/round of redundant
        cold-init compute observed in evoguard_agentdojo_full where every round
        re-ran SFT from scratch.
        """
        if not self.cfg.training.enabled or self._dataset_builder is None:
            return None

        # Skip SFT cold-start on rounds > 0 when configured + adapter already live.
        skip_cold_start = (
            self.cfg.training.sft_coldstart_only_round_zero
            and round_id > 0
            and bool(getattr(self.cfg.defense.llm, "lora_adapter", "") )
        )
        # GRPO needs fresh signal to be worth firing; count NEW B-trajectories.
        n_new_success_b = sum(
            1 for rec in records
            if getattr(rec, "outcome", None) is not None
            and rec.outcome.value == "success"
        )

        grpo_min = max(0, int(self.cfg.training.grpo_min_new_successes))
        if (
            self.cfg.training.sft_coldstart_only_round_zero
            and round_id > 0
            and n_new_success_b < grpo_min
        ):
            logger.info(
                "[train] skipping defender update at %s: only %d new successes "
                "(threshold=%d); keeping existing lora_adapter=%r",
                round_label,
                n_new_success_b,
                grpo_min,
                getattr(self.cfg.defense.llm, "lora_adapter", ""),
            )
            return None

        try:
            from evoguard.training import train_defender

            effective_cfg = self.cfg.training
            method_override = None
            if skip_cold_start:
                # Switch this call's method to "grpo" so we don't redo cold start;
                # we leave cfg.training.method untouched for next-round reference.
                try:
                    import dataclasses as _dc
                    effective_cfg = _dc.replace(effective_cfg, method="grpo")
                except Exception:
                    method_override = "grpo"

            outcome_kwargs = dict(
                records=list(records),
                exp_rounds_root=self.exp_dir,
                training_cfg=effective_cfg,
                round_label=round_label,
                dataset_builder=self._dataset_builder,
            )
            if method_override is not None:
                logger.info(
                    "[train] skipping SFT cold-start at %s; running incremental "
                    "%s against existing adapter=%r",
                    round_label, method_override.upper(),
                    getattr(self.cfg.defense.llm, "lora_adapter", ""),
                )
                outcome_kwargs["training_cfg"] = _replace_training_method(
                    self.cfg.training, method_override,
                )

            outcome = train_defender(**outcome_kwargs)
            # Only propagate the suggested adapter name into the live runtime
            # config when an actual LoRA artifact was trained AND loaded onto the
            # serving backend. Under ``training.dry_run=True`` (or when
            # ``new_lora_adapter_name`` is empty) we MUST keep ``lora_adapter``
            # as-is, otherwise subsequent rounds try to request a non-existent
            # model name and crash with HTTP 404 against vLLM.
            if (
                outcome is not None
                and outcome.new_lora_adapter_name
                and not self.cfg.training.dry_run
            ):
                old_name = self.cfg.defense.llm.lora_adapter
                new_name = f"{old_name}::{outcome.new_lora_adapter_name}" if old_name else \
                           outcome.new_lora_adapter_name

                # Hot-load the freshly-trained adapter onto the running vLLM
                # server so the next round's defense-agent calls can request it
                # by name without restarting the server between rounds.
                lora_path = getattr(outcome, "adapter_dir", "") or ""
                registered_ok = False
                if lora_path:
                    try:
                        import os as _os
                        import subprocess as _sp
                        script = _os.path.join(
                            _os.path.dirname(_os.path.dirname(_os.path.dirname(
                                _os.path.abspath(__file__)
                            ))),
                            "scripts", "register_vllm_lora.sh",
                        )
                        base_url = getattr(self.cfg.defense.llm, "base_url",
                                           "http://127.0.0.1:8000/v1") or ""
                        port = (
                            int(base_url.rstrip("/").split(":")[-1].split("/")[0])
                            if ":" in base_url else 8000
                        )
                        proc = _sp.run(
                            ["bash", script, new_name, str(lora_path), str(port)],
                            capture_output=True, text=True, timeout=60.0,
                        )
                        if proc.returncode == 0:
                            registered_ok = True
                            logger.info(
                                "[train] vLLM hot-loaded LoRA %r <- %s",
                                new_name, lora_path,
                            )
                        else:
                            logger.warning(
                                "[train] vLLM hot-load failed (rc=%d): "
                                "stdout=%s | stderr=%s; keeping previous adapter.",
                                proc.returncode,
                                proc.stdout.strip()[:200],
                                proc.stderr.strip()[:300],
                            )
                    except Exception as reg_exc:  # noqa: BLE001 - never crash round-loop here
                        logger.warning(
                            "[train] register_vllm_lora helper raised: %s; "
                            "continuing with previous adapter.", reg_exc,
                        )

                self.cfg.defense.llm.lora_adapter = (
                    new_name if registered_ok else old_name
                )
                action = "updated" if registered_ok \
                         else "kept-previous-after-failed-register"
                logger.info(
                    "[train] defense lora_adapter %r -> %r (%s)",
                    old_name, self.cfg.defense.llm.lora_adapter, action,
                )
            elif outcome is not None and outcome.new_lora_adapter_name:
                logger.info(
                    "[train] dry-run mode: keeping base defense LLM "
                    "(suggested adapter %r not trained/registered on server)",
                    outcome.new_lora_adapter_name,
                )
            return outcome
        except Exception as exc:
            logger.error("[train] failed: %s", exc)
            return None
            return None

    # ------------------------------------------------------------------ #
    # Full driver                                                       #
    # ------------------------------------------------------------------ #
    def run(self) -> ExperimentSummary:
        import json as _json

        cfg_snap = self.save_config_snapshot()

        env, judge, attackers, gen = self.setup()

        streak = 0
        history_dicts: list[dict] = []

        completed = 0
        term_reason = ""
        terminated = False

        for rid in range(0, self.cfg.pipeline.max_rounds):
            rr = self.run_one(
                env=env, judge=judge, attackers=attackers, round_id=rid,
            )
            history_dicts.append(rr.metrics.to_dict())

            self.evolve_after_round(attackers, rr.rollouts)

            label = f"r{rid}"
            self.maybe_train_defender(
                round_label=label,
                records=[rec for rec in rr.rollouts.records],
                round_id=rid,
            )

            streak, stop_now, reason = update_termination_state(
                streak,
                rr.metrics,
                patience_rounds=self.cfg.pipeline.patience_rounds,
                asr_threshold=self.cfg.pipeline.asr_threshold,
                stop_on_zero_success=self.cfg.pipeline.stop_on_zero_success,
            )
            rr.metrics.consecutive_low_asr_streak = streak

            completed += 1
            if stop_now:
                terminated = True
                term_reason = reason
                break

        # Persist final summaries + curve plot.
        try:
            from evoguard.utils.plots import plot_curves, write_metrics_csv
            png_out = os.path.join(self.exp_dir, "curves.png")
            csv_out = os.path.join(self.exp_dir, "metrics.csv")
            plotted = plot_curves(history_dicts, png_out) if history_dicts else None
            csv_written = write_metrics_csv(history_dicts, csv_out) if history_dicts else ""
            logger.info("[done] wrote curves.png?%s ; csv=%s",
                        bool(plotted), csv_written)
        except Exception as exc:
            logger.warning("[done] plotting failed: %s", exc)
            png_out = ""

        # Per docs/todo.md item #4: also write a focused results/ folder with
        # tabular CSV + summary.json keyed on acc/f1/recall/precision + ASR/delta
        # signals so analysts can compare experiments without parsing JSONL.
        try:
            from evoguard.utils.metrics import (
                write_safety_summary_block,
                write_safety_metrics_csv,
            )
            results_csv = write_safety_metrics_csv(self.exp_dir, history_dicts)
            results_summ = write_safety_summary_block(self.exp_dir, history_dicts)
            logger.info("[done] results.csv=%s summary=%s",
                        os.path.basename(results_csv) if results_csv else "(none)",
                        os.path.basename(results_summ) if results_summ else "(none)")
        except Exception as exc:
            logger.warning("[done] results-folder aggregation failed: %s", exc)

        # Compute simple 'best-of-run' snapshot used in summary.json.
        best_asr_seen = None
        worst_delta_norm_mean = None
        best_precision = None
        best_recall = None
        best_f1 = None
        best_acc = None
        for m in history_dicts:
            cur_asr = float(m.get("attack_success_rate", 0))
            cur_dnmean = float(m.get("delta_normalized_mean_on_success", 0))
            if best_asr_seen is None or cur_asr > best_asr_seen:
                best_asr_seen = cur_asr
            if worst_delta_norm_mean is None or cur_dnmean > worst_delta_norm_mean:
                worst_delta_norm_mean = cur_dnmean

            def _best(cur_field: str | None, prev: float | None) -> float | None:
                v = m.get(cur_field)
                try:
                    fv = float(v) if v is not None else None
                except (TypeError, ValueError):
                    return prev
                return max(prev, fv) if (prev is not None and fv is not None) \
                    else (fv if fv is not None else prev)

            best_precision = _best("safety_precision", best_precision)
            best_recall = _best("safety_recall", best_recall)
            best_f1 = _best("safety_f1", best_f1)
            best_acc = _best("safety_acc", best_acc)

        # Best defender-side safety scores are "highest seen across rounds".
        # Attacker-side ASR/delta are reported as their peak observed value so a
        # reader can immediately see how strong the red team got.
        best_metrics_by_metric: dict[str, float] = {
            "max_attack_success_rate": best_asr_seen or 0.0,
            "max_delta_normalized_mean_on_success": worst_delta_norm_mean or 0.0,
        }
        if best_precision is not None:
            best_metrics_by_metric["max_safety_precision"] = round(best_precision, 6)
        if best_recall is not None:
            best_metrics_by_metric["max_safety_recall"] = round(best_recall, 6)
        if best_f1 is not None:
            best_metrics_by_metric["max_safety_f1"] = round(best_f1, 6)
        if best_acc is not None:
            best_metrics_by_metric["max_safety_acc"] = round(best_acc, 6)

        summary = ExperimentSummary(
            config_path=os.path.abspath(cfg_snap),
            exp_dir=os.path.abspath(self.exp_dir),
            n_rounds_completed=completed,
            terminated=terminated,
            terminate_reason=term_reason,
            best_metrics_by_metric=best_metrics_by_metric,
            last_metrics_jsonl_path=os.path.join(self.exp_dir, "metrics.jsonl"),
            curves_png_path=(png_out or None) or None,
        )
        write_summary_summary(summary)
        return summary

    # ------------------------------------------------------------------ #
    # Internal helpers                                                  #
    # ------------------------------------------------------------------ #
    def _build_dataset_builder(self, env, all_tasks):
        tasks_map: dict[str, object] = {}
        tools_map: dict[str, list] = {}
        for t in all_tasks:
            tasks_map[t.task_id] = t
            tools_map[t.task_id] = env.get_tools(t)
        self._dataset_builder = DefenderDatasetBuilder(tasks_map, tools_map)


# --------------------------------------------------------------------------- #
# Module-level helpers                                                        #
# --------------------------------------------------------------------------- #
_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _safe_name(name: str) -> str:
    out_chars = [c if c in _SAFE_CHARS else "_" for c in (name or "evoguard-exp")]
    safe = "".join(out_chars).strip("_") or "evoguard-exp"
    return safe[:128]


def write_metrics_file(path: str, text_payload: str) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text_payload)


def append_line(path: str, line: str) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n"))
        f.write("\n")


def json_dump(obj) -> str:
    import json as _json
    return _json.dumps(obj, ensure_ascii=False)


def write_summary_summary(s: ExperimentSummary) -> None:
    target = os.path.join(s.exp_dir, "summary.json")
    os.makedirs(s.exp_dir, exist_ok=True)
    payload = s.__dict__.copy() if hasattr(s, "__dict__") else dict(**vars(s))
    with open(target, "w", encoding="utf-8") as f:
        import json as _json2
        _json2.dump(payload, f, ensure_ascii=False, indent=2)


def _replace_training_method(training_cfg, new_method: str):
    """Return a shallow copy of ``training_cfg`` with ``method`` swapped.

    Used by :meth:`Pipeline.maybe_train_defender` to switch an SFT-then-GRPO
    config into GRPO-only mode for incremental rounds after the cold start,
    without mutating the live config object.
    """
    try:
        import dataclasses as _dc
        return _dc.replace(training_cfg, method=new_method)
    except Exception:
        # Fallback: mutate a copy via __dict__ if dataclasses.replace fails.
        copy = type(training_cfg)(**{**vars(training_cfg)})
        setattr(copy, "method", new_method)
        return copy


# Public surface ----------------------------------------------------------- #
__all__ = ["Pipeline", "RoundResult", "ExperimentSummary"]
