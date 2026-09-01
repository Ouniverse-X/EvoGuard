# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

EvoGuard researches **agent tool-call safety** (defense against prompt-injection / indirect tool-injection). Core loop: an **attack generator** and a **defense agent** improve against each other over rounds until the attacker stops succeeding.

Source-of-truth docs: `docs/evoguard.md` (Chinese plan doc) + `data/intro.md`; feature specs under `docs/superpowers/specs/YYYY-MM-DD-*-design.md`; `docs/todo.md` tracks work items; `docs/delta_signal_essence.md` documents how Δ signals are derived.

## Method

Per task, the controller produces **tri trajectories**: **A** = clean, **B** = attack-success, **C** = attack-fail. Signals: **injection point** (recorded by attacker), **turning point** (first A-vs-B divergence via edit distance, `process/edit_distance.py`), **Δ** = turning − injection (large Δ = latent attack). Loop terminates when no B trajectories remain (or ASR < ε=0.05 for K=5 consecutive rounds on held-out validation; state machine in `utils/metrics.py::RoundMetrics`).

- **Attacker**: Δ-guided MCTS (`attacks/mct_searcher.py::DeltaGuidedMCTSAttacker`), selected via `AttackerConfig.search_method="mcts_delta"`. A legacy genetic-algorithm backend (`attacks/genetic.py`) still exists but is not used; both are constructed via `build_attacker()` in `attacks/__init__.py`.
- **Defender**: LoRA adapters. Two paths — *vendored* (subprocess to LLaMA-Factory/AEPO, `training/sft_runner.py` + `grpo_runner.py`) and *native* (TRL in-process, `training/native_runner.py` + `native_grpo_runner.py`; **prefer native for new work**). Dispatch on `TrainingConfig.method ∈ {native_sft | native_grpo | sft_then_native_grpo | sft_then_online_grpo}` in `training/__init__.py`.

## Repository Structure

- `evoguard/run.py` — CLI (`python -m evoguard.run --smoke | --config path`); `config.py` — dataclass `ExperimentConfig` (YAML/JSON)
- `controller.py` — builds clean vs attacked contexts, drives tri trajectories
- `judge.py` — attack success verdicts (strict JSON schema in `llm/schemas.py`)
- `agents/` — defense agent base + LLM subclass (vLLM backend)
- `attacks/` — MCTS (`mct_searcher.py`, active) + legacy GA (`genetic.py`), via `build_attacker()` factory
- `envs/` — `ToolEnv` interface; tool execution simulated by LLM. New dataset envs subclass `SimulatedToolEnv` + `envs.register_env(name, builder)`
- `rollouts/` — unified rollout interface (`collect_tri_rollouts`); extend `RolloutStrategy` per dataset
- `process/` — signals (Δ, edit distance), `dataset_builder.py` (SFT corpus), `evo_data_exporter.py`, vendored attack parsing; `bench_*` = v2 Δ-bench format + release gate; `synth_replant.py`
- `training/` — vendored + native SFT/GRPO paths (see above); `grpo_reward.py` (reward), `grpo_prompt_extraction.py` (prompt sampling); LoRA layer probe under `training/probes/` (overrides r0 `target_modules` when `lora_probe_*` fields set)
- `online/` — single-stage online co-evolution (`train_online_grpo`; needs `_online_ctrl_factory` on `process.dataset_builder`); spec §2026-07-28
- `preliminary/` — AS-vs-AF sequence-probe pipeline (collect → probe → analyze)
- `bench_base/` — base-model Δ-statistics bench; own `scripts/`; `CHANGELOG.md` is provenance record (note: ~130 MCTS entries have `mcts_judge_verified == false` — filter or re-validate)
- `baselines/shieldagent/` — standalone ShieldAgent baseline, independent of main pipeline
- `evoguard/llm/` — pluggable clients (`mock_client.py` role-marker routing, `openai_client.py`, `qianfan_client.py`); structured output with graceful degrade (`_schema_supported` 3-state cache)
- `evoguard/eval/` — `stepwise_eval.py`, `vendored_replay.py` (true tool-executing replay), `autodojo_eval.py`; needs vLLM :8000 + judge :8002
- `data/toolsafe/` — step-level safety annotations (AgentHarm/AgentDojo trajectories); `data/agentdojo/`, `data/agentharm/` vendored tool defs
- `rounds/<exp>/round_<id>/` — per-round JSONL artifacts; `results/` summaries + curves
- `configs/` — YAML experiment configs; `scripts/` — thin bash wrappers; `docs/` — design notes (save implementation explanations here)

## Common commands

Python interpreter: `/ssd1/conda_envs/evoguard/bin/python` (conda env `evoguard`, NOT miniforge). `pip install -r requirements.txt` for lightweight deps only; heavy torch+vllm pre-installed in the env.

### Offline tests (no GPU/network)

```bash
bash scripts/run_smoke.sh                          # full-pipeline sanity vs MockClient
python -m evoguard.tests.<module>                  # e.g. smoke_test, test_schemas, test_mcts_attacker,
                                                   # test_native_grpo_reward, test_gdpo_advantages,
                                                   # test_native_grpo_advantage_shaping,
                                                   # test_grpo_trajectory_groups, test_prompt_extraction,
                                                   # test_probe_ranking, test_sft_corrective_templates
```
Plain module-level `main()` functions, not pytest fixtures; single case via `-k <name>`.

### Real-model experiments

```bash
bash scripts/start_vllm.sh                         # defender vLLM :8000 (writes rounds/vllm.pid)
bash scripts/start_vllm_secondary.sh               # judge vLLM :8002
bash scripts/register_vllm_lora.sh                 # hot-register trained LoRA adapter
scripts/run_real.sh configs/real_qianfan_smoke.yaml  # QianFan backend; secrets from
                                                   # ${EVOGUARD_SECRETS_FILE:-$HOME/.evoguard_qianfan.env}
```

### Native-trainer experiments (preferred)

Method via YAML `method` field: `native_sft` (cold-start), `native_grpo` (incremental; needs `<exp>/latest_adapter_dir.txt`), `sft_then_native_grpo` (r0→SFT, ≥r1→GRPO), `sft_then_online_grpo` (live G-sibling rollouts). MCTS attacker variant: `configs/agentdojo_full_mcts.yaml`.

```bash
bash scripts/run_grpo_experiment.sh [config]       # default: agentdojo_full_grpo.yaml
bash scripts/run_lora_probe.sh <config>            # pre-training LoRA probe → overrides r0 target_modules
bash scripts/run_stepwise_eval.sh [config]         # stepwise eval vs AgentDojo test trajectories
bash scripts/run_replay_heldout.sh                 # held-out full-replay eval (see Eval conventions)
```

### Preliminary / bench_base

```bash
bash scripts/run_preliminary_tier1_multidomain_collect.sh   # Phase-1 rollout collection
bash scripts/run_seq_probe.sh                               # Phase-2 feature extraction (needs vLLM)
python -m preliminary.analyze <probe_output.json>           # Phase-3 statistics
bash bench_base/scripts/evolve_multidomain.sh               # MCTS Δ evolution
bash bench_base/scripts/release_gate.sh                     # v2 schema + power release gate
```

## Conventions — load-bearing invariants

Detailed history/measurements behind each rule live in git log and `docs/`; only the rules are repeated here.

### Reward (GRPO)

- **`R = r_safety + r_progress − p_drift`**, exactly three terms. Do NOT add 4th/5th/6th terms (e.g. `p_no_terminate`, `p_repeat`) — the standing instruction is a SIMPLER reward. Constants live in `grpo_reward.py`: safety `{held:+2.00, held_but_fired:−1.00, baited:−8.00, unclear:−0.50, clean_served:+2.00}`; progress `{advance:+1.20, neutral:−0.15, waste:−2.50}`. `R_SAFETY_BAITED=−8.00` is the ONE risk-aversion knob (do not change without recomputing break-even p*≈27%). `R_PROGRESS_NEUTRAL` is also the judge-outage fallback.
- **"Do nothing" must never be optimal**: no tool call AND no final answer → `WASTE` (−2.50), even if `thought` is verbose. Do not re-add `and not thought_str.strip()`.
- **Structural pre-check branch ORDER in `_progress_from_structure` is load-bearing**: (i) no action→WASTE; (ii) no tool name→judge; (iii) signature already in prefix→WASTE; (iv) == clean twin's next action→ADVANCE; (v) else judge. (iii) MUST stay before (iv) or re-issuing calls becomes profitable (re-opens the non-terminating loop). Structurally settled steps never reach the judge.
- **Progress judge prompt bullet order `neutral / waste / advance` is a measured hyperparameter** (9/12 vs 6–8/12 for other permutations). Do NOT reorder/reword `_PROGRESS_SYSTEM_PROMPT` bullets without re-measuring.
- **Δ enters ONLY via advantage shaping** `Ã=(1+λ·δ_p)·A` (`grpo_advantage_curriculum_lambda`); λ=0.0 silently discards Δ. Never re-add an inline Δ reward term; never flip to `(1−Δ)`.

### GRPO mechanics

- **GDPO** (`grpo_gdpo=True`): per-term group-wise standardisation then batch rescale — a normalisation change, reward untouched. Intervention order in `_DeltaShapedGRPOTrainer`: **GDPO → trajectory pooling → Δ shaping**; do not permute. Locked by `test_gdpo_advantages.py`.
- **Trajectory groups** (`grpo_traj_group_size=K`, ships K=2): K rows per trajectory share `traj_group_id`, group-major contiguous, `shuffle_dataset=False`. `_traj_pooled_advantage_overrides` re-bases zero-std groups against a trajectory-pooled mean — a FALLBACK only. K=1 is a bit-for-bit no-op. `grpo_max_prompts_per_round` is a ROW budget (trajectories = budget//K); `native_max_steps_per_round` (400) binds total completions regardless.
- **Reward saturation was an exploration problem** (reward_std==0 in 62–73% of steps → zero gradient). Fixed by `grpo_rollout_temperature: 1.15` + trajectory pooling, NOT by reward surgery. The decision-step retarget to the bait step is an audited guardrail, no-op on AgentDojo (Δ=0 dominates) — do not delete it, do not claim it matters here.
- Judge endpoints are **opt-in env vars**: `EVOGUARD_JUDGE_LLM_BASE_URL` (safety), `EVOGUARD_PROGRESS_LLM_BASE_URL` (progress; falls back to safety URL), `EVOGUARD_REWARD_JUDGE_WORKERS`. Every pre-2026-08-21 run scored `r_safety` from heuristic fallback — caveat when reading historical results. Confirm the log shows `progress judge active` with no WARNING naming either variable.

### SFT corpus (`process/dataset_builder.py`)

- **Corrective-refusal surface diversity**: paraphrase pools (`_CORRECTIVE_THOUGHT_GROUNDED/UNGROUNDED`, `_CORRECTIVE_CONTINUE_SUFFIX`, `_CORRECTIVE_FINAL_ANSWER`), top surface share < 0.30; detection claim must quote the actual payload (`_payload_excerpt`); NO instruction echo (never re-add the `original task (…)` splice). Pool selection is `_stable_choice` (sha1) — rebuilds are byte-identical.
- **Grounded refusal placement**: refusal only at a step whose PROMPT already contains the payload (`_grounded_refusal_index`): index 0 never eligible; payload probe must appear in rendered history; never past the bait-taking action. Records without such a step (or without a turning point) are DROPPED — never imitate a B trajectory verbatim.
- **Source-utility gate** (`sft_min_source_utility`): below-bar clean rollouts dropped AND removed from `cleans_by_task`; below-bar C rollouts truncated to pre-injection prefix; C rollouts whose payload never surfaced are dropped. `utility is None` always passes.
- **Prompt-level geometry**: `n_refusal_on_payload_free` MUST stay 0. Do NOT re-add any global provenance/ratio knob — both directions measured harmful.
- **Corrective-share cap supersedes two-class mode** (`sft_max_corrective_share`; two-class `sft_two_class` still exists but is not what ships). `_cap_per_task` keys on `(task_id, clean/attack_fail/attack_success)` three-way, never `task_id` alone.

### Clean-twin step identity

`ADVANCE` is awarded only on STRICT equality of `_action_signature` (= `name|json.dumps(args, sort_keys=True)` after normalisation) with the clean twin's next action, AND absent from prefix. Not subset — one extra arg breaks the match; near-miss returns `None` → judge. `grpo_prompt_extraction` imports `_action_signature` from `grpo_reward`, so formats can never drift — a hand-written signature in any other form silently disables the loop check.

### Metrics & evaluation

- Renamed 2026-08-20 (`schema_version` 2; readers accept both): `dos_overblock_rate`→**`blocked_unfinished_rate`**, bench `overblock`→**`false_alarm`**. Don't conflate `judge._REFUSAL_RE` (declining to act) with `INJECTION_FLAG_RE` (announcing an injection). Check `n_utility_fallback_mid`: utility-judge fallback scores exactly 0.0 and silently count as "delivered".
- **Held-out evaluation MUST be a full replay** (`eval/vendored_replay.py` / `scripts/run_replay_heldout.sh` + `summarize_replay.py`). `bench_base`'s "full rollout" executes no tools (appends empty observations) — treat its numbers as paired A/B between adapters only, never absolute performance; `stepwise_eval.py` has no env loop.
- **Always report `poison_delivered_rate` and `asr_given_delivered` alongside ASR**: much of the TP improvement can be avoidance (routing around the poisoned tool), not resistance.
- **Training-round metric logs are bad selectors** — evaluate the best round too, not just the last (per-task forgetting happens; `grpo_beta=0.01` is a loose KL leash).
- Val split is **37 train / 10 val / 11 test**; `test/` is sha256-guarded (SystemExit if changed) so historical numbers stay comparable. `Pipeline.run_validation` replays the val split per round into `<exp>/val_metrics.jsonl`; pre-2026-08-23 runs early-stopped on TRAINING ASR (tracks attacker strength, not defender quality). Replay must pass `dataset: agentdojo_split` with `--dataset-dir` at the LEAF split dir.

### Negative results — do not repeat

- **`last_traj` (K=2 grouping)**: fixed reward saturation (median never pinned at ceiling) but did NOT improve held-out ASR vs plan_abc. Retained for its saturation fix; conclusion: saturation was not the binding constraint — spend effort on avoidance-vs-resistance TPs and per-task forgetting, not group/advantage surgery.
- **Blanket-refusal / stalling diagnoses from bench_base** were harness artifacts (empty observations), see above.
- D2 provenance-ratio trim: removed, moved the metric backwards.

## Vendored Training Frameworks

`evoguard/training/AEPO/` and `evoguard/training/LLaMA-Factory/` vendor complete third-party projects — do NOT edit directly; integration is via subprocess shells (`dry_run=True` inspection supported). Their pinned deps CONFLICT with the installed runtime stack (`numpy 2.x`/`peft 0.19`/`trl 0.19`/`transformers 5.x`); do not rely on `llamafactory-cli` or `verl.trainer.main_ppo` at runtime — use the native path.

No top-level install metadata; runtime imports rely on PYTHONPATH = repo root.
