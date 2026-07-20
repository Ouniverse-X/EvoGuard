# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EvoGuard is a research project on **agent tool-call safety** (defending against prompt-injection / indirect tool-injection attacks). The core idea is a co-evolutionary loop: an **attack generator** and a **defense agent** are improved against each other over multiple rounds until the attacker can no longer succeed.

The design lives in `docs/plan.md` and per-folder `intro.md` files; both are written in Chinese and are the source of truth for intended behavior whenever an implementation detail looks ambiguous. `docs/todo.md` tracks outstanding work items.

## Method (from `docs/plan.md`)

The pipeline produces **dual/tri trajectories** for the same task via a controller:
- **A** = clean tool-call trajectory (controller gives clean context)
- **B** = attack-success trajectory (attacker's injected content spliced into context, attack succeeded)
- **C** = attack-fail trajectory (injection present but attack failed)

Key signals computed per trajectory:
- **Injection point** — which interaction turn the attacker injected at (recorded by the attack generator).
- **Behavior turning point** — first significant divergence between A and B action sequences, found via edit distance over (tool + args) in `evoguard/process/edit_distance.py`.
- **Δ (delta)** = turning point − injection point. Large Δ = "latent" attack (stealthy); small Δ = "immediate-trigger". Attacker wants large Δ; defender wants small Δ.

**Attacker** uses a genetic algorithm over a population of ~N=50 injection strategies (`evoguard/attacks/genetic.py`). Fitness = 0 if attack failed, else normalized Δ. Core operators: tournament selection (k=3) with diversity penalty for similar injection position/method, LLM-based crossover+mutation (~M=45 children), elite retention (E=5). Only attacks that fool the defense are eligible.

**Defender** is trained (SFT cold-start + GRPO, or GRPO alone) on **LoRA adapters** (not full base-model weights), driven by `evoguard/training/sft_runner.py` + `grpo_runner.py`.

The loop repeats until no B trajectories are produced (or attack success rate stays below threshold ε=0.05 for K≈5 consecutive rounds on a held-out validation set). Termination state machine lives in `evoguard/utils/metrics.py::RoundMetrics`.

Match plan hyperparameters exactly (N=50,M=45,E=5,k=3,K=5,ε=0.05).

## Repository Structure

- `evoguard/` — main package
  - `run.py` — CLI entry point (`python -m evoguard.run --smoke | --config path`)
  - `config.py` — dataclass-based `ExperimentConfig`; loads YAML/JSON, recurses nested fields via `get_type_hints()` because of `from __future__ import annotations`
  - `controller.py` — controller logic from plan.md (builds clean vs. attacked contexts, drives dual/tri trajectories)
  - `judge.py` — judges whether each attacked trajectory succeeded (B vs C split); emits verdicts through strict JSON schema in `llm/schemas.py`
  - `core/types.py` — shared types incl. `TrajectoryKind`
  - `agents/` — defense agent base class + LLM-backed subclass (`base.py`, `agent.py`, `llm_agent.py`, `prompts.py`). Use the open-source model served by vLLM as backend.
  - `attacks/` — attack generator (LLM-driven) + genetic algorithm (`base.py`, `genetic.py`, `llm_attacker.py`, `prompts.py`)
  - `envs/base.py` — unified `ToolEnv` interface; tool execution simulated by an LLM given tool def+inputs.
    `envs/toolsafe.py` registers the ToolSafe dataset env;
    `envs/tool_parsing.py::parse_env_info` parses free-text tool defs out of records' `env_info`.
    New dataset envs subclass `SimulatedToolEnv` and register via `envs.register_env(name,builder)` so `EnvConfig.dataset` picks them up by name.
  - `rollouts/{base,__init__,rollout}.py` — unified rollout interface (`collect_tri_rollouts`). Extend `RolloutStrategy` to add new dataset-specific strategies.
  - `process/` — processing helpers: input wrapping/preprocessing, computing injection point / turning point / Δ (`signals.py`, `edit_distance.py`) plus `dataset_builder.py` for assembling training datasets
  - `training/` — SFT & GRPO runners that shell out to vendored frameworks (`sft_runner.py`, `grpo_runner.py`, dry-run friendly). Vendored frameworks under here are external deps — never edit.
  - `utils/metrics.py` — `RoundMetrics` aggregate + termination state machine + four-agent-safety metrics (`safety_precision = clamp(1-ASR)`, `safety_recall = clean_utility_mean`, harmonic-mean F1, arithmetic-mean acc).
    `utils/plots.py` writes `curves.png` + end-of-run CSV/JSON summaries under `<exp_dir>/results/`. Pipeline also streams JSONL during runs.
  - `tests/smoke_test.py` — deterministic offline smoke test using `MockClient`. Additional tests live alongside it (`test_schemas.py`, `test_openai_degrade.py`, etc.)
- `evoguard/llm/` — pluggable LLM clients behind one interface (`LLMClient.chat(...)`):
  - `mock_client.py` offline mock routed by role markers at top of system prompts (`roles.py`)
  - `openai_client.py` OpenAI-compatible backends including vLLM endpoint
  - `qianfan_client.py` direct HTTP client for QianFan API at qianfan.baidubce.com/v2/chat/completions (appid/Bearer headers, credentials from `EVOGUARD_QIANFAN_APPID/TOKEN` env vars OR pipe-separated literal in config.api_key="<appid>|<bearer>")
  - `schemas.py` Draft 2020-12 JSON Schema constants passed via new `chat(response_format=...)` kwarg; all three call sites accept+degrade gracefully if server rejects structured output (three-state `_schema_supported ∈ {None,True,False}` cache flips False on first BadRequestError mentioning response_format/json_schema then retries once without it)
- `data/`
  - `toolsafe/` — step-level safety annotations of AgentHarm and AgentDojo tool-call results (`agentdojo-tragj/*.json`, `agentharm-traj/*.json`). Each record carries `instruction`, free-text `env_info`, `history`.
  - `agentdojo/`, `agentharm/` — vendored official benchmark repos containing the tool definitions used by envs
- `rounds/<exp_name>/round_<id>/` — all per-round artifacts persisted as JSON Lines. Each experiment dir also gets top-level `results/{safety_metrics.jsonl,safety_metrics.csv,summary.json}` and optional `curves.png`,`metrics.csv`.
- `configs/` — YAML experiment configs (`example.yaml` mock-only smoke; `real_qianfan.yaml` production-scale pop=50×max_tasks=20; `real_qianfan_smoke.yaml` small-scale live validation pop=6×max_tasks=3×max_rounds=2)
- `scripts/` — minimal bash launchers/wrappers (see below)
- `sh/qianfan_run.sh` — alternative wrapper used historically
- `deploy/{bos,docker,qianfan}/` — deployment artifacts
- `docs/` — design notes; save implementation explanations here as `.md`

## Common commands

Python interpreter for this project lives at `/ssd1/conda_envs/evoguard/bin/python` (NOT under `/root/miniforge3/envs/`). All scripts below assume you're invoking that interpreter either directly or after activating the conda env named `evoguard`.

Install runtime dependencies first time on a fresh machine::

    pip install -r requirements.txt   # lightweight pure-python extras only; heavy torch+vllm come pre-installed inside the evoguard conda env which was cloned from stabletool

### Offline tests (no GPU/network)

```bash
bash scripts/run_smoke.sh                          # full-pipeline sanity check against MockClient (~seconds)
python -m evoguard.tests.smoke_test                # equivalent direct invocation
python -m evoguard.tests.test_schemas              # JSON Schema shape validation
python -m evoguard.tests.test_openai_degrade       # MockClient degrade-path coverage w/o network
python -m evoguard.tests.test_attacker_ceiling     # GA convergence behavior assertions
python -m evoguard.tests.test_judge_temporal        # judge temporal consistency checks
```

To run any single unittest case use the standard `-k <name>` flag or pytest discovery once those adapters exist; today these scripts invoke plain module-level `main()` functions rather than pytest fixtures.

### Real-model experiments

vLLM serves the local defender model before launching anything heavy (start_qwen2.5-7b-it on GPU#6 port 8000):

```bash
bash scripts/start_vllm.sh           # writes PID -> rounds/vllm.pid ; log -> rounds/vllm.log
bash scripts/register_vllm_lora.sh   # hot-register trained LoRA adapter onto running vLLM instance
bash scripts/stop_vllm.sh            # graceful shutdown
```

QianFan remote-backend experiment launchers read secrets from `${EVOGUARD_SECRETS_FILE:-$HOME/.evoguard_qianfan.env}` exporting `EVOGUARD_QIANFAN_APPID/EVOGUARD_QIANFAN_TOKEN`, falling back to inline literals matching `docs/todo.md` so out-of-box runs still work:

```bash
scripts/run_real.sh configs/real_qianfan_smoke.yaml   # ~minutes-scale live validation pass
scripts/run_real.sh configs/real_qianfan_full.yaml      # full pop=50 × max_tasks=20 multi-hour run
```

Each invocation launches `python -m evoguard.run --config …` under nohup writing logs to `rounds/<exp>/logs/run_<ts>.log` and prints pid+tail command immediately. Legacy path still works too: `bash scripts/run_experiment.sh configs/example.yaml`.

Both real-config yamls ship with `training.dry_run=true`; flip false after Phase 5 dry-run proves clean before kicking off LoRA SFT+GRPO for real.

## Conventions

- **Design for extensibility**: rollouts and envs share a single abstract base; register new dataset envs instead of branching on names. Agents must inherit from the base class.
- **Document as you build**: save implementation notes explaining each design decision as markdown under `docs/` (e.g., see `docs/implementation_notes.md`).
- Persist every per-round artifact as JSON Lines under `rounds/<exp>/round_<id>/`.
- Keep mock-client routing driven exclusively by role markers placed atop system prompts (see `evoguard/llm/roles.py`).
- Shell scripts stay tiny wrappers around python invocations.

## Vendored Training Frameworks

`evoguard/training/AEPO/verl_aepo_entropy/` (RL stack built on verl) and `evoguard/training/LLaMA-Factory/` (SFT/LoRA) vendor two complete third-party projects. Treat them as external dependencies — do not edit their contents directly. Our integration layer wraps them purely via subprocess shells generated by `grpo_runner.py` and `sft_runner.py`, emitting yaml/hydra overrides/dataset jsonl/shell commands in a way that supports `dry_run=True` inspection before actually firing GPU jobs. Their own requirements.txt/pyproject.toml hold the heavyweight ML dependencies pulled separately at the moment you flip `dry_run=False`.

There is no top-level build/install metadata for the EvoGuard package itself yet; runtime imports rely solely on PYTHONPATH being the repo root while dev work happens inside the conda env cloned from `/ssd1/conda_envs/stabletool`.
