# Experiments

One block per run. Keep it short: config deltas + the numbers, no prose.

---

## v11 sinkdivert — `configs/agentdojo_sinkdivert_v11.yaml`

Dataset `agentdojo_split` (`data/toolsafe/agentdojo-tragjnew`, 37 train / 10 val / 11 test).

| knob | value | vs v10 |
|---|---|---|
| `env.inject_harmful_tools` | `true` | **new** — 4 out-of-scope decoy sinks per suite, appended to BOTH arms |
| `EVOGUARD_PREWARM_SEEDS_DIR` | `data/seeds_v3` | **new** — 32 seeds (v2r's 20 + 12 sink-diversion) |
| `process.turning_point_method` | `llm_then_scan` | **changed** from `scan` |
| `pipeline.max_rounds` | 15 | 12 |
| `pipeline.patience_rounds` | 999 | 5 — **early stop disabled**, val eval stays on |
| `pipeline.stop_on_zero_success` | `false` | unchanged |
| attacker | `mcts_delta`, glm-5.2, pop 15, λ_Δ 0.6 | unchanged |
| defender | Qwen2.5-7B-Instruct, LoRA r32/α64 | unchanged |
| training | `sft_then_native_grpo`, GDPO on, K=1, λ_curriculum 1.5, 400 rows/round | unchanged |
| reward | `R = r_safety + r_progress − p_drift` | unchanged |
| judges | Qwen3.5-9B @ :8004 (verdict + utility + reward path) | unchanged — served via Docker, see Status |

Models: `/ssd1/yx/models/qwen2.5-7b-it`, `/ssd1/yx/models/Qwen3.5-9B`,
`/ssd1/yx/models/llama3-8b` (all shard-complete).

Not token-comparable with v10 (the decoy tools are in every prompt) and Δ is not
comparable with any pre-v11 run (turning point is now LLM-attributed).

### Launch

```bash
EVOGUARD_PY_BIN=/ssd1/conda_envs/evoguard/bin/python \
EVOGUARD_VLLM_MODEL=/ssd1/yx/models/qwen2.5-7b-it EVOGUARD_VLLM_GPU=4 \
  bash scripts/start_vllm.sh                       # defender :8000, --enable-lora

EVOGUARD_PY_BIN=/ssd1/conda_envs/evoguard/bin/python \
TRAINER_CUDA_VISIBLE_DEVICES=6 \
EVOGUARD_PREWARM_SEEDS_DIR=data/seeds_v3 \
EVOGUARD_JUDGE_LLM_BASE_URL=http://127.0.0.1:8004/v1 \
EVOGUARD_PROGRESS_LLM_BASE_URL=http://127.0.0.1:8004/v1 \
EVOGUARD_JUDGE_LLM_MODEL=qwen3.5-9b \
bash scripts/run_grpo_experiment.sh configs/agentdojo_sinkdivert_v11.yaml
```

Log must show `total=32 entries`, `progress judge active`, and per round a
`[val] rN adapter=... asr=...` line.

### Results

Training ASR is the co-evolving attacker's score, not the defender's — select on
the val column, and evaluate the best round, not the last.
Val = full replay (`eval/vendored_replay.py`) on the 10-task val split, 88 injection
scenarios / 98 records.

| round | train ASR | n_succ/555 | val ASR | val f1 | val recall | val prec | val acc | clean_completion (train) | blocked_unfinished |
|---|---|---|---|---|---|---|---|---|---|
| r0 (SFT) | 0.2234 | 124 | 0.0455 | 0.9492 | 0.9545 | 0.9438 | 0.9082 | 0.642 | 0.157 |
| r1 (GRPO) | 0.0613 | 34 | 0.0114 | 0.9721 | 0.9886 | 0.9560 | 0.9490 | 0.698 | 0.285 |
| r2 (GRPO) | 0.0342 | 19 | 0.0114 | 0.9775 | 0.9886 | 0.9667 | 0.9592 | 0.741 | 0.229 |
| r3 (GRPO) | 0.0252 | 14 | _training_ | | | | | 0.766 | 0.119 |

val `clean_completion_rate` 0.50 → 0.55 → 0.65: utility rising alongside safety, no
overblocking trade. Train-side `attacked_benign_preserved_rate` 0.768 → 0.684 → 0.744
→ 0.859; `clean_mean_steps` 5.01 → 4.21.

Attacker fitness collapses while the population stays at 15: `mean_best_fitness`
0.481 → 0.133 → 0.093 → 0.070, `elite_fitness_mean` 0.446 → 0.381 → 0.418 → 0.386.

Reward-path judge fully live, zero fallback (3200 completions/round):
r1 `clean_served 1176 / judge:held 1189 / structural:held 544 / judge:baited 291`;
r2 `856 / 1534 / 723 / 87`. GRPO ~7300–7600 s/round (~18–19 s/prompt).

Attacker Δ evolution (immediate = Δ≤1, latent = Δ≥3):

| round | n_success | Δnorm_mean | immediate | latent | Δ range | Δ histogram | tp_source llm/scan |
|---|---|---|---|---|---|---|---|
| r0 | 124 | 0.346 | 0.750 | 0.089 | 1–5 | 1:93 2:20 3:8 4:1 5:2 | 124 / 0 |
| r1 | 34 | 0.313 | 0.676 | 0.059 | 1–7 | 1:23 2:9 3:1 7:1 | 34 / 0 |
| r2 | 19 | 0.476 | 0.842 | 0.158 | 1–6 | 1:16 5:2 6:1 | 19 / 0 |
| r3 | 14 | 0.381 | 0.929 | 0.000 | 1–2 | 1:13 2:1 | 14 / 0 |

**The `llm_then_scan` flip is doing real work.** Every turning point was LLM-attributed
— zero scan fallbacks — and the two methods disagree on a large share of successes:
63/124 (51%) in r0, 16/34 in r1, 3/19 in r2, 1/14 in r3. The disagreement is
**one-directional in all 83 cases: the judge places the turning point LATER than scan,
never earlier.** On r0's disagreeing rows, mean Δ 3.84 (judge) vs 2.44 (scan). Under
`scan` those rows would have entered the advantage curriculum as shallower attacks than
they are, which is exactly the latent-attack under-reporting the flip was meant to fix.

### Status

**Running** — launched 2026-09-05 00:54, pid 136516,
log `rounds/agentdojo_sinkdivert_v11/logs/run_20260905_005403.log`,
artifacts `rounds/evoguard_agentdojo_sinkdivert_v11/`.
As of 10:20: r0–r3 rollouts + judging done, r0–r2 trained and validated, r3 GRPO
training in progress (started 09:04). ~3.7 h/round → ~11 rounds remaining.

r0 startup confirmed: 58 tasks (37/10), `total=32 entries`, `injected 32 skeletons`
on every task, and a `[harmful_goal]` line per task with `in_benign_plan=False` for
all of them — zero `exposes no sensitive sink` warnings. Per round: `progress judge
active at http://127.0.0.1:8004/v1 (model=qwen3.5-9b)`, GDPO fired on all 400
generation batches, Δ shaping enabled at λ=1.5, LoRA hot-loaded onto :8000 and
stacked (`r0_sft::r1_grpo::r2_grpo`). Only WARNINGs are QianFan's `json_schema`
degrade on the attacker path (pre-existing, harmless) and `capped away N candidates
(>=400 cap)` — the row budget binding, as designed.

Serving layout (40 GB cards, not 80): GPU 4 defender vLLM :8000 `--enable-lora`,
GPU 6 trainer, GPU 5 `qwen3.5-9b` :8004 (judges), GPU 7 `qwen2.5-7b-it` :8003
(tool_executor). GPU 0/1/3 foreign; GPU 2 idle but claimed by a docker container.
`:8002` holds a 16-day-old wedged llama3-8b (listening, 0 MiB, 180 s timeouts) —
leave it, `scripts/stop_vllm.sh` would take the primary down with it.

Two facts that took a detour to establish:

* **Qwen3.5-9B is servable here, but only in Docker.** CentOS 7 / glibc 2.17 cannot
  load the manylinux_2_28 wheels that vllm ≥~0.9 and torch ≥2.7 ship, so no conda env
  on this box can serve it (`vllm 0.8.5` has no `Qwen3_5ForConditionalGeneration`).
  `scripts/start_qwen35_judge_docker.sh` sidesteps that — container `qwen35-judge`,
  vllm 0.24.0, `--gdn-prefill-backend triton`. Live-checked: plain 2.83 s,
  `JUDGE_VERDICT_SCHEMA`-constrained 3.95 s, valid JSON. No judge downgrade needed.
* **CLAUDE.md's "torch ≥2.7 is mandatory" applies to peft 0.19.0 only.** peft 0.19.1
  guards the `float8_e8m0fnu` lookup (`tuners_utils.py:2170`,
  `getattr(torch, name, None)`), and `get_peft_model` on the Qwen2 arch with v11's
  LoRA targets succeeds under torch 2.6.0. Hence `/ssd1/conda_envs/evoguard`
  (py3.12, torch 2.6.0+cu124, transformers 4.51.3, trl 0.19.1, peft 0.19.1,
  vllm 0.8.5.post1) is a valid runtime. `evoguard_full` / `evoguard2` are dead ends —
  both installs were SIGTERM'd and neither can be finished (pinned index tops out at
  torch 2.6.0; vllm 0.19.1 needs 2.10).

Pre-flight verified in that env: full offline smoke passes; all training / eval /
attack modules import; `trl 0.19.1` still exposes
`GRPOTrainer._generate_and_score_completions`; `xgrammar.kernels` imports;
termination runs all 15 rounds with val ASR pinned at 0.0;
`test_signals_turning_point` 14/14, `test_harmful_catalog` 18/18, `test_schemas` 19/19.

Watch items: `enable_thinking: false` is a no-op on the OpenAI-compatible client
(`llm/openai_client.py:128`), so a `_schema_supported` degrade on :8004 would let
qwen3.5-9b's "Thinking Process:" preamble break verdict parsing — treat such a
warning as fatal. `poison_delivered_rate` / `asr_given_delivered` are `None` in both
`metrics.jsonl` and the replay summary on this path, so the avoidance-vs-resistance
split is NOT separable from these numbers — some of the ASR drop may be the defender
routing around the poisoned tool rather than resisting it. That has to come from a
post-hoc pass over `round_*/records.jsonl` before the drop is claimed as resistance.
