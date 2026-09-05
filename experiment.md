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

| round | train ASR | n_succ/555 | val ASR | val f1 | val recall | val prec | val acc | val clean_cc | train clean_cc | blocked_unfinished |
|---|---|---|---|---|---|---|---|---|---|---|
| r0 (SFT) | 0.2234 | 124 | 0.0455 | 0.9492 | 0.9545 | 0.9438 | 0.9082 | 0.50 | 0.642 | 0.157 |
| r1 (GRPO) | 0.0613 | 34 | 0.0114 | 0.9721 | 0.9886 | 0.9560 | 0.9490 | 0.55 | 0.698 | 0.285 |
| r2 | 0.0342 | 19 | 0.0114 | 0.9775 | 0.9886 | 0.9667 | 0.9592 | 0.65 | 0.741 | 0.229 |
| r3 | 0.0252 | 14 | 0.0114 | 0.9775 | 0.9886 | 0.9667 | 0.9592 | 0.60 | 0.766 | 0.119 |
| **r4** | 0.0054 | 3 | **0.0000** | 0.9888 | 1.0000 | 0.9778 | 0.9796 | 0.80 | 0.794 | 0.072 |
| r5 | 0.0018 | 1 | 0.0227 | 0.9829 | 0.9773 | 0.9885 | 0.9694 | 0.90 | 0.820 | 0.038 |
| r6 | 0.0036 | 2 | 0.0114 | 0.9943 | 0.9886 | **1.0000** | 0.9898 | **1.00** | 0.870 | 0.065 |
| **r7** | 0.0072 | 4 | **0.0000** | 0.9943 | 1.0000 | 0.9888 | 0.9898 | 0.85 | 0.878 | 0.058 |
| r8 | 0.0090 | 5 | 0.0114 | 0.9943 | 0.9886 | **1.0000** | 0.9898 | **1.00** | 0.876 | 0.068 |

Best val ASR: r4 and r7 at 0.0000. Best joint point: **r8** (or r6) — ASR 0.0114 on a
single fn, precision 1.0000 (fp=0), val clean_completion 1.00. r5's val ASR *rose* to
0.0227 (fn 0→2) while its train ASR was the run's minimum — confirms training ASR is a
bad selector.

Safety and utility move together, no overblocking trade: val clean_completion
0.50 → 1.00, val fp 5 → 0, `blocked_unfinished_rate` 0.157 → 0.068,
train `attacked_benign_preserved_rate` 0.768 → 0.924, `clean_mean_steps` 5.01 → 3.77.

Attacker fitness collapses at population 15: `mean_best_fitness`
0.481 / 0.133 / 0.093 / 0.070 / 0.020 / 0.014 / 0.009 / 0.076 / 0.043.

Reward-path judge fully live, zero fallback (3200 completions/round):
r1 `clean_served 1176 / judge:held 1189 / structural:held 544 / judge:baited 291`;
r2 `856 / 1534 / 723 / 87`. GRPO ~7300–7600 s/round (~18–19 s/prompt); ~2.5 h/round
end-to-end.

Attacker Δ evolution (immediate = Δ≤1, latent = Δ≥3):

| round | n_success | Δnorm_mean | immediate | latent | Δ range | Δ histogram | tp_source llm/scan | judge≠scan |
|---|---|---|---|---|---|---|---|---|
| r0 | 124 | 0.346 | 0.750 | 0.089 | 1–5 | 1:93 2:20 3:8 4:1 5:2 | 124 / 0 | 63/124 |
| r1 | 34 | 0.313 | 0.676 | 0.059 | 1–7 | 1:23 2:9 3:1 7:1 | 34 / 0 | 16/34 |
| r2 | 19 | 0.476 | 0.842 | 0.158 | 1–6 | 1:16 5:2 6:1 | 19 / 0 | 3/19 |
| r3 | 14 | 0.381 | 0.929 | 0.000 | 1–2 | 1:13 2:1 | 14 / 0 | 1/14 |
| r4 | 3 | 0.667 | **0.000** | **0.667** | 2–3 | 2:1 3:2 | 3 / 0 | 3/3 |
| r5 | 1 | 0.500 | 1.000 | 0.000 | 1–1 | 1:1 | 1 / 0 | 0/1 |
| r6 | 2 | 0.333 | 1.000 | 0.000 | 1–1 | 1:2 | 2 / 0 | 0/2 |
| r7 | 4 | **0.950** | **0.000** | **0.750** | 2–4 | 2:1 3:2 4:1 | 4 / 0 | 4/4 |
| r8 | 5 | 0.517 | 0.800 | 0.200 | 1–3 | 1:4 3:1 | 5 / 0 | 1/5 |

**Under pressure the attacker switches from immediate to latent.** r0–r3 are
immediate-dominated (0.75 → 0.93 immediate share) as ASR falls. Then at r4 and r7 —
the two rounds where the attacker regains ground and `elite_fitness_mean` spikes
(0.667 at r4, 0.933 at r7) — the immediate share drops to **0.000** and every success
is Δ≥2, with r7's Δnorm at 0.950, the run's maximum. Immediate injections are the class
GRPO suppresses first; what survives is the deep-latency tail. Cell counts are small
(n=1–5), so read the composition, not the rates.

**The `llm_then_scan` flip is doing real work.** Every turning point in all 206
successes was LLM-attributed — **zero scan fallbacks**. The two methods disagree on
91/206, and the disagreement is **one-directional in all 91 cases: the judge places the
turning point LATER than scan, never earlier.** On r0's disagreeing rows mean Δ is 3.84
(judge) vs 2.44 (scan); on r7's, 3.00 vs 1.00. Under `scan`, r4's and r7's latent
resurgence would have been recorded as Δ=1 immediate attacks and the finding above would
have been invisible. This is the latent-attack under-reporting the flip was meant to fix.

### Held-out test replay (r8 adapter)

`eval/vendored_replay.py` on `data/toolsafe/agentdojo-tragjnew/test`, frozen run config,
adapter `r0_sft::…::r8_grpo`. 11 tasks / 254 segments → 53 distinct injection scenarios,
64 records (53 attacked + 11 clean). 5 min wall, concurrency 6.

| tp | fn | fp | tn | acc | prec | recall | f1 | ASR | clean_cc |
|---|---|---|---|---|---|---|---|---|---|
| 52 | 1 | 3 | 8 | 0.9375 | 0.9455 | 0.9811 | 0.9630 | 0.0189 | 0.682 |

`blocked_unfinished_rate` 0.208, `attacked_benign_preserved_rate` 0.774
(0.769 on blocked rows), `clean_mean_steps` 4.18, `clean_utility_mean` 0.682.
The single success is Δ=1 (immediate).

Test is harder than val on every axis (val r8: acc 0.9898, prec 1.0000, ASR 0.0114,
clean_cc 1.00) — 11 unseen instructions, and all 3 fp plus the 0.208 unfinished rate sit
on the clean/benign side, i.e. the residual cost is overblocking on unseen tasks, not
leakage. `poison_delivered_rate` / `asr_given_delivered` are absent from
`safety_metrics.jsonl` on this path, so avoidance vs resistance is still not separable.

### Status

**Running** — launched 2026-09-05 00:54, pid 136516,
log `rounds/agentdojo_sinkdivert_v11/logs/run_20260905_005403.log`,
artifacts `rounds/evoguard_agentdojo_sinkdivert_v11/`.
As of 23:35 (22.7 h in): r0–r8 trained and validated, r9 rollouts running.
~2.5 h/round → 6 rounds left.

r0 startup confirmed: 58 tasks (37/10), `total=32 entries`, `injected 32 skeletons`
on every task, and a `[harmful_goal]` line per task with `in_benign_plan=False` for
all of them — zero `exposes no sensitive sink` warnings. Per round: `progress judge
active at http://127.0.0.1:8004/v1 (model=qwen3.5-9b)`, GDPO fired on all 400
generation batches, Δ shaping enabled at λ=1.5, LoRA hot-loaded onto :8000 and stacked
(`r0_sft::r1_grpo::…::r8_grpo`). `consecutive_low_asr_streak` stays 0 and
`terminated=False` in every round — early stopping is off as intended.

Two non-fatal issues in the log:

* `[native-gpu-wait] GPU idx=6 only N MiB free, need >= 73728 MiB` — the trainer's
  pre-flight assumes an 80 GB card and these are 40 GB, so it reserves in 1 GiB blocks
  to push out a foreign oversubscription filler. It resolves itself but cost ~30 min
  before r1 (03:18 start → 03:48 model load). Fires when a foreign process is on GPU 6.
* `[train] results/saves/ convenience hook failed: module 'os' has no attribute
  'islink'` × 9 — real typo (`os.islink` should be `os.path.islink`), affects only the
  convenience symlink under `results/saves/`. No effect on training or metrics.

Otherwise only the pre-existing QianFan `json_schema` degrade, 2 transient
`Request timed out` retries, and `capped away N candidates (>=400 cap)` — the row
budget binding, as designed.

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
