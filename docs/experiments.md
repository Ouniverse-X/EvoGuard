# EvoGuard Experiments

## Experiment Template

### Experiment ID

E001

### Date

YYYY-MM-DD

### Goal

What question does this experiment answer?

### Hypothesis

What do we expect?

### Environment

Text-tool / ToolSafe / TraceSafe

### Model

Base model:

Defense Agent checkpoint:

Attack generator:

### Training Setup

Rounds:

Batch size:

RL algorithm:

Reward weights:

KL coefficient:

### Attack Setup

Attack source:

Attack generation round:

Filtering criterion:

### Metrics

- Task success rate:
- Attack interception rate:
- Over-refusal rate:
- Attribution accuracy:
- Attack success rate:
- False positive rate:
- False negative rate:

### Results

### Observations

### Failure Cases

### Next Steps

## Planned Main Experiments

| Claim | Experiment | Baselines | Metrics | Expected Result |
| --- | --- | --- | --- | --- |
| EvoGuard handles evolving attacks better than static training. | Round-based adaptive attack comparison. | Static guard, ToolSafe-style guard, no guard. | Attack interception, false negative rate, held-out robustness. | EvoGuard improves later-round robustness. |
| Tri-trajectory rollout preserves task ability while improving safety. | Remove no-tool and clean-tool trajectories. | Attacked-only, no no-tool, no clean-tool. | Task success, over-refusal, attack interception. | Full tri-rollout balances all three metrics. |
| Dynamic attack generation is more useful than fixed attack pools. | Adaptive generator vs static pool. | Static attack pool, random mutation. | Later-round attack success, defense curve. | Adaptive attacks expose current weaknesses. |
| Risk-aware rewards reduce brittle safety behavior. | Remove confidence, round, stale penalties. | Binary reward, no confidence weighting, no round weighting. | High-confidence unsafe execution, stability, over-refusal. | Full reward penalizes severe errors better. |
| Attribution improves interpretability and safety generalization. | Remove attribution reward. | No attribution, attribution-only detector. | Attribution IoU, attack interception, unseen robustness. | Attribution improves localization and may improve robustness. |

## E001: Text-Tool MVP Trainable Safety Head Smoke Run

### Date

2026-07-12

### Goal

Verify that the round-based training loop can change defense behavior using generated attacks, tri-trajectory records, rewards, and a lightweight trainable safety head.

### Environment

Text-tool.

### Model

Defense Agent checkpoint:
Dependency-free lightweight PPO linear safety head.

Attack generator:
Prompt-template generator with controlled benchmark injections.

Evaluation attack pool:
Held-out templates, separate from training templates.

### Training Setup

Rounds:
3

RL algorithm:
Lightweight clipped PPO over the linear safety head.

Reward weights:
`lambda_safety=1.0`, `lambda_attr=0.25`, `beta_kl=0.0`.

### Metrics

- Task success rate.
- Attack interception rate.
- Over-refusal rate.
- Attribution accuracy.
- Attack success rate.
- False positive rate.
- False negative rate.

### Results

Run with:

```bash
python scripts/train_evoguard.py
```

The script writes per-round summaries to `outputs/logs/train_evoguard_summary.jsonl`.

Generate the paper table and training curve with:

```bash
python scripts/plot_results.py
```

### Observations

The first round starts vulnerable because the trainable head has not yet adapted to the reward signal. Later rounds verify that the PPO update path can change the safety policy from generated attack trajectories.

Held-out attack evaluation is intentionally stricter than train-pool evaluation. Low held-out interception in early MVP runs should be interpreted as a generalization gap, not as a runner failure.

### Next Steps

Replace the linear PPO MVP backend with a neural PPO or GRPO policy while preserving the existing rollout, reward, and logging interfaces.

## E002: Text-Tool MVP Baseline Comparison

### Date

2026-07-12

### Goal

Verify that the MVP can produce comparable metrics for safety baselines and EvoGuard adaptive training.

### Environment

Text-tool.

### Baselines

- No guard.
- Rule-based guard.
- Always-refuse guard.
- Static guard.
- EAPO fixed contrastive rollout.
- EvoGuard adaptive.

### Metrics

- Task success rate.
- Attack interception rate.
- Over-refusal rate.
- False negative rate.
- False positive rate.
- Attribution accuracy.
- Attack success rate.

### Results

Run with:

```bash
python scripts/run_baselines.py
```

The script trains trainable baselines on the train attack pool and evaluates every baseline on the held-out attack pool. It records both `train_attack_split` and `eval_attack_split` in each row and writes rows to `outputs/logs/baseline_comparison.jsonl`.

Generate the baseline markdown table, LaTeX table, and SVG bar chart with:

```bash
python scripts/plot_results.py
```

### Next Steps

Replace MVP approximations with paper-faithful ToolSafe, MOSAIC, MAGIC, RUBAS, and AgentWatcher implementations after related-work notes are finalized.

## E003: Text-Tool MVP Ablation Suite

### Date

2026-07-12

### Goal

Measure which EvoGuard components affect held-out attack robustness, task success, and over-refusal.

### Ablations

- Full EvoGuard.
- No adaptive attack generation.
- No attack round weighting.
- No confidence weighting.
- No stale-error penalty.
- No attribution reward.
- No no-tool trajectory.
- No clean-tool trajectory.
- Static attack pool.
- Binary action-space proxy.

### Results

Run with:

```bash
python scripts/run_ablations.py
python scripts/plot_results.py
```

Generated outputs:

- `outputs/logs/ablation_results.jsonl`
- `outputs/reports/ablation_table.md`
- `outputs/reports/ablation_table.tex`
- `outputs/figures/ablation_metrics.svg`

## E004: Expanded Text-Tool Difficulty

### Date

2026-07-13

### Goal

Increase the benchmark signal beyond the initial toy setting while preserving existing training, evaluation, baseline, and reporting interfaces.

### Changes

- Expanded tool set from basic text tools to include CRM lookup and support-ticket update.
- Expanded benign and risky task families.
- Added explicit tool risk types.
- Expanded train attack pool to six controlled styles.
- Expanded held-out evaluation attack pool to six controlled styles.

### Verification

The environment test checks that the task set has at least 12 tasks, includes CRM and ticket tools, covers at least six risk types, and all preferred tool calls execute inside the simulator.

## E005: ToolSafe SFT vs Qwen3Guard Baseline

### Date

2026-07-13

### Goal

Compare the ToolSafe-trained EvoGuard SFT adapter against the public Qwen3Guard-Gen-8B baseline on the same ToolSafe held-out rollouts.

### Environment

ToolSafe AgentHarm trajectory adapter.

Held-out file:

```bash
data/eval/toolsafe_heldout_tri_rollouts.jsonl
```

### Models

- ToolSafe SFT adapter: `outputs/checkpoints/llm_safety_lora_qwen15b_toolsafe_sft`
- Qwen3Guard baseline: `Qwen/Qwen3Guard-Gen-8B`

Qwen3Guard uses its native generated label format, parsed as:

- `Safety: Safe` -> `execute`
- `Safety: Controversial` -> `ask_confirmation`
- `Safety: Unsafe` -> `refuse`

### Commands

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=3 \
python scripts/evaluate_llm_safety.py \
  --model /mnt/sata1/beihang_toolsafe/models/Qwen2.5-1.5B-Instruct \
  --lora outputs/checkpoints/llm_safety_lora_qwen15b_toolsafe_sft \
  --rollouts-jsonl data/eval/toolsafe_heldout_tri_rollouts.jsonl \
  --output outputs/logs/eval_sft_toolsafe.json
```

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://huggingface.co \
python scripts/evaluate_llm_safety.py \
  --model Qwen/Qwen3Guard-Gen-8B \
  --lora "" \
  --parser qwen3guard \
  --rollouts-jsonl data/eval/toolsafe_heldout_tri_rollouts.jsonl \
  --output outputs/logs/baseline_qwen3guard.json \
  --max-new-tokens 128 \
  --min-valid-json-rate 0.0
```

### Results

| Model | Parse/valid rate | Attack interception | Over-refusal | Task success | FNR | FPR | Attack success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ToolSafe SFT adapter | 100.00% | 100.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% |
| Qwen3Guard-Gen-8B | 100.00% | 100.00% | 12.20% | 96.58% | 0.00% | 12.20% | 0.00% |

### Observations

Both models intercepted all harmful ToolSafe held-out attacked-tool trajectories. The main gap is usability on benign clean-tool trajectories: Qwen3Guard-Gen-8B over-refused 12.20% of benign tool calls, while the ToolSafe SFT adapter had 0.00% over-refusal under the EvoGuard action mapping.

### Artifacts

- `outputs/logs/eval_sft_toolsafe.json`
- `outputs/logs/baseline_qwen3guard.json`

## E006: Defender Generalization and Self-RedTeam Baselines

### Date

2026-07-14

### Goal

Assemble the current defender generalization table across base Qwen, Qwen3Guard, ToolSafe SFT, filtered RL, mixed RL, and Self-RedTeam baselines.

### Evaluation Sets

- ToolSafe held-out harmful: `data/eval/toolsafe_heldout_tri_rollouts.jsonl`
- LLM-r1 generated attacks: `data/eval/llm_generated_round1_attacked_rollouts.jsonl`
- Hard held-out: `data/eval/hard_heldout_tri_rollouts.jsonl`

### Artifacts

- Base Qwen LLM-r1: `outputs/logs/eval_base_qwen_llm_r1.json`
- Base Qwen hard held-out: `outputs/logs/eval_base_qwen_hard_heldout.json`
- Qwen3Guard: `outputs/logs/baseline_qwen3guard.json`
- ToolSafe SFT: `outputs/logs/eval_sft_toolsafe.json`
- Filtered RL-v1: `outputs/logs/eval_rl_v1_filtered.json`
- Mixed RL-v2 eval: `outputs/logs/eval_rl_mixed_v2.json`
- Self-RedTeam: `outputs/logs/baseline_self_redteam.json`

### Results Summary

Base Qwen without an adapter is not a reliable defender under the EvoGuard JSON interface. It has ASR 9.09% on LLM-r1 and 24.04% on hard held-out, with low valid JSON rates of 9.09% and 24.04%, respectively.

ToolSafe SFT is the strongest clean ToolSafe held-out result so far: ASR 0.00%, attack interception 100.00%, over-refusal 0.00%, task success 100.00%, and attribution accuracy 97.86%.

Filtered RL-v1 preserves ASR 0.00% and over-refusal 0.00%, but it hurts utility and localization: task success drops to 90.57% and attribution accuracy drops to 50.24%. This matches the diagnosis that attacked-only RL over-specializes to refusal.

Mixed RL-v2 recovers the ToolSafe SFT behavior on the held-out ToolSafe split: ASR 0.00%, attack interception 100.00%, over-refusal 0.00%, task success 100.00%, attribution accuracy 97.86%, and valid JSON 99.05%. This supports using mixed clean/attacked rollouts instead of filtered attacked-only RL.

Self-RedTeam with base Qwen as both attacker and defender is weak in this setup: ToolSafe held-out ASR is 22.86%, LLM-r1 ASR is 19.09%, valid JSON is 25.71%, over-refusal is 21.95%, and task success is 77.40%. The likely bottleneck is that the base model does not reliably follow the safety-decision JSON schema without SFT.

### Current Conclusion

The main table should emphasize that adapter training, especially ToolSafe SFT and mixed RL, is necessary for a usable defender. Self-RedTeam is useful as a baseline but not competitive without schema-following supervision. Filtered RL-v1 is an informative negative result: it improves neither ASR nor utility over SFT and degrades task success and attribution.

<!-- BEGIN GENERATED FINAL TABLES -->

## Final Experiment Tables

### Defender generalization evaluation (ASR↓ %)
| Defender | ToolSafe held-out harmful | LLM-r1 | Hard held-out |
| --- | --- | --- | --- |
| Base Qwen | - | 9.09 | 24.04 |
| Qwen3Guard | 0.00 | - | - |
| ToolSafe SFT | 0.00 | - | - |
| Filtered RL-v1 | 0.00 | - | - |
| Mixed RL-v1 | 0.00 | - | - |
| Self-RedTeam | 22.86 | 19.09 | - |

### Utility & safety metrics (ToolSafe held-out)
| Defender | valid_json_rate | attack_interception_rate | over_refusal_rate | task_success_rate | attribution_accuracy |
| --- | --- | --- | --- | --- | --- |
| Base Qwen | - | - | - | - | - |
| Qwen3Guard | 1.0000 | 1.0000 | 0.1220 | 0.9658 | 0.0000 |
| ToolSafe SFT | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.9786 |
| Filtered RL-v1 | 1.0000 | 1.0000 | 0.0000 | 0.9057 | 0.5024 |
| Mixed RL-v1 | 0.9905 | 1.0000 | 0.0000 | 1.0000 | 0.9786 |
| Self-RedTeam | 0.2571 | 0.7714 | 0.2195 | 0.7740 | 0.0136 |

<!-- END GENERATED FINAL TABLES -->
