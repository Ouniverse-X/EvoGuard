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

Text-tool / AI2-THOR / ManiSkill

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
