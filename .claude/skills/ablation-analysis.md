---
name: ablation-analysis
description: Plan and analyze ablation experiments for EvoGuard.
---

# Purpose

Understand which components of EvoGuard contribute to performance.

# Workflow

Analyze these ablations:

1. Without adaptive attack generation.
2. Without attack round weighting.
3. Without confidence-aware penalty.
4. Without attribution reward.
5. Without no-tool trajectory.
6. Without clean-tool trajectory.
7. Without KL regularization.
8. Binary action instead of three-state action.
9. Static attack pool instead of evolving attack pool.

Output:
- Ablation table.
- Expected outcomes.
- Interpretation guide.
