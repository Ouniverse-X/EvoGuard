# EvoGuard Writing Plan

## Target Story

EvoGuard trains tool-calling agents to make safer pre-execution decisions under evolving adversarial context while preserving benign task performance.

## Paper Sections

- Abstract
- Introduction
- Related Work
- Problem Formulation
- Method
- Experiments
- Limitations
- Conclusion

## Core Figures

- EvoGuard dynamic attack-defense loop.
- Tri-trajectory rollout diagram.
- Reward decomposition.
- Evolution-round curves: attack success, interception, task success, over-refusal.
- Attribution visualization.

## Generated Result Artifacts

Run:

```bash
python scripts/plot_results.py
```

Generated files:

- `outputs/reports/baseline_table.md`
- `outputs/reports/baseline_table.tex`
- `outputs/reports/training_curve_table.md`
- `outputs/reports/ablation_table.md`
- `outputs/reports/ablation_table.tex`
- `outputs/figures/baseline_metrics.svg`
- `outputs/figures/training_curves.svg`
- `outputs/figures/ablation_metrics.svg`
