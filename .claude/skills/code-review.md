---
name: code-review
description: Review EvoGuard code for modularity, reproducibility, and experimental correctness.
---

# Purpose

Review code changes before experiments.

# Workflow

1. Check whether modules follow project structure.
2. Check whether configs are separated from code.
3. Check whether random seeds are controlled.
4. Check whether logs include all required metrics.
5. Check whether train/eval attack pools are separated.
6. Check whether baselines use comparable settings.
7. Output:
   - Bugs
   - Reproducibility risks
   - Suggested fixes
