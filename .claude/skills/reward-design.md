---
name: reward-design
description: Design and debug EvoGuard's task, safety, attribution, and KL-regularized RL rewards.
---

# Purpose

Design reward functions that improve safety without causing over-refusal.

# Workflow

1. Read current reward code and `docs/method_design.md`.
2. Decompose reward into:
   - Task completion reward.
   - Safety judgment reward.
   - Attribution reward.
   - KL penalty.
   - Over-refusal penalty if needed.
3. Check edge cases:
   - Benign tool call refused.
   - Dangerous tool call executed.
   - Dangerous tool call asks confirmation.
   - No-tool task unnecessarily uses tool.
   - High-confidence wrong decision.
4. Suggest formulas.
5. Suggest default weights.
6. Suggest logging metrics.
7. Output implementation plan and tests.
