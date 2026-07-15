# Experiments Draft

See `docs/experiments.md` for the active experiment plan.

## Modern Automated Jailbreak Evaluation

Add a held-out evaluation column for PAIR/TAP/GPTFuzzer-inspired attacks. The experiment should compare no guard, static guard, EAPO-style fixed contrastive rollout, and EvoGuard adaptive training on:

- standard held-out templates,
- hard held-out templates,
- automated-jailbreak split.

Report attack interception, attack success, false negatives, task success, over-refusal, and attribution IoU. Keep this split separate from default training unless the experiment is explicitly labeled as robustness training on modern automated jailbreak styles.
