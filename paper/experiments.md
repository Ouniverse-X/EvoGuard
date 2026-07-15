# Experiments Draft

See `docs/experiments.md` for the active experiment plan.

## Modern Automated Jailbreak Evaluation

Add a held-out evaluation column for PAIR/TAP/GPTFuzzer-inspired attacks. The experiment should compare no guard, static guard, EAPO-style fixed contrastive rollout, and EvoGuard adaptive training on:

- standard held-out templates,
- hard held-out templates,
- automated-jailbreak split.

Report attack interception, attack success, false negatives, task success, over-refusal, and attribution IoU. Keep this split separate from default training unless the experiment is explicitly labeled as robustness training on modern automated jailbreak styles.

In addition to the deterministic `automated_jailbreak` split, run an API-backed online adaptive red-team setting. In this setting, the attacker model generates candidates, EvoGuard immediately evaluates each candidate against the selected Defense Agent, successful attacks are retained as attack memory, blocked attacks are fed back as negative examples, and later generations condition on both groups. This is the closer analogue to PAIR/TAP/GPTFuzzer-style black-box red teaming.
