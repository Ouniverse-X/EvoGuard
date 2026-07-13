"""Lightweight PPO trainer for EvoGuard's tabular/linear safety head.

This trainer reuses the round-based EvoGuard loop and delegates the clipped
policy-gradient update to `TrainableSafetyHead.update`. It is intentionally
dependency-free so the current text-tool MVP can exercise PPO-style updates
without requiring TRL, Transformers, or a neural language-model policy.
"""

from evoguard.training.trainer import EvoGuardTrainer


class PPOTrainer(EvoGuardTrainer):
    """Round-based trainer using the lightweight PPO safety-head backend."""
