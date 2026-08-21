"""EvoGuard single-stage online co-evolution trainer (spec §2026-07-28).

Public surface re-exported here::

    train_online_grpo   -- entry point mirroring native_grpo_runner.train_native_grpo()
                            signature plus ``controller_factory_fn`` parameter driving
                            external G-sibling trajectory collection during GRPO rollout phase.
    make_live_judge_closure -- build judge_call closure threading live outcome verdicts
                                through compute_evoguard_reward bypassing heuristic fallback.
    OnlineGrpoOutcome -- result dataclass parallel to NativeGrpoOutcome.

Heavy ML deps stay lazily imported behind dry-run gate exactly like
``evoguard.training.native_runner`` keeping top-level import cheap & CI-safe.
"""

from __future__ import annotations

from online.trio_controller import (
    OnlineGrpoOutcome,
    make_live_judge_closure,
    train_online_grpo,
)

__all__: list[str] = [
    "train_online_grpo",
    "make_live_judge_closure",
    "OnlineGrpoOutcome",
]
