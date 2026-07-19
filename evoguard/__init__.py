"""EvoGuard: co-evolutionary attack/defense for agent tool-call safety.

The package implements the research design in ``docs/plan.md``: a controller
produces clean (A) and attacked (B/C) tool-calling trajectories for the same
task, a :mod:`evoguard.process` module derives the injection point, behavior
turning point and their difference ``delta``, an evolutionary attacker maximizes
``delta`` on successful attacks, and a defense agent is trained (SFT + GRPO on
LoRA) to minimize it until attacks no longer succeed.
"""

from evoguard.config import ExperimentConfig

__version__ = "0.1.0"

__all__ = ["ExperimentConfig", "__version__"]
