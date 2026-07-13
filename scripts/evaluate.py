#!/usr/bin/env python
"""Evaluate the MVP defender."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.attacks.attack_generator import build_attack_generator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.evaluation.evaluator import Evaluator


def main() -> None:
    evaluator = Evaluator(TextToolEnv(), build_attack_generator("heldout"))
    metrics = evaluator.evaluate(DefenseAgent(), round_id=0)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
