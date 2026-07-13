#!/usr/bin/env python
"""Generate controlled benchmark attack samples."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.attacks.attack_generator import build_attack_generator
from evoguard.envs.text_tool_env import TextToolEnv


def main() -> None:
    env = TextToolEnv()
    generator = build_attack_generator("train")
    attacks = generator.generate(env.get_tasks(), round_id=0)
    print(json.dumps([asdict(attack) for attack in attacks], indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
