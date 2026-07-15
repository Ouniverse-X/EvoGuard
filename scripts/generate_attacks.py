#!/usr/bin/env python
"""Generate controlled benchmark attack samples."""

from __future__ import annotations

import json
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.attacks.attack_generator import build_attack_generator
from evoguard.envs.text_tool_env import TextToolEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    env = TextToolEnv()
    generator = build_attack_generator(args.split)
    attacks = generator.generate(env.get_tasks(), round_id=args.round_id)
    rows = [asdict(attack) for attack in attacks]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    else:
        print(json.dumps(rows, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
