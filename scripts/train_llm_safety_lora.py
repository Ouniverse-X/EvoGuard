#!/usr/bin/env python
"""Train a LoRA safety-judge LLM from EvoGuard SFT JSONL data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.training.llm_sft import parse_args, train_lora_sft


def main() -> None:
    metrics = train_lora_sft(parse_args())
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
