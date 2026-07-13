#!/usr/bin/env python
"""Train the local neural safety head and evaluate on held-out attacks."""

from __future__ import annotations

import json
import sys
import argparse
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.agents.neural_safety_head import NeuralSafetyConfig
from evoguard.training.neural_trainer import train_neural_safety_head


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local neural EvoGuard safety head.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-path", default="outputs/checkpoints/neural_safety_head.pt")
    args = parser.parse_args()
    config = NeuralSafetyConfig(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    result = train_neural_safety_head(
        rounds=args.rounds,
        config=config,
        checkpoint_path=args.checkpoint_path,
    )
    print(
        json.dumps(
            {
                "num_records": len(result.records),
                "train_stats": result.train_stats,
                "eval_metrics": result.eval_metrics,
                "checkpoint_path": str(result.checkpoint_path),
                "config": asdict(config),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
