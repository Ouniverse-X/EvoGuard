#!/usr/bin/env python3
"""Download the gated CyCraftAI/TraceSafe dataset after Hugging Face approval."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TraceSafe-Bench from Hugging Face.")
    parser.add_argument("--repo-id", default="CyCraftAI/TraceSafe")
    parser.add_argument("--output-dir", default="data/raw/tracesafe_bench")
    parser.add_argument("--token", help="Optional Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        token=args.token,
        allow_patterns=["*.jsonl", "README.md", ".gitattributes"],
    )
    print(path)


if __name__ == "__main__":
    main()
