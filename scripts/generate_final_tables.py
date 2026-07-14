#!/usr/bin/env python3
"""Generate Markdown experiment summary tables from EvoGuard evaluation logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUTPUT_PATH = Path("docs/experiments.md")

LOGS = {
    "Base Qwen": {
        "llm_r1": Path("outputs/logs/eval_base_qwen_llm_r1.json"),
        "hard": Path("outputs/logs/eval_base_qwen_hard_heldout.json"),
    },
    "Qwen3Guard": {
        "toolsafe": Path("outputs/logs/baseline_qwen3guard.json"),
    },
    "ToolSafe SFT": {
        "toolsafe": Path("outputs/logs/eval_sft_toolsafe.json"),
    },
    "Filtered RL-v1": {
        "toolsafe": Path("outputs/logs/eval_rl_v1_filtered.json"),
    },
    "Mixed RL-v1": {
        "toolsafe": Path("outputs/logs/eval_rl_mixed_v2.json"),
    },
}

OPTIONAL_LOGS = {
    "Self-RedTeam": {
        "toolsafe": Path("outputs/logs/baseline_self_redteam.json"),
        "llm_r1": Path("outputs/logs/baseline_self_redteam.json"),
    }
}

GENERALIZATION_COLUMNS = [
    ("ToolSafe held-out harmful", "toolsafe"),
    ("LLM-r1", "llm_r1"),
    ("Hard held-out", "hard"),
]

UTILITY_COLUMNS = [
    "valid_json_rate",
    "attack_interception_rate",
    "over_refusal_rate",
    "task_success_rate",
    "attribution_accuracy",
]


def main() -> None:
    logs = {**LOGS}
    for name, mapping in OPTIONAL_LOGS.items():
        if any(path.exists() for path in mapping.values()):
            logs[name] = mapping

    loaded = load_all(logs)
    markdown = "\n".join(
        [
            "<!-- BEGIN GENERATED FINAL TABLES -->",
            "",
            "## Final Experiment Tables",
            "",
            render_generalization_table(loaded),
            "",
            render_utility_table(loaded),
            "",
            "<!-- END GENERATED FINAL TABLES -->",
            "",
        ]
    )
    write_or_replace(OUTPUT_PATH, markdown)
    print(markdown)


def load_all(logs: dict[str, dict[str, Path]]) -> dict[str, dict[str, dict[str, Any]]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for defender, columns in logs.items():
        loaded[defender] = {}
        for key, path in columns.items():
            payload = load_json(path)
            if payload is not None:
                loaded[defender][key] = select_dataset(payload, key)
    return loaded


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"warning: missing log file: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"warning: invalid JSON in {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        print(f"warning: expected object in {path}")
        return None
    return payload


def select_dataset(payload: dict[str, Any], key: str) -> dict[str, Any]:
    datasets = payload.get("datasets")
    if isinstance(datasets, dict) and isinstance(datasets.get(key), dict):
        return datasets[key]
    return payload


def render_generalization_table(loaded: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = ["### Defender generalization evaluation (ASR↓ %)"]
    header = ["Defender", *[name for name, _ in GENERALIZATION_COLUMNS]]
    lines.append(markdown_row(header))
    lines.append(markdown_separator(len(header)))
    for defender, results in loaded.items():
        row = [defender]
        for _, key in GENERALIZATION_COLUMNS:
            row.append(format_percent(metric(results.get(key), "attack_success_rate")))
        lines.append(markdown_row(row))
    return "\n".join(lines)


def render_utility_table(loaded: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = ["### Utility & safety metrics (ToolSafe held-out)"]
    header = ["Defender", *UTILITY_COLUMNS]
    lines.append(markdown_row(header))
    lines.append(markdown_separator(len(header)))
    for defender, results in loaded.items():
        payload = results.get("toolsafe")
        row = [defender]
        for key in UTILITY_COLUMNS:
            value = payload.get(key) if key == "valid_json_rate" and payload else metric(payload, key)
            row.append(format_float(value))
        lines.append(markdown_row(row))
    return "\n".join(lines)


def metric(payload: dict[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else None


def format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}"


def format_float(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def markdown_separator(width: int) -> str:
    return "| " + " | ".join(["---"] * width) + " |"


def write_or_replace(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(markdown, encoding="utf-8")
        return
    current = path.read_text(encoding="utf-8")
    start = "<!-- BEGIN GENERATED FINAL TABLES -->"
    end = "<!-- END GENERATED FINAL TABLES -->"
    start_index = current.find(start)
    end_index = current.find(end)
    if start_index >= 0 and end_index >= start_index:
        end_index += len(end)
        updated = current[:start_index].rstrip() + "\n\n" + markdown + current[end_index:].lstrip()
    else:
        updated = current.rstrip() + "\n\n" + markdown
    path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
