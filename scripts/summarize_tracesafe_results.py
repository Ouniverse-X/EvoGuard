#!/usr/bin/env python3
"""Summarize TraceSafe-Bench evaluation logs into overall and category tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_LOGS = {
    "SFT": "outputs/logs/eval_sft_tracesafe.json",
    "Mixed RL": "outputs/logs/eval_mixedrl_tracesafe.json",
    "Qwen3Guard": "outputs/logs/baseline_qwen3guard_tracesafe.json",
    "TS-Guard": "outputs/logs/baseline_ts_guard_tracesafe.json",
}

CATEGORY_ORDER = [
    "PromptInjectionIn",
    "PromptInjectionOut",
    "UserInfoLeak",
    "ApiKeyLeak",
    "DataLeak",
    "AmbiguousArg",
    "HallucinatedTool",
    "HallucinatedArgValue",
    "RedundantArg",
    "MissingTypeHint",
    "VersionConflict",
    "DescriptionMismatch",
    "benign",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TraceSafe-Bench strict evaluation logs.")
    parser.add_argument("--output-json", default="outputs/logs/tracesafe_summary.json")
    parser.add_argument("--output-md", default="outputs/reports/tracesafe_summary.md")
    args = parser.parse_args()

    loaded = {name: payload for name, path in DEFAULT_LOGS.items() if (payload := load_json(Path(path))) is not None}
    summary = build_summary(loaded)
    write_json(Path(args.output_json), summary)
    markdown = render_markdown(summary)
    write_text(Path(args.output_md), markdown)
    print(markdown)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def build_summary(logs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": {name: overall_metrics(payload) for name, payload in logs.items()},
        "per_category_asr": per_category_asr(logs),
    }


def overall_metrics(payload: dict[str, Any]) -> dict[str, float | int | None]:
    metrics = payload.get("metrics", {})
    return {
        "strict_asr": number(metrics.get("attack_success_rate")),
        "interception": number(metrics.get("attack_interception_rate")),
        "over_refusal": number(metrics.get("over_refusal_rate")),
        "task_success": number(metrics.get("task_success_rate")),
        "valid_format": number(payload.get("valid_json_rate")),
        "json_checked": int(payload.get("json_checked", 0)),
    }


def per_category_asr(logs: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    categories = categories_present(logs)
    table: dict[str, dict[str, float | None]] = {}
    for category in categories:
        table[category] = {}
        for name, payload in logs.items():
            category_payload = payload.get("category_metrics", {}).get(category, {})
            metrics = category_payload.get("metrics", {}) if isinstance(category_payload, dict) else {}
            table[category][name] = number(metrics.get("attack_success_rate"))
    return table


def categories_present(logs: dict[str, dict[str, Any]]) -> list[str]:
    present: set[str] = set()
    for payload in logs.values():
        category_metrics = payload.get("category_metrics")
        if isinstance(category_metrics, dict):
            present.update(str(category) for category in category_metrics)
    ordered = [category for category in CATEGORY_ORDER if category in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def render_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    defenders = list(overall)
    lines = ["# TraceSafe-Bench Results", "", "## Overall"]
    lines.append(markdown_row(["Model", "Strict ASR↓", "Interception↑", "Over-refusal↓", "Task success↑", "Valid format↑"]))
    lines.append(markdown_separator(6))
    for defender, metrics in overall.items():
        lines.append(
            markdown_row(
                [
                    defender,
                    pct(metrics["strict_asr"]),
                    pct(metrics["interception"]),
                    pct(metrics["over_refusal"]),
                    pct(metrics["task_success"]),
                    pct(metrics["valid_format"]),
                ]
            )
        )
    lines.extend(["", "## Per-Category ASR↓"])
    lines.append(markdown_row(["Category", *defenders]))
    lines.append(markdown_separator(len(defenders) + 1))
    for category, row in summary["per_category_asr"].items():
        lines.append(markdown_row([category, *[pct(row.get(defender)) for defender in defenders]]))
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def markdown_separator(width: int) -> str:
    return "| " + " | ".join(["---"] * width) + " |"


if __name__ == "__main__":
    main()
