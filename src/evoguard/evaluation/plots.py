"""Report and SVG generation for EvoGuard experiment logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASELINE_ORDER = (
    "no_guard",
    "rule_based_guard",
    "always_refuse_guard",
    "static_guard",
    "eapo_fixed_contrastive_rollout",
    "evoguard_adaptive",
)

BASELINE_METRICS = (
    "task_success_rate",
    "attack_interception_rate",
    "over_refusal_rate",
    "attack_success_rate",
)

TRAINING_METRICS = (
    "task_success_rate",
    "attack_interception_rate",
    "over_refusal_rate",
    "attack_success_rate",
)

ABLATION_ORDER = (
    "full_evoguard",
    "no_adaptive_attack_generation",
    "no_attack_round_weighting",
    "no_confidence_weighting",
    "no_stale_error_penalty",
    "no_attribution_reward",
    "no_no_tool_trajectory",
    "no_clean_tool_trajectory",
    "static_attack_pool",
    "binary_action_space_proxy",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_path = Path(path)
    if not input_path.exists():
        return rows
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def latest_baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_name[str(row["baseline"])] = row
    ordered = [by_name[name] for name in BASELINE_ORDER if name in by_name]
    ordered.extend(row for name, row in by_name.items() if name not in BASELINE_ORDER)
    return ordered


def latest_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_round: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_round[int(row["round_id"])] = row
    return [by_round[round_id] for round_id in sorted(by_round)]


def latest_ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_name[str(row["ablation"])] = row
    ordered = [by_name[name] for name in ABLATION_ORDER if name in by_name]
    ordered.extend(row for name, row in by_name.items() if name not in ABLATION_ORDER)
    return ordered


def baseline_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Baseline",
        "Task Success",
        "Attack Interception",
        "Over-Refusal",
        "Attack Success",
        "Train Split",
        "Eval Split",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _display_name(row["baseline"]),
                    _pct(metrics["task_success_rate"]),
                    _pct(metrics["attack_interception_rate"]),
                    _pct(metrics["over_refusal_rate"]),
                    _pct(metrics["attack_success_rate"]),
                    str(row.get("train_attack_split", "train")),
                    str(row.get("eval_attack_split", "heldout")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def baseline_latex_table(rows: list[dict[str, Any]]) -> str:
    body = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Baseline & Task Success & Interception & Over-Refusal & Attack Success \\\\",
        "\\midrule",
    ]
    for row in rows:
        metrics = row["metrics"]
        body.append(
            " & ".join(
                [
                    _latex_escape(_display_name(row["baseline"])),
                    _pct(metrics["task_success_rate"]),
                    _pct(metrics["attack_interception_rate"]),
                    _pct(metrics["over_refusal_rate"]),
                    _pct(metrics["attack_success_rate"]),
                ]
            )
            + " \\\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(body)


def training_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Round",
        "Task Success",
        "Attack Interception",
        "Over-Refusal",
        "Attack Success",
        "Mean Reward",
        "Policy Updates",
        "Approx KL",
        "Clip Fraction",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["round_id"]),
                    _pct(metrics["task_success_rate"]),
                    _pct(metrics["attack_interception_rate"]),
                    _pct(metrics["over_refusal_rate"]),
                    _pct(metrics["attack_success_rate"]),
                    f"{float(row['mean_reward']):.3f}",
                    str(row["policy_updates"]),
                    f"{float((row.get('update_stats') or {}).get('approx_kl', 0.0)):.4f}",
                    f"{float((row.get('update_stats') or {}).get('clip_fraction', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def ablation_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Ablation",
        "Task Success",
        "Attack Interception",
        "Over-Refusal",
        "Attack Success",
        "Policy Updates",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _display_name(row["ablation"]),
                    _pct(metrics["task_success_rate"]),
                    _pct(metrics["attack_interception_rate"]),
                    _pct(metrics["over_refusal_rate"]),
                    _pct(metrics["attack_success_rate"]),
                    str(row.get("policy_updates", 0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def ablation_latex_table(rows: list[dict[str, Any]]) -> str:
    body = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Ablation & Task Success & Interception & Over-Refusal & Attack Success \\\\",
        "\\midrule",
    ]
    for row in rows:
        metrics = row["metrics"]
        body.append(
            " & ".join(
                [
                    _latex_escape(_display_name(row["ablation"])),
                    _pct(metrics["task_success_rate"]),
                    _pct(metrics["attack_interception_rate"]),
                    _pct(metrics["over_refusal_rate"]),
                    _pct(metrics["attack_success_rate"]),
                ]
            )
            + " \\\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(body)


def baseline_svg(rows: list[dict[str, Any]]) -> str:
    width = 980
    height = 430
    margin_left = 64
    margin_bottom = 110
    chart_width = width - margin_left - 28
    chart_height = height - 70 - margin_bottom
    y0 = 70 + chart_height
    group_width = chart_width / max(1, len(rows))
    bar_width = min(18, group_width / (len(BASELINE_METRICS) + 1))
    colors = {
        "task_success_rate": "#2f7ed8",
        "attack_interception_rate": "#2ca02c",
        "over_refusal_rate": "#d62728",
        "attack_success_rate": "#ff7f0e",
    }
    parts = _svg_header(width, height, "Held-out Baseline Metrics")
    parts.extend(_axis_parts(margin_left, y0, chart_width, chart_height))

    for group_idx, row in enumerate(rows):
        base_x = margin_left + group_idx * group_width + group_width * 0.18
        for metric_idx, metric in enumerate(BASELINE_METRICS):
            value = float(row["metrics"][metric])
            bar_height = value * chart_height
            x = base_x + metric_idx * (bar_width + 2)
            y = y0 - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'fill="{colors[metric]}"><title>{_display_name(row["baseline"])} {metric}: {_pct(value)}</title></rect>'
            )
        label_x = margin_left + group_idx * group_width + group_width * 0.42
        parts.append(
            f'<text x="{label_x:.1f}" y="{y0 + 18}" font-size="10" text-anchor="end" '
            f'transform="rotate(-35 {label_x:.1f},{y0 + 18})">{_xml_escape(_display_name(row["baseline"]))}</text>'
        )

    parts.extend(_legend(colors, width - 330, 22))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def training_svg(rows: list[dict[str, Any]]) -> str:
    width = 880
    height = 430
    margin_left = 64
    margin_bottom = 60
    chart_width = width - margin_left - 38
    chart_height = height - 70 - margin_bottom
    y0 = 70 + chart_height
    colors = {
        "task_success_rate": "#2f7ed8",
        "attack_interception_rate": "#2ca02c",
        "over_refusal_rate": "#d62728",
        "attack_success_rate": "#ff7f0e",
    }
    parts = _svg_header(width, height, "EvoGuard Training Curves")
    parts.extend(_axis_parts(margin_left, y0, chart_width, chart_height))

    rounds = [int(row["round_id"]) for row in rows]
    min_round = min(rounds) if rounds else 0
    max_round = max(rounds) if rounds else 1
    span = max(1, max_round - min_round)

    for metric in TRAINING_METRICS:
        points: list[tuple[float, float]] = []
        for row in rows:
            x = margin_left + ((int(row["round_id"]) - min_round) / span) * chart_width
            y = y0 - float(row["metrics"][metric]) * chart_height
            points.append((x, y))
        if not points:
            continue
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{colors[metric]}" stroke-width="2.5" />')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[metric]}" />')

    for row in rows:
        x = margin_left + ((int(row["round_id"]) - min_round) / span) * chart_width
        parts.append(f'<text x="{x:.1f}" y="{y0 + 24}" font-size="11" text-anchor="middle">{row["round_id"]}</text>')
    parts.append(f'<text x="{margin_left + chart_width / 2:.1f}" y="{height - 12}" font-size="12" text-anchor="middle">Evolution Round</text>')
    parts.extend(_legend(colors, width - 330, 22))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def ablation_svg(rows: list[dict[str, Any]]) -> str:
    compact_rows = rows[:10]
    width = 1120
    height = 500
    margin_left = 64
    margin_bottom = 145
    chart_width = width - margin_left - 28
    chart_height = height - 70 - margin_bottom
    y0 = 70 + chart_height
    group_width = chart_width / max(1, len(compact_rows))
    bar_width = min(16, group_width / (len(BASELINE_METRICS) + 1))
    colors = {
        "task_success_rate": "#2f7ed8",
        "attack_interception_rate": "#2ca02c",
        "over_refusal_rate": "#d62728",
        "attack_success_rate": "#ff7f0e",
    }
    parts = _svg_header(width, height, "Held-out Ablation Metrics")
    parts.extend(_axis_parts(margin_left, y0, chart_width, chart_height))

    for group_idx, row in enumerate(compact_rows):
        base_x = margin_left + group_idx * group_width + group_width * 0.14
        for metric_idx, metric in enumerate(BASELINE_METRICS):
            value = float(row["metrics"][metric])
            bar_height = value * chart_height
            x = base_x + metric_idx * (bar_width + 2)
            y = y0 - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'fill="{colors[metric]}"><title>{_display_name(row["ablation"])} {metric}: {_pct(value)}</title></rect>'
            )
        label_x = margin_left + group_idx * group_width + group_width * 0.42
        parts.append(
            f'<text x="{label_x:.1f}" y="{y0 + 18}" font-size="9" text-anchor="end" '
            f'transform="rotate(-40 {label_x:.1f},{y0 + 18})">{_xml_escape(_display_name(row["ablation"]))}</text>'
        )

    parts.extend(_legend(colors, width - 330, 22))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_report_artifacts(
    baseline_log: str | Path,
    training_log: str | Path,
    report_dir: str | Path,
    figure_dir: str | Path,
    ablation_log: str | Path | None = None,
) -> dict[str, Path]:
    report_path = Path(report_dir)
    figure_path = Path(figure_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    baseline_rows = latest_baseline_rows(read_jsonl(baseline_log))
    training_rows = latest_training_rows(read_jsonl(training_log))
    ablation_rows = latest_ablation_rows(read_jsonl(ablation_log)) if ablation_log else []

    outputs = {
        "baseline_markdown": report_path / "baseline_table.md",
        "baseline_latex": report_path / "baseline_table.tex",
        "training_markdown": report_path / "training_curve_table.md",
        "ablation_markdown": report_path / "ablation_table.md",
        "ablation_latex": report_path / "ablation_table.tex",
        "baseline_svg": figure_path / "baseline_metrics.svg",
        "training_svg": figure_path / "training_curves.svg",
        "ablation_svg": figure_path / "ablation_metrics.svg",
    }
    outputs["baseline_markdown"].write_text(baseline_markdown_table(baseline_rows), encoding="utf-8")
    outputs["baseline_latex"].write_text(baseline_latex_table(baseline_rows), encoding="utf-8")
    outputs["training_markdown"].write_text(training_markdown_table(training_rows), encoding="utf-8")
    outputs["ablation_markdown"].write_text(ablation_markdown_table(ablation_rows), encoding="utf-8")
    outputs["ablation_latex"].write_text(ablation_latex_table(ablation_rows), encoding="utf-8")
    outputs["baseline_svg"].write_text(baseline_svg(baseline_rows), encoding="utf-8")
    outputs["training_svg"].write_text(training_svg(training_rows), encoding="utf-8")
    outputs["ablation_svg"].write_text(ablation_svg(ablation_rows), encoding="utf-8")
    return outputs


def summarize_curve_points(metrics_by_round: list[dict[str, float]]) -> list[dict[str, float]]:
    return metrics_by_round


def _display_name(name: str) -> str:
    titled = name.replace("_", " ").title()
    return titled.replace("Eapo", "EAPO").replace("Evoguard", "EvoGuard")


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}"


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2:.1f}" y="28" font-size="18" font-weight="700" text-anchor="middle">{_xml_escape(title)}</text>',
    ]


def _axis_parts(margin_left: int, y0: int, chart_width: int, chart_height: int) -> list[str]:
    parts = [
        f'<line x1="{margin_left}" y1="{y0}" x2="{margin_left + chart_width}" y2="{y0}" stroke="#333" />',
        f'<line x1="{margin_left}" y1="{y0}" x2="{margin_left}" y2="{y0 - chart_height}" stroke="#333" />',
    ]
    for tick in range(0, 101, 25):
        y = y0 - (tick / 100) * chart_height
        parts.append(f'<line x1="{margin_left - 4}" y1="{y:.1f}" x2="{margin_left + chart_width}" y2="{y:.1f}" stroke="#ddd" />')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{tick}</text>')
    parts.append(
        f'<text x="18" y="{y0 - chart_height / 2:.1f}" font-size="12" text-anchor="middle" '
        f'transform="rotate(-90 18,{y0 - chart_height / 2:.1f})">Percent</text>'
    )
    return parts


def _legend(colors: dict[str, str], x: int, y: int) -> list[str]:
    parts: list[str] = []
    for idx, (metric, color) in enumerate(colors.items()):
        item_y = y + idx * 18
        parts.append(f'<rect x="{x}" y="{item_y}" width="11" height="11" fill="{color}" />')
        parts.append(f'<text x="{x + 16}" y="{item_y + 10}" font-size="11">{_xml_escape(metric.replace("_", " "))}</text>')
    return parts
