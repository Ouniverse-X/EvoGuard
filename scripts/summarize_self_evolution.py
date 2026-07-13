#!/usr/bin/env python
"""Summarize EvoGuard self-evolution metrics and draw an SVG curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round",
        dest="round_specs",
        nargs="+",
        required=True,
        help="Round specs in the form round_id:pre_eval_json:post_eval_json",
    )
    parser.add_argument("--metrics-output", default="outputs/logs/self_evolution_metrics.json")
    parser.add_argument("--markdown-output", default="outputs/logs/self_evolution_metrics.md")
    parser.add_argument("--figure-output", default="outputs/figures/attack_defense_curve.svg")
    args = parser.parse_args()

    rows = []
    for spec in args.round_specs:
        round_id_text, pre_path, post_path = spec.split(":", 2)
        pre_payload = json.loads(Path(pre_path).read_text(encoding="utf-8"))
        post_payload = json.loads(Path(post_path).read_text(encoding="utf-8"))
        pre_metrics = pre_payload["metrics"]
        post_metrics = post_payload["metrics"]
        rows.append(
            {
                "round": int(round_id_text),
                "pre_eval": pre_path,
                "post_eval": post_path,
                "pre_valid_json_rate": pre_payload["valid_json_rate"],
                "post_valid_json_rate": post_payload["valid_json_rate"],
                "pre_attack_success_rate": pre_metrics["attack_success_rate"],
                "pre_attack_interception_rate": pre_metrics["attack_interception_rate"],
                "post_attack_success_rate": post_metrics["attack_success_rate"],
                "post_attack_interception_rate": post_metrics["attack_interception_rate"],
                "post_over_refusal_rate": post_metrics["over_refusal_rate"],
                "post_task_success_rate": post_metrics["task_success_rate"],
            }
        )

    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(_markdown_table(rows), encoding="utf-8")
    _write_svg(Path(args.figure_output), rows)
    print(json.dumps({"rounds": len(rows), "metrics": args.metrics_output, "figure": args.figure_output}, indent=2))


def _markdown_table(rows: list[dict[str, float]]) -> str:
    lines = [
        "| Round | Pre attack success | Pre interception | Post attack success | Post interception | Post over-refusal | Post valid JSON |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {round} | {pre_attack_success_rate:.2%} | {pre_attack_interception_rate:.2%} | "
            "{post_attack_success_rate:.2%} | {post_attack_interception_rate:.2%} | "
            "{post_over_refusal_rate:.2%} | {post_valid_json_rate:.2%} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _write_svg(path: Path, rows: list[dict[str, float]]) -> None:
    width, height = 760, 420
    left, right, top, bottom = 70, 30, 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_round = max(1, len(rows) - 1)

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + plot_w * (index / max_round)
        y = top + plot_h * (1.0 - max(0.0, min(1.0, value)))
        return x, y

    def poly(metric: str) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in [point(i, row[metric]) for i, row in enumerate(rows)])

    attack_points = poly("pre_attack_success_rate")
    defense_points = poly("post_attack_interception_rate")
    over_refusal_points = poly("post_over_refusal_rate")
    ticks = "\n".join(
        f'<text x="{left - 12}" y="{top + plot_h * (1 - t / 4) + 4:.1f}" text-anchor="end" font-size="12">{t * 25}%</text>'
        f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h * (1 - t / 4):.1f}" y2="{top + plot_h * (1 - t / 4):.1f}" stroke="#e5e7eb"/>'
        for t in range(5)
    )
    round_labels = "\n".join(
        f'<text x="{point(i, 0)[0]:.1f}" y="{height - 35}" text-anchor="middle" font-size="12">R{row["round"]}</text>'
        for i, row in enumerate(rows)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="20" text-anchor="middle" font-size="16" font-family="Arial">EvoGuard Attack-Defense Self-Evolution</text>
  {ticks}
  <line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" stroke="#111827"/>
  <line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" stroke="#111827"/>
  {round_labels}
  <polyline points="{attack_points}" fill="none" stroke="#dc2626" stroke-width="3"/>
  <polyline points="{defense_points}" fill="none" stroke="#2563eb" stroke-width="3"/>
  <polyline points="{over_refusal_points}" fill="none" stroke="#f59e0b" stroke-width="3"/>
  <rect x="520" y="48" width="200" height="80" fill="white" stroke="#d1d5db"/>
  <line x1="535" y1="70" x2="575" y2="70" stroke="#dc2626" stroke-width="3"/><text x="585" y="74" font-size="12">Attack success</text>
  <line x1="535" y1="92" x2="575" y2="92" stroke="#2563eb" stroke-width="3"/><text x="585" y="96" font-size="12">Defense interception</text>
  <line x1="535" y1="114" x2="575" y2="114" stroke="#f59e0b" stroke-width="3"/><text x="585" y="118" font-size="12">Over-refusal</text>
</svg>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
