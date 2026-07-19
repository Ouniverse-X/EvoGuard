"""Co-evolution curve plotting (``utils/intro.md``).

Renders the per-round :class:`~evoguard.utils.metrics.RoundMetrics` snapshots
into PNG plots so an experiment's attacker-vs-defender dynamics can be inspected
at a glance. Plotting is *optional*: ``matplotlib`` is imported lazily so that
environments without it still load EvoGuard.
"""

from __future__ import annotations

import os
from typing import Sequence

from evoguard.utils.logging import get_logger

logger = get_logger("utils.plots")


def plot_curves(metrics: Sequence["dict"], out_path: str) -> str | None:
    """Plot ASR, mean best fitness and delta distribution across rounds.

    ``metrics`` is a sequence of ``RoundMetrics.to_dict()`` dicts. Returns
    ``out_path`` on success or ``None`` if matplotlib is unavailable / plotting
    failed (the experiment still completes without charts).
    """

    try:
        import matplotlib  # noqa: F401  - backend selection side-effect only
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dep guard
        logger.warning("matplotlib unavailable (%s); skipping plot", exc)
        return None

    if not metrics:
        return None

    rounds = [m.get("round_id", i + 1) for i, m in enumerate(metrics)]
    asr = [float(m.get("attack_success_rate", 0.0)) for m in metrics]
    mean_best = [float(m.get("mean_best_fitness", 0.0)) for m in metrics]
    elite = [float(m.get("elite_fitness_mean", 0.0)) for m in metrics]
    dn = [
        float(m.get("delta_normalized_mean_on_success", 0.0))
        for m in metrics
    ]
    immediate_share = [float(m.get("delta_immediate_share", 0.0)) for m in metrics]
    latent_share = [float(m.get("delta_latent_share", 0.0)) for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0][0]
    ax.plot(rounds, asr, marker="o", label="ASR")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=0.7, label="ε=0.05")
    ax.set_title("Attack success rate")
    ax.set_xlabel("round"); ax.set_ylabel("rate"); ax.set_ylim(-0.02, 1.02); ax.legend()

    ax = axes[0][1]
    ax.plot(rounds, mean_best, marker="s", label="mean best fitness")
    ax.plot(rounds, elite, marker="^", label="elite fitness mean")
    ax.set_title("Attacker fitness over rounds")
    ax.set_xlabel("round"); ax.set_ylabel("fitness ∈[0,1]"); ax.set_ylim(-0.02, 1.02); ax.legend()

    ax = axes[1][0]
    ax.plot(rounds, dn, marker="o", color="purple")
    ax.set_title("Mean Δ_normalized on successful attacks\n(defender wants this small)")
    ax.set_xlabel("round"); ax.set_ylabel("Δ_norm")

    ax = axes[1][1]
    ax.plot(rounds, immediate_share, marker="o", label="immediate share (Δ≤1)")
    ax.plot(rounds, latent_share, marker="^", label="latent share (Δ≥3)")
    ax.set_title("Attack latency mix on successes")
    ax.set_xlabel("round"); ax.set_ylabel("share"); ax.set_ylim(-0.02, 1.02); ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_metrics_csv(metrics: list["dict"], csv_path: str) -> str:
    """Append a flat CSV of scalar metric fields for quick spreadsheet viewing."""

    import csv as _csv

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    fields: dict[str, object] = {}
    rows: list[list[str]] = []
    header_written = False

    def _flatten(d: dict, prefix: str = "") -> dict[str, "object"]:
        flat: dict[str, object] = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, list):
                flat[key] = ";".join(str(x) for x in v)
            elif isinstance(v, dict):
                continue
            else:
                flat[key] = "" if v is None else str(v)
        return flat

    flattened_rows = [_flatten(m) for m in metrics]
    keys: list[str] = sorted({k for r in flattened_rows for k in r.keys()})
    with open(csv_path, "w", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(keys)
        for row in flattened_rows:
            writer.writerow([row.get(k, "") for k in keys])
    return csv_path
