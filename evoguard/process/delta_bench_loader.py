"""Δ-Bench scenario rows -> :class:`VendoredAttack` objects.

Structurally identical to
:mod:`evoguard.process.agentdojo_latent_loader` -- every field the replay needs
is a first-class column and there is no ``<INFORMATION>`` delimiter to scrape --
so only the file layout differs: the rows live in
``scenarios/bucket_{imm,d1,d2,d3}.jsonl`` rather than per suite.

``include_clean=False`` is not optional here. ``bucket_clean.jsonl`` exists for
the *env*, which serves those rows' ``benign_observation`` on the clean arm;
replaying them as attacks would count 18 payload-free rows as attempts and drag
ASR down by a fifth. The extension groups ship no clean row at all -- they were
added to measure ASR only -- so BU and UA remain figures about the core tasks.

Δ does not travel on :class:`VendoredAttack` (no metadata slot), so the tier is
recovered by joining the scenario files back onto the replay records on
``(task_id, payload)`` -- unique across every injected row, verified by
``tests/test_delta_bench.py``. ``scripts/summarize_delta_bench.py`` does exactly
that, and it recovers ``source`` the same way. The ``tiers`` and ``sources``
arguments are the other route: they restrict the loader so a replay can be run
per rung or per source.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

from evoguard.envs.delta_bench import (
    DATASET_NAME,
    SOURCES,
    SUITES,
    TIERS,
    delta_bench_root,
    iter_scenario_rows,
)
from evoguard.process.vendored_attack_parser import VendoredAttack
from evoguard.utils.logging import get_logger

logger = get_logger("process.delta_bench_loader")


def _root_from_dir(root: str, dataset_dir: Optional[str]) -> str:
    """Accept ``data/delta_bench`` or ``data/delta_bench/scenarios`` or ``None``."""

    if not dataset_dir:
        return root
    norm = os.path.normpath(dataset_dir)
    if os.path.basename(norm) == "scenarios":
        return os.path.dirname(norm)
    return norm


def load_delta_bench_attacks(
    data_root: str = "data",
    suites: Optional[list[str]] = None,
    dataset_dir: Optional[str] = None,
    dataset: str = DATASET_NAME,  # noqa: ARG001 - the built uid owns the name
    *,
    subdir: str = DATASET_NAME,
    tiers: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
) -> list[VendoredAttack]:
    """Return every injected row as a replayable scenario.

    Signature mirrors
    :func:`evoguard.process.vendored_attack_parser.load_all_vendored_attacks` so
    :mod:`evoguard.eval.vendored_replay` can dispatch on dataset name alone.
    ``dataset`` is accepted and ignored: ``scripts/build_delta_bench.py`` owns
    the uid prefix and overriding it would silently match zero tasks.

    ``sources`` restricts to ``agentdojo`` / ``asb`` / ``injecagent``. The
    ladder is ragged -- the extension sources contribute the ``imm`` rung only
    and their base ASR differs from AgentDojo's by a factor of seven -- so a
    by-tier number pooled across sources is not interpretable. Either run one
    replay per source or let ``scripts/summarize_delta_bench.py`` split them
    apart afterwards; it reports per source either way.
    """

    root = _root_from_dir(delta_bench_root(data_root, subdir), dataset_dir)
    wanted_suites = [s for s in (suites or SUITES) if s in SUITES]
    wanted_tiers = [t for t in (tiers or TIERS) if t in TIERS]
    wanted_sources = [s for s in (sources or SOURCES) if s in SOURCES]

    out: list[VendoredAttack] = []
    by_tier: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for rec in iter_scenario_rows(root, wanted_suites, wanted_tiers,
                                  include_clean=False):
        if rec.get("sample_type") != "injected":
            continue
        # Rows written before schema 3 carry no ``source``; they are all
        # AgentDojo, which is what the default admits.
        src = str(rec.get("source") or "agentdojo")
        if src not in wanted_sources:
            continue
        mal_tool = str(rec["harmful_tool"])
        goal_text = str(rec["attacker_goal"]).strip()
        hint = rec.get("harmful_args_hint") or {}
        by_tier[str(rec["tier"])] = by_tier.get(str(rec["tier"]), 0) + 1
        by_source[src] = by_source.get(src, 0) + 1
        out.append(
            VendoredAttack(
                task_id=str(rec["task_id"]),
                suite=str(rec["suite"]),
                target_tool=str(rec["carrier_tool"]),
                real_observation=str(rec["observation"]),
                payload=str(rec["payload"]),
                mal_tool=mal_tool,
                mal_args=json.dumps(hint, sort_keys=True, ensure_ascii=False),
                goal_text=goal_text,
                # `call <tool>` prefix is load-bearing: dataset_builder.
                # _infer_target_tool and the judge prompt both read the target
                # tool out of this string.
                goal=f"call {mal_tool} -- {goal_text}".strip(" -"),
                # Authored, never replayed against a reference model, so there
                # is no upstream score; 0.0 means "unknown", as elsewhere.
                ground_truth_score=0.0,
                reference_fooled_action=" -> ".join(
                    str(t) for t in rec.get("expected_induced_plan") or []
                ),
            )
        )

    logger.info(
        "delta_bench_loader: root=%s suites=%s scenarios=%d by_tier=%s by_source=%s",
        root, ",".join(wanted_suites), len(out),
        ",".join(f"{t}={by_tier.get(t, 0)}" for t in wanted_tiers),
        ",".join(f"{s}={by_source.get(s, 0)}" for s in wanted_sources),
    )
    return out


__all__ = ["load_delta_bench_attacks"]
