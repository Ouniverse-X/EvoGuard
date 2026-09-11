"""Δ-Bench environment: the Δ-stratified twin ladder.

Data lives in ``data/delta_bench`` and is built by
``scripts/build_delta_bench.py``. Structurally it is the latent arm -- same
AgentDojo v1 tasks, same clean plans, same authored carrier observations, same
sinks -- so this module is a thin re-point of
:class:`~evoguard.envs.agentdojo_latent.AgentDojoLatentEnv` at another
directory, exactly like :mod:`evoguard.envs.agentdojo_stock`.

One thing genuinely differs: **the scenarios are sharded by Δ tier, not by
suite.** ``scenarios/bucket_{imm,d1,d2,d3}.jsonl`` hold the injected rows and
``bucket_clean.jsonl`` the payload-free ones, so the latent arm's per-suite
``iter_scenario_rows`` cannot read this tree. The ``_iter_rows`` hook on the
base env is the seam; everything after it -- task de-duplication, the
per-carrier benign map, tool parsing -- stays shared, which is what keeps the
arms comparable.

**The buckets are RAGGED and the tree is MULTI-SOURCE.** Only the core
AgentDojo groups carry all four tiers; the AgentDojo extension carries
``imm``+``d1`` and the ``asb``/``injecagent`` groups carry ``imm`` alone, so
``bucket_imm.jsonl`` is the longest file by a wide margin. Nothing in this
module needs to know that -- the base env is generic over the row schema and
derives each task's tool specs from the rendered ``env_info`` string, which is
why ASB and InjecAgent rows need no execution-layer work. What it DOES need is
the widened :data:`SUITES` below. A by-tier ASR pooled across sources is
meaningless (base ASR differs across them by a factor of seven); report per
``source``.

The ``tiers`` argument exists so a caller can load one rung in isolation, but
note that **it filters the env's tasks, not the attacks**: two groups can share a
task, and the clean bucket is loaded unconditionally so the clean arm always has
its carrier observation. To replay one tier, filter the *scenarios* -- see
``tiers`` on :func:`evoguard.process.delta_bench_loader.load_delta_bench_attacks`.

Two banking groups share ``banking:UserTask15`` while injecting at different
carriers, so the env's task count (17) is below the group count (19). That is
handled by the inherited ``benign_observations`` map being keyed on carrier
tool rather than on task alone.

Like the latent and stock arms this carries no ``metadata["split"]``: the whole
ladder is a held-out diagnostic. Training on it would defeat its purpose.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, Optional, Sequence

from evoguard.envs.agentdojo_latent import (
    AgentDojoLatentEnv,
    verify_guard,
)

#: Registered env name and on-disk subdirectory. The built ``task_id`` prefix
#: spells this too, so a mismatch matches zero attacks against zero tasks.
DATASET_NAME = "delta_bench"

#: Provenance of a group. ``asb`` and ``injecagent`` groups contribute the
#: ``imm`` rung ONLY -- their clean plans are 2 and 1 tool long, so no deeper
#: tier is constructible. See ``scripts/build_delta_bench.py``.
SOURCES = ("agentdojo", "asb", "injecagent")

#: Suite values present in the tree. The four AgentDojo suites plus one
#: pseudo-suite per extension source, so the by-suite table stays readable
#: instead of fanning out into ten one-group ASB agents.
#:
#: **This is deliberately NOT ``agentdojo_latent.SUITES``.** The parent env
#: filters rows against its own four-suite tuple, and a non-empty filter that
#: omits ``asb``/``injecagent`` drops every extension row SILENTLY -- the tree
#: still loads and the replay still reports an ASR, just over the core 19.
SUITES = ("banking", "slack", "travel", "workspace", "asb", "injecagent")

#: The ladder's rungs, in increasing deferral distance.
TIERS = ("imm", "d1", "d2", "d3")

#: Declared plan-index offset from the carrier for each rung. ``imm`` is 1, not
#: 0: the smallest constructible offset places the harmful call on the very next
#: action. See ``data/delta_bench/README.md``.
TIER_DELTA = {"imm": 1, "d1": 2, "d2": 3, "d3": 4}

#: Bucket holding the payload-free rows that supply the clean arm's carrier
#: observations. Always loaded, never replayed as an attack.
CLEAN_BUCKET = "clean"

#: Same six camouflage families as the latent arm; one per case group, constant
#: across that group's four tiers.
TECHNIQUES = (
    "compliance_deferral",
    "dependency_chain",
    "staged_verification",
    "false_correction",
    "quota_precheck",
    "audit_trail",
)


def delta_bench_root(data_root: str = "data",
                     subdir: str = DATASET_NAME) -> str:
    return os.path.join(data_root, subdir)


def iter_scenario_rows(
    root: str,
    suites: Optional[Sequence[str]] = None,
    tiers: Optional[Sequence[str]] = None,
    *,
    include_clean: bool = True,
) -> Iterator[dict]:
    """Yield rows from ``scenarios/bucket_<tier>.jsonl`` in tier order.

    ``suites`` filters rows after loading (the buckets are not sharded by
    suite). ``include_clean`` appends ``bucket_clean.jsonl`` -- keep it on for
    the env, turn it off in the scenario loader.
    """

    wanted_suites = set(suites) if suites else None
    buckets = list(tiers or TIERS)
    if include_clean:
        buckets.append(CLEAN_BUCKET)
    for bucket in buckets:
        path = os.path.join(root, "scenarios", f"bucket_{bucket}.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"delta_bench scenarios not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if wanted_suites and rec.get("suite") not in wanted_suites:
                    continue
                yield rec


class DeltaBenchEnv(AgentDojoLatentEnv):
    """Δ-Bench tasks from ``data/delta_bench``."""

    name = DATASET_NAME

    def __init__(self, *args, subdir: str = DATASET_NAME,
                 suites: Optional[Sequence[str]] = None,
                 tiers: Optional[Sequence[str]] = None, **kwargs):
        self._tiers = [t for t in (tiers or TIERS) if t in TIERS]
        # Own suite filter, resolved BEFORE ``super().__init__`` because the
        # base class calls ``_load()`` (and therefore ``_iter_rows``) from its
        # constructor. Reading ``self._suite_filter`` instead would silently
        # drop the ASB and InjecAgent rows -- the parent narrows that attribute
        # to the four AgentDojo suites.
        self._dbench_suites = [s for s in (suites or SUITES) if s in SUITES]
        super().__init__(*args, subdir=subdir, suites=suites, **kwargs)

    def _iter_rows(self) -> Iterator[dict]:
        return iter_scenario_rows(self._root, self._dbench_suites, self._tiers)


__all__ = [
    "CLEAN_BUCKET",
    "DATASET_NAME",
    "SOURCES",
    "SUITES",
    "TECHNIQUES",
    "TIERS",
    "TIER_DELTA",
    "DeltaBenchEnv",
    "delta_bench_root",
    "iter_scenario_rows",
    "verify_guard",
]
