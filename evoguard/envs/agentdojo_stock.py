"""AgentDojo-Stock environment: the Δ=0 control arm for AgentDojo-Latent.

Data lives in ``data/agentdojo_stock`` and is built by
``scripts/build_agentdojo_stock.py`` from the latent set, replacing only the
``payload`` with AgentDojo's own ``ToolKnowledgeAttack`` ``<INFORMATION>``
imperative. Everything the env itself needs -- tasks, ``env_info``, clean plans,
per-carrier benign observations -- is identical to the latent arm, so the
loading behaviour is identical too and this module is a thin re-point of
:class:`~evoguard.envs.agentdojo_latent.AgentDojoLatentEnv` at another
directory.

It is a *separate registered env* rather than a ``subdir`` argument because
:func:`evoguard.envs.build_env` passes only
``(executor, data_root, suites, max_tasks, utility_judge)`` -- there is no seam
for a directory override, and adding one would let a config silently point the
latent dataset name at other data.

Like the latent set this carries no ``metadata["split"]``: it is a held-out
diagnostic control, not training data.
"""

from __future__ import annotations

from evoguard.envs.agentdojo_latent import SUITES, AgentDojoLatentEnv

#: Registered env name and on-disk subdirectory. Both the scenario loader
#: (:mod:`evoguard.process.agentdojo_stock_loader`) and the built ``task_id``
#: prefix spell this, so a mismatch matches zero attacks against zero tasks.
DATASET_NAME = "agentdojo_stock"

#: Declared plan-index offset every injected row carries -- see the build
#: script's docstring for why the stock arm cannot declare 0.
STOCK_DELTA = 1

#: The single ``stealth_technique`` value in this set; the latent twin's own
#: technique travels on ``twin_stealth_technique``.
TECHNIQUES = ("stock_important_instructions",)


class AgentDojoStockEnv(AgentDojoLatentEnv):
    """AgentDojo-Stock tasks from ``data/agentdojo_stock``."""

    name = DATASET_NAME

    def __init__(self, *args, subdir: str = DATASET_NAME, **kwargs):
        super().__init__(*args, subdir=subdir, **kwargs)


__all__ = [
    "DATASET_NAME",
    "STOCK_DELTA",
    "SUITES",
    "TECHNIQUES",
    "AgentDojoStockEnv",
]
