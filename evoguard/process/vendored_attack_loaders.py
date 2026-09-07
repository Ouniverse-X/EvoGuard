"""Dataset name -> vendored-attack loader.

``eval/vendored_replay.py`` used to call
:func:`~evoguard.process.vendored_attack_parser.load_all_vendored_attacks`
directly, which hardcodes AgentDojo's ``data/toolsafe/<...>/*.json`` layout and
its ``<INFORMATION>`` transcript scraping. ASB-OPI, InjecAgent, AgentDojo-Latent
and its AgentDojo-Stock control twin store their injections as first-class JSONL
fields instead, so the replay entry point dispatches through this table.

Every loader takes the same keyword-compatible signature
``(data_root, suites, dataset_dir, dataset) -> list[VendoredAttack]`` and is
responsible for emitting ``task_id`` values that match its env's
``get_tasks()``. Unregistered datasets fall back to the AgentDojo loader, which
is what the pre-registry behaviour was.
"""

from __future__ import annotations

from typing import Callable, Optional

from evoguard.envs.agentdojo_latent import DATASET_NAME as AGENTDOJO_LATENT_DATASET
from evoguard.envs.agentdojo_stock import DATASET_NAME as AGENTDOJO_STOCK_DATASET
from evoguard.envs.asb import DATASET_NAME as ASB_DATASET
from evoguard.envs.injecagent import DATASET_NAME as INJECAGENT_DATASET
from evoguard.process.agentdojo_latent_loader import load_agentdojo_latent_attacks
from evoguard.process.agentdojo_stock_loader import load_agentdojo_stock_attacks
from evoguard.process.asb_attack_loader import load_asb_attacks
from evoguard.process.injecagent_attack_loader import load_injecagent_attacks
from evoguard.process.vendored_attack_parser import (
    VendoredAttack,
    load_all_vendored_attacks,
)

VendoredAttackLoader = Callable[..., list[VendoredAttack]]

_LOADERS: dict[str, VendoredAttackLoader] = {
    AGENTDOJO_LATENT_DATASET: load_agentdojo_latent_attacks,
    AGENTDOJO_STOCK_DATASET: load_agentdojo_stock_attacks,
    ASB_DATASET: load_asb_attacks,
    INJECAGENT_DATASET: load_injecagent_attacks,
}


def register_vendored_attack_loader(dataset: str, loader: VendoredAttackLoader) -> None:
    _LOADERS[dataset] = loader


def load_vendored_scenarios(
    data_root: str = "data",
    suites: Optional[list[str]] = None,
    dataset_dir: Optional[str] = None,
    dataset: str = "agentdojo",
) -> list[VendoredAttack]:
    """Load replay scenarios for ``dataset`` from its own on-disk layout."""

    loader = _LOADERS.get(dataset, load_all_vendored_attacks)
    return loader(
        data_root, suites=suites, dataset_dir=dataset_dir, dataset=dataset,
    )


__all__ = [
    "VendoredAttackLoader",
    "load_vendored_scenarios",
    "register_vendored_attack_loader",
]
