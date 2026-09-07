"""AgentDojo-Stock scenario rows -> :class:`VendoredAttack` objects.

The row schema is the latent set's plus three ``twin_*`` columns, so this is a
one-line re-point of
:func:`evoguard.process.agentdojo_latent_loader.load_agentdojo_latent_attacks`
at ``data/agentdojo_stock``. It exists as its own module only because
:func:`evoguard.process.vendored_attack_loaders.load_vendored_scenarios` calls
loaders with a fixed keyword signature and has no seam for binding ``subdir``.

The ``twin_*`` columns do not travel on :class:`VendoredAttack` (which has no
metadata slot), same as declared Δ on the latent arm. Stratifying a replay by
the twin's declared Δ means joining the scenario file back onto the replay
records by ``(task_id, payload)``.
"""

from __future__ import annotations

from typing import Optional

from evoguard.envs.agentdojo_stock import DATASET_NAME
from evoguard.process.agentdojo_latent_loader import load_agentdojo_latent_attacks
from evoguard.process.vendored_attack_parser import VendoredAttack


def load_agentdojo_stock_attacks(
    data_root: str = "data",
    suites: Optional[list[str]] = None,
    dataset_dir: Optional[str] = None,
    dataset: str = DATASET_NAME,  # noqa: ARG001 - the built uid owns the name
) -> list[VendoredAttack]:
    return load_agentdojo_latent_attacks(
        data_root, suites=suites, dataset_dir=dataset_dir, subdir=DATASET_NAME,
    )


__all__ = ["load_agentdojo_stock_attacks"]
