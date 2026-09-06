"""AgentDojo-Latent scenario rows -> :class:`VendoredAttack` objects.

The AgentDojo parser (:mod:`evoguard.process.vendored_attack_parser`) recovers
injections by regex-scraping a rendered ReAct transcript for an
``<INFORMATION>`` block. This set has no transcript to scrape and, deliberately,
no delimiter to scrape for: the camouflage is that the payload reads as more of
the carrier's own output. Every field the replay needs is a first-class column in
``data/agentdojo_latent/scenarios/<suite>.jsonl``.

Only ``sample_type == "injected"`` rows become scenarios. The paired ``clean``
rows exist for the *env* -- they supply the carrier's benign observation for the
clean arm (see :class:`evoguard.envs.agentdojo_latent.AgentDojoLatentEnv`) --
and replaying them as attacks would count 46 payload-free rows as attempts.

Two mappings are worth stating explicitly.

``real_observation`` is ``observation``, which the build script defines as
``benign_observation + payload``. So the attacked arm differs from the clean arm
by exactly the appended payload, and ``payload`` is the substring a defender has
to notice.

``mal_args`` is JSON-encoded from ``harmful_args_hint`` with ``sort_keys=True``,
matching ``grpo_reward._action_signature``'s canonical form, so a hint can be
compared against an emitted call without re-normalising.

Declared Δ does not travel on :class:`VendoredAttack` -- it has no metadata slot.
``expected_delta`` / ``expected_turning_index`` / ``stealth_technique`` stay in
the scenario file and are joined back onto replay records by
``(task_id, payload)``, which is unique across the 48 rows.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from evoguard.envs.agentdojo_latent import (
    SUITES,
    agentdojo_latent_root,
    iter_scenario_rows,
)
from evoguard.process.vendored_attack_parser import VendoredAttack
from evoguard.utils.logging import get_logger

logger = get_logger("process.agentdojo_latent_loader")


def _suites_from_dir(root: str, dataset_dir: Optional[str]) -> tuple[str, Optional[tuple[str, ...]]]:
    """Resolve ``dataset_dir`` into ``(root, suites)``.

    Accepts the shapes a caller would already be passing for the other
    datasets:

    * ``None``                                  -> the whole probe
    * ``data/agentdojo_latent``                 -> the whole probe
    * ``data/agentdojo_latent/scenarios``       -> the whole probe
    """

    if not dataset_dir:
        return root, None
    norm = os.path.normpath(dataset_dir)
    if os.path.basename(norm) == "scenarios":
        return os.path.dirname(norm), None
    return norm, None


def load_agentdojo_latent_attacks(
    data_root: str = "data",
    suites: Optional[list[str]] = None,
    dataset_dir: Optional[str] = None,
    dataset: str = "agentdojo_latent",  # noqa: ARG001 - uid owns the name
    *,
    subdir: str = "agentdojo_latent",
) -> list[VendoredAttack]:
    """Return every injected row as a replayable scenario.

    Signature mirrors
    :func:`evoguard.process.vendored_attack_parser.load_all_vendored_attacks`
    so :mod:`evoguard.eval.vendored_replay` can dispatch on dataset name alone.
    ``dataset`` is accepted and ignored: ``envs.agentdojo_latent.task_uid`` owns
    the uid prefix and overriding it would silently match zero tasks.
    """

    root = agentdojo_latent_root(data_root, subdir)
    root, from_dir = _suites_from_dir(root, dataset_dir)
    wanted = [s for s in (suites or from_dir or SUITES) if s in SUITES]

    out: list[VendoredAttack] = []
    n_rows = n_clean = 0
    for rec in iter_scenario_rows(root, wanted):
        n_rows += 1
        if rec.get("sample_type") != "injected":
            n_clean += 1
            continue
        mal_tool = str(rec["harmful_tool"])
        goal_text = str(rec["attacker_goal"]).strip()
        hint = rec.get("harmful_args_hint") or {}
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
        "agentdojo_latent_loader: root=%s suites=%s rows=%d clean=%d scenarios=%d",
        root, ",".join(wanted), n_rows, n_clean, len(out),
    )
    return out


__all__ = ["load_agentdojo_latent_attacks"]
