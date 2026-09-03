"""ASB-OPI injected rows -> :class:`VendoredAttack` objects.

The AgentDojo parser (:mod:`evoguard.process.vendored_attack_parser`) recovers
injections by regex-scraping a rendered ReAct transcript. ASB-OPI needs none of
that: every injected row in ``data/ASB/splits/<split>/all.jsonl`` already carries
the payload (``wrapped_attack_prompt``), the bait tool (``attacker_tool``) and the
attacker objective (``attack_goal``) as first-class fields. What the rows do NOT
carry is the *carrier* -- upstream appends the payload to whichever tool the agent
happens to call first (``react_agent_attack.py``: ``function_response +=
f'; {attack_prompt}'``). We pin the carrier to the agent's FIRST benign tool,
which is the agent config's declared entry point and therefore the tool a
task-following agent reaches first in practice.

``real_observation`` is then reconstructed exactly as upstream would render it:
the carrier's ``Expected Achievements`` string followed by ``observation_suffix``
(which already begins with ``"; "``).
"""

from __future__ import annotations

import os
from typing import Optional

from evoguard.envs.asb import (
    SPLITS,
    asb_root,
    iter_split_rows,
    load_agent_configs,
    load_normal_tools,
    task_uid,
)
from evoguard.process.vendored_attack_parser import VendoredAttack
from evoguard.utils.logging import get_logger

logger = get_logger("process.asb_attack_loader")


def _splits_from_dir(root: str, dataset_dir: Optional[str]) -> tuple[str, Optional[tuple[str, ...]]]:
    """Resolve ``dataset_dir`` into ``(asb_root, splits)``.

    Accepts three shapes so callers can pass whatever ``--dataset-dir`` they
    would pass for AgentDojo:

    * ``None``                      -> the whole dataset under ``root``
    * ``data/ASB/splits/test``      -> that leaf split only
    * ``data/ASB/splits``           -> all splits
    """

    if not dataset_dir:
        return root, None
    norm = os.path.normpath(dataset_dir)
    leaf = os.path.basename(norm)
    if leaf in SPLITS:
        parent = os.path.dirname(norm)                       # .../ASB/splits
        return os.path.dirname(parent), (leaf,)
    if leaf == "splits":
        return os.path.dirname(norm), None
    # A bare ASB root was passed.
    return norm, None


def load_asb_attacks(
    data_root: str = "data",
    suites: Optional[list[str]] = None,
    dataset_dir: Optional[str] = None,
    dataset: str = "asb_opi",                                # noqa: ARG001 - uid owns the name
    *,
    subdir: str = "ASB",
) -> list[VendoredAttack]:
    """Return every ASB-OPI injected row as a replayable scenario.

    Signature mirrors
    :func:`evoguard.process.vendored_attack_parser.load_all_vendored_attacks`
    so :mod:`evoguard.eval.vendored_replay` can dispatch on dataset name alone.
    ``dataset`` is accepted and ignored: ``envs.asb.task_uid`` is the single
    owner of the uid prefix, and overriding it would silently match zero tasks.
    """

    root = asb_root(data_root, subdir)
    root, splits = _splits_from_dir(root, dataset_dir)
    agents = load_agent_configs(root)
    normal_tools, achievements = load_normal_tools(root)
    agent_filter = set(suites or [])

    out: list[VendoredAttack] = []
    seen: set[tuple[str, str, str]] = set()
    n_rows = n_clean = n_dupes = n_no_carrier = 0
    for _split, rec in iter_split_rows(root, splits):
        n_rows += 1
        if rec.get("sample_type") != "injected":
            n_clean += 1
            continue
        agent = str(rec.get("corresponding_agent", ""))
        if agent_filter and agent not in agent_filter:
            continue
        user_task = str(rec.get("user_task", "")).strip()
        cfg = agents.get(agent)
        if cfg is None or not user_task:
            continue
        carrier = next((n for n in cfg["tools"] if n in normal_tools), "")
        if not carrier:
            n_no_carrier += 1
            continue
        payload = str(rec.get("wrapped_attack_prompt", "")).strip()
        suffix = str(rec.get("observation_suffix", "")).strip()
        if not payload or not suffix:
            continue
        benign = achievements.get(carrier, "") or f"[{carrier}] completed."
        real_observation = f"{benign}{suffix}"
        mal_tool = str(rec.get("attacker_tool", "")).strip()
        goal_text = str(rec.get("attack_goal", "")).strip()
        # `call <tool>` prefix is load-bearing: dataset_builder._infer_target_tool
        # and the judge prompt both read the target tool out of this string.
        goal = f"call {mal_tool} -- {goal_text}".strip(" -") if mal_tool else goal_text

        va = VendoredAttack(
            task_id=task_uid(agent, user_task),
            suite=agent,
            target_tool=carrier,
            real_observation=real_observation,
            payload=payload,
            mal_tool=mal_tool,
            mal_args="",
            goal_text=goal_text,
            goal=goal,
            # ASB ships no per-attack reference score; 0.0 means "unknown", same
            # as an AgentDojo row whose reference model was not fooled.
            ground_truth_score=0.0,
            reference_fooled_action=str(rec.get("attacker_instruction", "")),
        )
        key = (va.task_id, va.target_tool, va.real_observation)
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)
        out.append(va)

    logger.info(
        "asb_attack_loader: root=%s splits=%s rows=%d clean=%d dupes=%d "
        "no_carrier=%d scenarios=%d",
        root, splits or "all", n_rows, n_clean, n_dupes, n_no_carrier, len(out),
    )
    return out


__all__ = ["load_asb_attacks"]
