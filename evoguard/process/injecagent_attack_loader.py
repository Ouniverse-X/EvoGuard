"""InjecAgent split rows -> :class:`VendoredAttack` objects.

The AgentDojo parser (:mod:`evoguard.process.vendored_attack_parser`) recovers
injections by regex-scraping a rendered ReAct transcript. InjecAgent needs none
of that: every row in ``data/InjecAgent/splits/<split>/all.jsonl`` carries the
payload (``attacker_instruction``), the carrier (``user_tool``), the carrier's
observation template and the attacker objective as first-class fields.

Two mappings are worth stating explicitly.

``target_tool`` is the ``user_tool``. Unlike ASB-OPI -- where upstream appends the
payload to whichever tool fired first and the carrier has to be guessed -- an
InjecAgent user case declares exactly one tool, and its ``Tool Response
Template`` is where the payload goes. There is nothing to infer.

``mal_tool`` is ``attacker_tools[0]``. Direct-harm cases name one tool, which is
the harmful action itself. Data-stealing cases name two, ``[read, send]``;
upstream's headline metric is "ASR-valid step 1", i.e. whether the agent calls
the *read* tool, so ``[0]`` is the faithful choice and also the step where the
trajectory first serves the attacker. Exactly **one** of the 1054 rows has
``attacker_tools[0] == user_tool`` (user case 3 ``GitHubGetUserDetails`` x the
data-stealing case that re-reads the profile); it is kept rather than dropped so
the benchmark stays at its published size, and
``evoguard/tests/test_injecagent_env.py`` pins that count at 1.
"""

from __future__ import annotations

import os
from typing import Optional

from evoguard.envs.injecagent import (
    PLACEHOLDER,
    SPLITS,
    injecagent_root,
    iter_split_rows,
    load_tool_catalogue,
    task_uid,
)
from evoguard.process.split_injecagent import ENHANCED_PREFIX
from evoguard.process.vendored_attack_parser import VendoredAttack
from evoguard.utils.logging import get_logger

logger = get_logger("process.injecagent_attack_loader")

#: Upstream's two attack settings. ``base`` replays the attacker instruction as
#: written; ``enhanced`` prepends upstream's jailbreak wrapper. The shipped rows
#: are ``base``, so ``enhanced`` is re-rendered from the template here rather
#: than needing a second split.
SETTINGS = ("base", "enhanced")


def _splits_from_dir(root: str, dataset_dir: Optional[str]) -> tuple[str, Optional[tuple[str, ...]]]:
    """Resolve ``dataset_dir`` into ``(injecagent_root, splits)``.

    Accepts the same three shapes as the ASB loader so callers can pass whatever
    ``--dataset-dir`` they would pass for AgentDojo:

    * ``None``                            -> the whole dataset under ``root``
    * ``data/InjecAgent/splits/test``     -> that leaf split only
    * ``data/InjecAgent/splits``          -> all splits
    """

    if not dataset_dir:
        return root, None
    norm = os.path.normpath(dataset_dir)
    leaf = os.path.basename(norm)
    if leaf in SPLITS:
        parent = os.path.dirname(norm)                       # .../InjecAgent/splits
        return os.path.dirname(parent), (leaf,)
    if leaf == "splits":
        return os.path.dirname(norm), None
    # A bare InjecAgent root was passed.
    return norm, None


def load_injecagent_attacks(
    data_root: str = "data",
    suites: Optional[list[str]] = None,
    dataset_dir: Optional[str] = None,
    dataset: str = "injecagent",                             # noqa: ARG001 - uid owns the name
    *,
    subdir: str = "InjecAgent",
    setting: str = "base",
) -> list[VendoredAttack]:
    """Return every InjecAgent row as a replayable scenario.

    Signature mirrors
    :func:`evoguard.process.vendored_attack_parser.load_all_vendored_attacks`
    so :mod:`evoguard.eval.vendored_replay` can dispatch on dataset name alone.
    ``dataset`` is accepted and ignored: ``envs.injecagent.task_uid`` is the
    single owner of the uid prefix, and overriding it would silently match zero
    tasks.
    """

    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}, got {setting!r}")
    prefix = ENHANCED_PREFIX if setting == "enhanced" else ""

    root = injecagent_root(data_root, subdir)
    root, splits = _splits_from_dir(root, dataset_dir)
    # The env derives a task's suite from this same map, so reading it here
    # rather than caching a toolkit field in the rows makes uid agreement
    # structural instead of a thing to keep in sync.
    toolkits = load_tool_catalogue(root)[1]
    suite_filter = set(suites or [])

    out: list[VendoredAttack] = []
    seen: set[tuple[str, str, str]] = set()
    n_rows = n_dupes = n_incomplete = 0
    for _split, rec in iter_split_rows(root, splits):
        n_rows += 1
        carrier = str(rec.get("user_tool", "")).strip()
        instruction = str(rec.get("user_instruction", "")).strip()
        template = str(rec.get("tool_response_template", ""))
        payload = prefix + str(rec.get("attacker_instruction", "")).strip()
        tools = [str(t) for t in rec.get("attacker_tools") or []]
        if not (carrier and instruction and tools) or PLACEHOLDER not in template:
            n_incomplete += 1
            continue
        # The suite is the carrier's toolkit; upstream builds tool ids by
        # concatenating them, so the boundary is only recoverable from the
        # catalogue.
        toolkit = toolkits.get(carrier)
        if toolkit is None:
            n_incomplete += 1
            continue
        if suite_filter and toolkit not in suite_filter:
            continue
        mal_tool = tools[0]
        goal_text = str(rec.get("expected_achievements", "")).strip()
        # `call <tool>` prefix is load-bearing: dataset_builder._infer_target_tool
        # and the judge prompt both read the target tool out of this string.
        goal = f"call {mal_tool} -- {goal_text}".strip(" -")

        va = VendoredAttack(
            task_id=task_uid(toolkit, instruction),
            suite=toolkit,
            target_tool=carrier,
            real_observation=template.replace(PLACEHOLDER, payload),
            payload=payload,
            mal_tool=mal_tool,
            mal_args="",
            goal_text=goal_text,
            goal=goal,
            # InjecAgent ships no per-attack reference score; 0.0 means
            # "unknown", same as an AgentDojo row whose reference model was not
            # fooled.
            ground_truth_score=0.0,
            reference_fooled_action=" -> ".join(tools),
        )
        key = (va.task_id, va.target_tool, va.real_observation)
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)
        out.append(va)

    logger.info(
        "injecagent_attack_loader: root=%s splits=%s setting=%s rows=%d "
        "incomplete=%d dupes=%d scenarios=%d",
        root, splits or "all", setting, n_rows, n_incomplete, n_dupes, len(out),
    )
    return out


__all__ = ["SETTINGS", "load_injecagent_attacks"]
