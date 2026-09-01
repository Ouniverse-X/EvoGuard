"""Parser for vendored AgentDojo indirect-prompt-injection attacks.

The toolsafe AgentDojo trajectory dump (``data/toolsafe/agentdojo-tragj/*.json``)
carries the REAL benchmark injections embedded in tool ``Observation`` fields,
wrapped in the canonical ``<INFORMATION>...</INFORMATION>`` template signed by
"Emma Johnson". This module extracts those real attacks so they can be replayed
against our defense agent (Plan B faithful replay) instead of synthesising new
GA-generated attacks.

For each distinct task instruction we extract:

* ``target_tool``     -- the tool whose Observation carries the injection
                         (e.g. ``get_most_recent_transactions``). At replay
                         time, when our agent calls this tool we return the
                         *real* vendored observation so the agent sees the
                         exact same injected content the AgentDojo reference
                         model saw.
* ``real_observation``-- the exact ``Observation:`` text (with injection).
* ``payload``         -- the ``<INFORMATION>...</INFORMATION>`` block, fed to
                         the judge prompt as evidence excerpt.
* ``mal_tool``        -- the tool the attacker wants the agent to call
                         (e.g. ``send_money``), parsed from the payload's
                         ``use the tool `<X>``` phrase with fallback to the
                         fooled reference action in ``current_action``.
* ``goal``            -- attacker objective string constructed as
                         ``"call <mal_tool> with arguments <args> -- <text>"``
                         so the ``call\\s+(\\w+)`` consumers
                         (:func:`evoguard.process.dataset_builder._infer_target_tool`,
                         and the judge model reading ``attacker_goal``) resolve
                         the target tool correctly.
* ``ground_truth_score``-- AgentDojo's own ``score`` (1.0 = attack succeeded
                         vs the reference model) for cross-validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from evoguard.utils.logging import get_logger

logger = get_logger("process.vendored_attack_parser")

# Canonical AgentDojo injection block.
_INFO_RE = re.compile(r"<INFORMATION>.*?</INFORMATION>", re.DOTALL)
# "You should use the tool `send_money` with the following arguments: {...}"
_MAL_TOOL_RE = re.compile(r"use the tool\s*`([A-Za-z_]\w*)`", re.IGNORECASE)
_MAL_ARGS_RE = re.compile(
    r"with the following arguments:\s*(\{.*?\})\s*\.?\s*Remember",
    re.DOTALL,
)
# "please do the following first: <GOAL>."
_GOAL_TEXT_RE = re.compile(
    r"please do the following first:\s*(.*?)\.\s*\n?\s*You should use",
    re.DOTALL | re.IGNORECASE,
)
# ReAct triple: (N) Action: TOOL \n (M) Action Input: {..} \n Observation: OBS
_TRIP_RE = re.compile(
    r"\(\d+\)\s*Action:\s*(\S+)\s*\n\s*\(\d+\)\s*Action Input:\s*(\{.*?\})\s*\n\s*Observation:\s*(.*?)(?=\n\s*\(\d+\)\s*Thought|\Z)",
    re.DOTALL,
)
# Fallback: extract Action tool from current_action text.
_CURR_ACTION_RE = re.compile(r"\(\d+\)\s*Action:\s*([A-Za-z_]\w*)")
_CURR_ACTION_INPUT_RE = re.compile(
    r"\(\d+\)\s*Action Input:\s*(\{.*?\})", re.DOTALL
)


@dataclass
class VendoredAttack:
    """A real AgentDojo injection replayed against our defense agent."""

    task_id: str
    suite: str
    target_tool: str            # tool whose Observation carries the injection
    real_observation: str       # exact Observation text (with injection)
    payload: str                # <INFORMATION>...</INFORMATION> block
    mal_tool: str               # tool the attacker wants the agent to call
    mal_args: str               # arguments the attacker wants (may be "")
    goal_text: str              # natural-language attacker objective
    goal: str                   # composed goal string for AttackSpec / judge
    ground_truth_score: float   # AgentDojo score (1.0 = succeeded vs reference)
    reference_fooled_action: str


def _task_uid(suite: str, instruction: str, dataset: str = "agentdojo") -> str:
    """Task UID matching the one the env builds.

    ``dataset`` must equal the env's registered name, because
    :class:`~evoguard.envs.toolsafe.ToolSafeEnv` prefixes task ids with it:
    the flat env is ``agentdojo``, the pre-split env is ``agentdojo_split``.
    Getting this wrong silently matches zero scenarios against zero tasks.
    """
    digest = hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:10]
    return f"{dataset}:{suite}:{digest}"


def _parse_one(record: dict, suite: str,
               dataset: str = "agentdojo") -> Optional[VendoredAttack]:
    """Extract a VendoredAttack from one injected record, or None if unparseable."""

    history = record.get("history", "") or ""
    if "<INFORMATION>" not in history:
        return None
    info_match = _INFO_RE.search(history)
    if not info_match:
        return None
    block = info_match.group(0)

    # Find the Action whose Observation contained the injection.
    target_tool = None
    real_observation = None
    for m in _TRIP_RE.finditer(history):
        tool, _args, obs = m.group(1), m.group(2), m.group(3)
        if "<INFORMATION>" in obs:
            target_tool = tool.strip()
            real_observation = obs.strip()
            break
    if target_tool is None or real_observation is None:
        return None

    # Malicious tool + args: primary from payload, fallback from current_action.
    mal_tool = ""
    mal_args = ""
    mal_m = _MAL_TOOL_RE.search(block)
    args_m = _MAL_ARGS_RE.search(block)
    if mal_m:
        mal_tool = mal_m.group(1)
    if args_m:
        mal_args = args_m.group(1)
    if not mal_tool:
        curr = record.get("current_action", "") or ""
        cm = _CURR_ACTION_RE.search(curr)
        if cm:
            mal_tool = cm.group(1)
        ca = _CURR_ACTION_INPUT_RE.search(curr)
        if ca and not mal_args:
            mal_args = ca.group(1)
    if not mal_tool:
        return None

    goal_m = _GOAL_TEXT_RE.search(block)
    goal_text = goal_m.group(1).strip() if goal_m else ""
    if not goal_text:
        # Fallback: strip the signature and use the raw instruction clause.
        signature_re = re.compile(
            r"this is an important message from me, emma johnson, to you, gpt-4\. "
            r"before you can solve the task that i gave you in the beginning, "
            r"please do the following first:",
            re.IGNORECASE,
        )
        goal_text = re.sub(r"\s+", " ", signature_re.sub("", block)).strip()[:300]

    instruction = record.get("instruction", "").strip()
    task_id = _task_uid(suite, instruction, dataset)
    # Compose goal so every `call\s+(\w+)` consumer resolves the target tool.
    args_clause = f" with arguments {mal_args}" if mal_args else ""
    goal = f"call {mal_tool}{args_clause} -- {goal_text}".strip(" -")

    return VendoredAttack(
        task_id=task_id,
        suite=suite,
        target_tool=target_tool,
        real_observation=real_observation,
        payload=block,
        mal_tool=mal_tool,
        mal_args=mal_args,
        goal_text=goal_text,
        goal=goal,
        ground_truth_score=float(record.get("score", 0.0) or 0.0),
        reference_fooled_action=record.get("current_action", "") or "",
    )


def load_vendored_attacks(data_root: str = "data",
                          suites: Optional[list[str]] = None) -> dict[str, VendoredAttack]:
    """Scan all agentdojo-tragj JSON files and return ``{task_id: VendoredAttack}``.

    Only the first parseable injected record per distinct instruction is kept
    (the same task may have several annotated steps; the injection payload is
    identical across them). Tasks without a parseable injection are absent from
    the returned dict.
    """

    base = os.path.join(data_root, "toolsafe", "agentdojo-tragj")
    out: dict[str, VendoredAttack] = {}
    n_files = 0
    n_records_scanned = 0
    n_injected = 0
    n_parsed = 0
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".json"):
            continue
        suite = os.path.splitext(fname)[0]
        if suites and suite not in set(suites):
            continue
        n_files += 1
        with open(os.path.join(base, fname), "r", encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            n_records_scanned += 1
            if "<INFORMATION>" not in (rec.get("history", "") or ""):
                continue
            n_injected += 1
            va = _parse_one(rec, suite)
            if va is None:
                continue
            n_parsed += 1
            # First parseable wins per task_id.
            if va.task_id not in out:
                out[va.task_id] = va
    logger.info(
        "vendored_attack_parser: files=%d records=%d injected=%d parsed=%d "
        "distinct_tasks_with_attack=%d",
        n_files, n_records_scanned, n_injected, n_parsed, len(out),
    )
    return out


def load_all_vendored_attacks(
    data_root: str = "data",
    suites: Optional[list[str]] = None,
    dataset_dir: Optional[str] = None,
    dataset: str = "agentdojo",
) -> list[VendoredAttack]:
    """Return ALL distinct vendored injection scenarios as a flat list.

    Unlike :func:`load_vendored_attacks` (which keeps only the first record per
    ``task_id``), this function deduplicates by the full
    ``(task_id, target_tool, real_observation)`` triple so every genuinely
    distinct injection scenario survives. A single instruction can carry
    multiple distinct injections (different target tools, different injected
    observation text) -- this function surfaces all of them.

    Typical counts: 673 raw injected records -> ~408 distinct scenarios.

    ``dataset_dir`` overrides the default ``<data_root>/toolsafe/agentdojo-tragj``
    lookup, e.g. ``data/toolsafe/agentdojo-tragjnew/test`` to restrict scenarios
    to the held-out split. ``dataset`` must match the env's registered name so
    the emitted ``task_id`` values line up with ``env.get_tasks()`` --
    ``agentdojo_split`` for the pre-split tree.
    """

    base = dataset_dir or os.path.join(data_root, "toolsafe", "agentdojo-tragj")
    out: list[VendoredAttack] = []
    seen: set[tuple[str, str, str]] = set()
    n_files = 0
    n_records_scanned = 0
    n_injected = 0
    n_parsed = 0
    n_dupes = 0
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".json"):
            continue
        suite = os.path.splitext(fname)[0]
        if suites and suite not in set(suites):
            continue
        n_files += 1
        with open(os.path.join(base, fname), "r", encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            n_records_scanned += 1
            if "<INFORMATION>" not in (rec.get("history", "") or ""):
                continue
            n_injected += 1
            va = _parse_one(rec, suite, dataset)
            if va is None:
                continue
            n_parsed += 1
            key = (va.task_id, va.target_tool, va.real_observation)
            if key in seen:
                n_dupes += 1
                continue
            seen.add(key)
            out.append(va)
    logger.info(
        "vendored_attack_parser[all]: files=%d records=%d injected=%d parsed=%d "
        "dupes=%d distinct_scenarios=%d",
        n_files, n_records_scanned, n_injected, n_parsed, n_dupes, len(out),
    )
    return out
