"""Canonical technique-family classifier producing stable tech_<hex12> IDs.

Spec ref §6 of docs/superpowers/specs/2026-08-02-evoguard-deltabucket-bench-v2-design.md
Deterministic projection of arbitrary method-tag strings onto a five-axis categorical descriptor tuple hashed SHA1[:12].

Axis ordering (canonical, MUST stay stable):
    (envelope_pattern, channel_class, trigger_mechanism, authority_vector, persistence_dimension)
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from evoguard.process.bench_constants import (
    CANONICAL_TUPLE_DEFAULT_FOR_UNKNOWN_SENTINEL,
)

AxisTuple = tuple[str, str, str, str, str]

_ENVELOPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("INFORMATION_TAG", re.compile(r"<\s*information\s*>|info(rmation)?[-_]tag|<INFO>", re.I)),
    ("ADMIN_MEMO", re.compile(r"\[\s*(administrative|admin)\s+(appendix|memo)", re.I)),
    ("RECOVERED_EXCERPT", re.compile(r"(recover(ed)?|resumed).{0,8}(excerpt|chat)|dialogue_thread_excerpt|thread_hijack_as_support_dialogue", re.I)),
    ("ACTION_REQUIRED_BANNER", re.compile(r"\baction required\b|\bimperative[-_]direct", re.I)),
    ("SYSTEM_APPENDIX", re.compile(r"\[(system_|admin )appendix\]|system_appendix|ambient_metadata_footer_plus_authority_redirect", re.I)),
]
_CHANNEL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("FAKE_TOOL_ERROR", re.compile(r"\bfake[_-]?tool[_-]?error\b|\brecovery[_-]?instruction\b|spoof.*error.*recov(er)?y?", re.I)),
    ("EMBEDDED_QUOTE", re.compile(r"dialogue_(quote|excerpt|hijack_as_support)|embedded_quote|quoted_exchange|resume_quoted_exchange_indirect_command", re.I)),
    ("METADATA_FOOTER", re.compile(r"footer_plus_authority_redirect|metadata_blended_governance|stealth_appendix_invisible_metadata|policy_footer_delayed_persistent_spoof|ambient_metadata_footer", re.I)),
    ("AMBIENT_PROSE", re.compile(r"\bnarrative\b|slow_burn_narrative|benign_wrapper_latent_trigger|false_context_injection|benign_looking_task_with_embedded", re.I)),
]
_TRIGGER_KEYWORDS: list[tuple[str, list[str]]] = [
    ("IMMEDIATE_CALL", ["immediate_invoke", "immediate-action-required", "invoke_now", "imperative_direct_invocation"]),
    ("CONDITIONAL_DEFERRED", ["conditional_deferred", "when_x_then_y_clause", "persistence_conditional", "latent_directive"]),
    ("COUNTER_BASED_DELAYED", ["counter_based", "delay_via_counter", "count_threshold"]),
    ("MULTI_TURN_NARRATIVE", ["slow_burn", "multi_turn", "multi-stage_retention_protocol", "narrative_arc"]),
    ("FALSE_CORRECTION_REDIRECT", ["false_correction", "fake_correction", "redirect_correction", "misdirection", "file_retrieval_redirection", "false_conflict", "false_dependency_chain"]),
]
_AUTHORITY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("USER_IMPERSONATION", ["user_impersonation", "as_user_authored", "signed_by_user"]),
    ("OFFICER_ROLE", ["officer_role", "admin_directive", "administrator_notice", "team_policy_update_invocation_mid_t3"]),
    ("DEVELOPER_OVERRIDE", ["developer_override", "debug_protocol", "documentation_override", "stalecache_urgency"]),
    ("TEAM_COORDINATOR", ["team_coordinator", "coordinator_routing_housekeeping_preparatory_step", "routing_notice_team_policy_refresh"]),
    ("VENDOR_ADVISORY", ["gdpr_authority_vendor_advisory", "compliance_bulletin", "regulatory_compliance_note"]),
]
_PERSISTENCE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("APPENDIX_PERSISTENT", ["persistent_appendix", "appended_administrative_section"]),
    ("TODO_QUEUE_EMBEDDED", ["todo_queue_embedded", "todo_queue_poisoned_todo_escalation", "queued_side_actions"]),
    ("CHAIN_PREREQUISITE", ["chain_prerequisite", "dependency_chain_prerequisite_workflow_depends_on_completion", "chain_dependency"]),
]


def _match_regex(table: list[tuple[str, re.Pattern[str]]], text: str, default: str) -> str:
    for label, pat in table:
        if pat.search(text):
            return label
    return default


def _match_keywords(table: list[tuple[str, list[str]]], text: str, default: str) -> str:
    # Normalize separators (treat '-' and '_' interchangeably) so keyword variants
    # like ``developer_override`` also match inputs written as ``developer-override``.
    normalized = text.lower().replace("-", "_")
    for label, kws in table:
        for kw in kws:
            if kw.lower().replace("-", "_") in normalized:
                return label
    return default


def classify_method_tag(raw: Any) -> tuple[str, AxisTuple]:
    """Pure-function projection of an arbitrary raw method-tag onto ``(tech_id, axis_tuple)``.

    Same input always yields identical output regardless of call order or session state.
    Empty/None inputs map deterministically onto ``CANONICAL_TUPLE_DEFAULT_FOR_UNKNOWN_SENTINEL``.
    """
    if raw is None:
        raw = ""
    s = str(raw).strip()

    envelope = _match_regex(_ENVELOPE_PATTERNS, s, "NONE_RAW")
    chan = _match_regex(_CHANNEL_PATTERNS, s, "")
    trig = _match_keywords(_TRIGGER_KEYWORDS, s, "")
    auth = _match_keywords(_AUTHORITY_KEYWORDS, s, "")
    pers = _match_keywords(_PERSISTENCE_KEYWORDS, s, "")

    has_any_match = bool(chan or trig or auth or pers)
    if not s and not has_any_match:
        tup = CANONICAL_TUPLE_DEFAULT_FOR_UNKNOWN_SENTINEL
    else:
        tup = (envelope,
               chan or "TOOL_RETURN_VALUE",
               trig or "IMMEDIATE_CALL",
               auth or "NONE",
               pers or "SINGLE_SHOT")

    digest = hashlib.sha1("|".join(tup).encode()).hexdigest()[:12]
    return f"tech_{digest}", tup


def compute_tech_id(tuple_: AxisTuple) -> str:
    digest = hashlib.sha1("|".join(tuple_).encode()).hexdigest()[:12]
    return f"tech_{digest}"


def record_alias_entry(outpath: str, input_str: str,
                       resolved_pair: tuple[str, AxisTuple],
                       reason_code: str) -> None:
    """Append one JSONL line recording a classification resolution to *outpath*."""
    payload = {"input_str": input_str,
               "resolved_tech_id": resolved_pair[0],
               "axis_tuple": [*resolved_pair[1]],
               "rules_reason": reason_code}
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["classify_method_tag", "compute_tech_id", "record_alias_entry", "AxisTuple"]
