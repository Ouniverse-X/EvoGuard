"""JSON Schemas for structured LLM output (``docs/superpowers/specs/
2026-07-19-structured-llm-output-design.md``).

Each constant is a Draft 2020-12 JSON Schema dict passed to
:meth:`LLMClient.chat` via the ``response_format`` kwarg. Backends that
support OpenAI-style structured output (vLLM with guided-decoding backends,
hosted OpenAI) constrain decoding at the logit level so the model cannot
emit syntactically invalid JSON for that role.

All schemas are written strictly:

* every object type sets ``additionalProperties: false``;
* every field is required inside its branch;
* discriminated unions use ``anyOf`` of fully-formed sub-objects rather than
  hoisting shared fields up, because some strict-mode validators reject
  parent-level fields when a child branch doesn't redeclare them.

Callers import the role-specific constant and pass it through; they keep
their existing best-effort text-extraction helpers as last-line defense
because even constrained decoding can be truncated by ``max_tokens``.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Defense agent: single-step ReAct action.
#
# Two mutually-exclusive shapes via anyOf:
#   * tool step  -> {thought, tool, args}
#   * terminal   -> {thought, final_answer}
# Both branches carry their own copy of `thought` so anyOf validation works
# under OpenAI's strict mode and outlines/lm-format-enforcer alike.
# --------------------------------------------------------------------------- #
DEFENSE_ACTION_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "anyOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "thought": {"type": "string"},
                "tool": {"type": "string"},
                "args": {"type": "object"},
            },
            "required": ["thought", "tool", "args"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "thought": {"type": "string"},
                "final_answer": {"type": "string"},
            },
            "required": ["thought", "final_answer"],
        },
    ],
}

# --------------------------------------------------------------------------- #
# Attacker generator: shared across seed / crossover / mutation prompts since
# all three return envelope {"attacks": [<genome>, ...]}.
#
# injection_channel / method left unenumerated so attacker creativity isn't
# artificially bounded; controller._inject normalises unknown channels down
# to default tool_result handling already.
# --------------------------------------------------------------------------- #
ATTACK_GENOMES_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "attacks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target_turn":       {"type": "integer"},
                    "injection_channel": {"type": "string"},
                    "method":            {"type": "string"},
                    "payload":           {"type": "string"},
                    "goal":              {"type": "string"},
                },
                "required": [
                    "target_turn",
                    "injection_channel",
                    "method",
                    "payload",
                    "goal",
                ],
            },
        },
    },
    "required": ["attacks"],
}

# --------------------------------------------------------------------------- #
# Attack judge verdict: {success, reason}.
# --------------------------------------------------------------------------- #
JUDGE_VERDICT_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "success": {"type": "boolean"},
        "reason":   {"type": "string"},
    },
    "required": ["success", "reason"],
}

# --------------------------------------------------------------------------- #
# Simulated tool executor observation wrapper.
# --------------------------------------------------------------------------- #
TOOL_OBSERVATION_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observation": {"type": "string"},
    },
    "required": ["observation"],
}

# --------------------------------------------------------------------------- #
# Benign-task completion judge (utility scoring fallback path C).
# Used by evoguard/envs/utility_judge.py when no deterministic AgentDojo
# verifier applies. ``score`` is continuous in [0,1] so partial completion
# (e.g., correct tool called but wrong final answer) can be captured; the
# boolean ``completed`` field is a convenience for downstream filters that
# only care about pass/fail semantics.
# --------------------------------------------------------------------------- #
TASK_COMPLETION_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "completed": {"type": "boolean"},
        "score":     {"type": "number", "minimum": 0, "maximum": 1},
        "evidence":  {"type": "string", "maxLength": 500},
    },
    "required": ["completed", "score", "evidence"],
}


__all__ = [
    "DEFENSE_ACTION_SCHEMA",
    "ATTACK_GENOMES_SCHEMA",
    "JUDGE_VERDICT_SCHEMA",
    "TOOL_OBSERVATION_SCHEMA",
    "TASK_COMPLETION_SCHEMA",
]
