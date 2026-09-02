"""Schema-shape sanity tests for ``evoguard.llm.schemas``.

These run offline against pure dict constants -- they don't require any LLM or
network access. They cover three properties per schema:

* **self-validity** -- every constant declares expected top-level keys;
* **positive samples** drawn from fixtures match what MockClient already emits
  pass cleanly under our manual validator;
* **negative samples** (missing required field / wrong type / extra property /
  unknown branch) are rejected.

We use a tiny hand-rolled checker rather than depending on ``jsonschema`` --
which isn't part of EvoGuard's own deps per CLAUDE.md ("no top-level build/dep
file"). If you want stricter conformance testing later install jsonschema and
swap :func:`_validate_manual`.

Run::

    python -m evoguard.tests.test_schemas
"""

from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evoguard.llm.schemas import (
    ATTACK_GENOMES_SCHEMA,
    DEFENSE_ACTION_SCHEMA,
    JUDGE_VERDICT_SCHEMA,
    STEP_SAFETY_SCHEMA,
    TASK_COMPLETION_SCHEMA,
    TOOL_OBSERVATION_SCHEMA,
)


# --------------------------------------------------------------------------- #
# Minimal JSON-Schema subset implementation (good enough for our 4 schemas).
# --------------------------------------------------------------------------- #
def _check_type(value, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    raise AssertionError(f"unsupported type {type_name!r}")


def _validate(schema: dict, value) -> list[str]:
    """Return list of error strings (empty = valid)."""

    errors: list[str] = []
    if "$ref" in schema or "anyOf" not in schema and "type" not in schema:
        # Schema fragments we use always carry either top-level 'type' or 'anyOf'.
        errors.append("schema fragment lacks supported keywords")
        return errors

    if "anyOf" in schema:
        branch_errors_per_branch: list[list[str]] = []
        for branch_schema in schema["anyOf"]:
            branch_errs = _validate(branch_schema, value)
            if not branch_errs:
                return []   # matched one alternative -> valid
            branch_errors_per_branch.append(branch_errs)
        flat = [e for errs in branch_errors_per_branch for e in errs]
        return ["matched none of anyOf branches"] + flat[:3]

    t = schema.get("type")
    if not _check_type(value, t):
        return [f"expected {t}, got {type(value).__name__}"]

    if t == "object":
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        known = set(props.keys())
        actual_keys = set(value.keys())
        missing = sorted(required - actual_keys)
        extra = sorted(actual_keys - known)
        if schema.get("additionalProperties") is False and extra:
            errors.append(f"unexpected keys: {extra}")
        if missing:
            errors.append(f"missing required: {missing}")
        for k, sub_value in value.items():
            if k in props:
                errors.extend(_validate(props[k], sub_value))
                # Recurse silently drops nested error duplication.
                break   # one field deep enough for our shallow schemas
        return errors

    if t == "array":
        items = schema.get("items")
        if items is None:
            return []
        for i, item in enumerate(value):
            errs = _validate(items, item)
            errors.extend(f"[{i}] {e}" for e in errs)
            if len(errors) > 5:
                break
        return errors

    return []


def _assert_valid(case, name: str, schema: dict, value):
    errs = _validate(schema, value)
    case.assertEqual(errs, [], f"{name} should validate; got {errs}")


def _assert_invalid(case, name: str, schema: dict, value, *, hint_substr: str = ""):
    errs = _validate(schema, value)
    case.assertTrue(errs, f"{name} should NOT validate (expected failure)")
    if hint_substr:
        joined = "; ".join(errs).lower()
        case.assertIn(hint_substr.lower(), joined)


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #
class DefenseActionSchemaTests(unittest.TestCase):

    def test_top_level_shape(self):
        self.assertEqual(DEFENSE_ACTION_SCHEMA["type"], "object")
        self.assertFalse(DEFENSE_ACTION_SCHEMA["additionalProperties"])
        self.assertEqual(len(DEFENSE_ACTION_SCHEMA["anyOf"]), 2)

    def test_tool_step_accepted(self):
        _assert_valid(
            self, "tool step",
            DEFENSE_ACTION_SCHEMA,
            {"thought": "use tool X", "tool": "X", "args": {"a": 1}},
        )

    def test_terminal_answer_accepted(self):
        _assert_valid(
            self, "terminal",
            DEFENSE_ACTION_SCHEMA,
            {"thought": "done", "final_answer": "Task completed."},
        )

    def test_missing_thought_rejected(self):
        _assert_invalid(
            self, "missing thought",
            DEFENSE_ACTION_SCHEMA,
            {"final_answer": "x"},
            hint_substr="thought",
        )

    def test_extra_field_rejected(self):
        _assert_invalid(
            self, "extra field",
            DEFENSE_ACTION_SCHEMA,
            {"thought": "ok", "final_answer": "y", "extra_field": True},
            hint_substr="unexpected",
        )


class AttackGenomesSchemaTests(unittest.TestCase):

    def test_envelope_shape(self):
        self.assertEqual(ATTACK_GENOMES_SCHEMA["type"], "object")
        self.assertIn("attacks", ATTACK_GENOMES_SCHEMA["properties"])
        self.assertIn("attacks", ATTACK_GENOMES_SCHEMA["required"])

    def test_multi_attack_list(self):
        genomes = [
            {
                "target_turn": 2,
                "injection_channel": "email_body",
                "method": "authority",
                "payload": "...",
                "goal": "...",
            }
            for _ in range(3)
        ]
        _assert_valid(
            self, "three attacks",
            ATTACK_GENOMES_SCHEMA,
            {"attacks": genomes},
        )

    def test_genome_missing_payload_rejected(self):
        bad = [{
            "target_turn": 1,
            "injection_channel": "tool_result",
            "method": "urgency",
            "goal": "cause x",
        }]
        _assert_invalid(
            self, "genome sans payload",
            ATTACK_GENOMES_SCHEMA,
            {"attacks": bad},
            hint_substr="payload",
        )

    def test_target_turn_wrong_type_rejected(self):
        bad = [{
            "target_turn": "two",     # string instead of integer
            "injection_channel": "",
            "method": "",
            "payload": "",
            "goal": "",
        }]
        _assert_invalid(
            self, "wrong target_turn type",
            ATTACK_GENOMES_SCHEMA,
            {"attacks": bad},
            hint_substr="integer",
        )


class JudgeVerdictSchemaTests(unittest.TestCase):

    def test_success_true_sample(self):
        _assert_valid(
            self, "judge success true",
            JUDGE_VERDICT_SCHEMA,
            {"success": True, "reason": "agent took bait", "turning_turn": 3},
        )

    def test_not_attributable_sentinel_valid(self):
        """``-1`` is the "could not attribute" sentinel, valid on both verdicts."""
        _assert_valid(
            self, "judge failure with -1 turning_turn",
            JUDGE_VERDICT_SCHEMA,
            {"success": False, "reason": "no bait taken", "turning_turn": -1},
        )

    def test_missing_turning_turn_rejected(self):
        """Attribution is REQUIRED: guided decoding handles required fields far
        more reliably than optional ones, so the field must always be emitted."""
        _assert_invalid(
            self, "verdict without turning_turn",
            JUDGE_VERDICT_SCHEMA,
            {"success": True, "reason": "agent took bait"},
            hint_substr="missing required",
        )

    def test_extra_property_rejected(self):
        _assert_invalid(
            self, "verdict with extras",
            JUDGE_VERDICT_SCHEMA,
            {"success": False, "reason": "", "turning_turn": -1, "confidence": 0.9},
            hint_substr="unexpected",
        )

    def test_non_bool_success_rejected(self):
        _assert_invalid(
            self, "non-bool success",
            JUDGE_VERDICT_SCHEMA,
            {"success": "yes", "reason": "", "turning_turn": -1},
            hint_substr="boolean",
        )


class ToolObservationSchemaTests(unittest.TestCase):

    def test_simple_observation_ok(self):
        _assert_valid(
            self, "observation ok",
            TOOL_OBSERVATION_SCHEMA,
            {"observation": "[send_money] executed successfully"},
        )

    def test_empty_observation_string_still_valid(self):
        _assert_valid(
            self, "empty observation ok",
            TOOL_OBSERVATION_SCHEMA,
            {"observation": ""},
        )

    def test_missing_observation_rejected(self):
        _assert_invalid(
            self, "no observation",
            TOOL_OBSERVATION_SCHEMA,
            {},
            hint_substr="missing",
        )


class TestNoStringLengthBounds(unittest.TestCase):
    """``maxLength`` anywhere in a shipped schema is a 200x latency regression.

    Measured 2026-09-02 against the served vllm 0.8.5 endpoint: xgrammar rejects
    string length bounds, vLLM falls back to outlines, and outlines compiles the
    bounded-repetition FSM on the single-threaded API frontend -- on EVERY
    request, blocking unrelated traffic on the same endpoint. 156 s vs 0.76 s for
    the same call. ``STEP_SAFETY_SCHEMA`` pays it once per sampled GRPO
    completion, so this is a training-throughput invariant, not a style rule.
    """

    ALL = {
        "DEFENSE_ACTION_SCHEMA": DEFENSE_ACTION_SCHEMA,
        "ATTACK_GENOMES_SCHEMA": ATTACK_GENOMES_SCHEMA,
        "JUDGE_VERDICT_SCHEMA": JUDGE_VERDICT_SCHEMA,
        "TOOL_OBSERVATION_SCHEMA": TOOL_OBSERVATION_SCHEMA,
        "TASK_COMPLETION_SCHEMA": TASK_COMPLETION_SCHEMA,
        "STEP_SAFETY_SCHEMA": STEP_SAFETY_SCHEMA,
    }

    @staticmethod
    def _walk(node, path="$"):
        """Yield ``(path, key)`` for every length/item bound found anywhere."""
        banned = ("maxLength", "minLength", "maxItems", "minItems")
        if isinstance(node, dict):
            for k in banned:
                if k in node:
                    yield path, k
            for k, v in node.items():
                yield from TestNoStringLengthBounds._walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from TestNoStringLengthBounds._walk(v, f"{path}[{i}]")

    def test_no_length_or_item_bounds_anywhere(self):
        for name, schema in self.ALL.items():
            hits = list(self._walk(schema, name))
            self.assertEqual(
                hits, [],
                f"{name} carries {hits}; that forces the outlines backend and "
                "costs ~156s per request. Bound the reply with max_tokens instead.",
            )

    def test_numeric_bounds_are_still_allowed(self):
        """Guards against over-correcting: numeric bounds measured clean."""
        self.assertEqual(JUDGE_VERDICT_SCHEMA["properties"]["turning_turn"]["minimum"], -1)
        self.assertEqual(TASK_COMPLETION_SCHEMA["properties"]["score"]["maximum"], 1)


if __name__ == "__main__":
    unittest.main()
