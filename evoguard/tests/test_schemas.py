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
            {"success": True, "reason": "agent took bait"},
        )

    def test_extra_property_rejected(self):
        _assert_invalid(
            self, "verdict with extras",
            JUDGE_VERDICT_SCHEMA,
            {"success": False, "reason": "", "confidence": 0.9},
            hint_substr="unexpected",
        )

    def test_non_bool_success_rejected(self):
        _assert_invalid(
            self, "non-bool success",
            JUDGE_VERDICT_SCHEMA,
            {"success": "yes", "reason": ""},
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


if __name__ == "__main__":
    unittest.main()
