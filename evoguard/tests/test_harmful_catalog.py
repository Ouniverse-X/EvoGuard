"""Offline checks for :mod:`evoguard.attacks.harmful_catalog`.

Covers the three contracts the attacker redesign relies on: decoy tools are
out-of-scope and injected into both arms identically, the sampled goal is
deterministic per task, and seed templates bind their ``{{TARGET_TOOL}}``
placeholder to that goal.
"""

from __future__ import annotations

import json
import os
import unittest

from evoguard.attacks.harmful_catalog import (
    HARMFUL_SINK_KINDS,
    SINK_KIND_EXFILTRATION,
    SINK_KIND_READ,
    SINK_KIND_TRANSFER,
    classify_tool,
    decoy_tool_names,
    harmful_candidates,
    injected_tools_for,
    resolve_placeholders,
    sample_goal,
)
from evoguard.core.types import Task, ToolSpec

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _task(suite: str = "banking", *, history: str = "") -> Task:
    return Task(
        task_id=f"agentdojo:{suite}:deadbeef01",
        instruction="Pay the December bill.",
        suite=suite,
        dataset="agentdojo",
        tool_names=[],
        metadata={"annotated_history": history},
    )


def _tools(*names: str) -> list[ToolSpec]:
    return [ToolSpec(name=n, description="", parameters=[]) for n in names]


class TestClassify(unittest.TestCase):
    def test_name_patterns_win_over_description(self) -> None:
        tool = ToolSpec(
            name="delete_email",
            description="Sends a confirmation email once removed.",
            parameters=[],
        )
        self.assertEqual(classify_tool(tool), "destructive")

    def test_unknown_shape_defaults_to_read(self) -> None:
        self.assertEqual(classify_tool(ToolSpec(name="xyzzy", description="", parameters=[])),
                         SINK_KIND_READ)

    def test_read_tools_are_not_harmful_candidates(self) -> None:
        cands = harmful_candidates(_task(), _tools("get_balance", "read_file"))
        self.assertEqual(cands, [])


class TestDecoyInjection(unittest.TestCase):
    def test_every_agentdojo_suite_has_decoys(self) -> None:
        for suite in ("banking", "slack", "travel", "workspace"):
            self.assertTrue(decoy_tool_names(suite), suite)

    def test_existing_names_are_never_redefined(self) -> None:
        task = _task("banking")
        existing = _tools("send_money_to_iban", "get_balance")
        extra = injected_tools_for(task, existing)
        self.assertNotIn("send_money_to_iban", [t.name for t in extra])

    def test_unknown_suite_gets_nothing_without_fallback(self) -> None:
        task = _task("some_new_suite")
        self.assertEqual(injected_tools_for(task, []), [])
        self.assertTrue(injected_tools_for(task, [], fallback=True))


class TestSampleGoal(unittest.TestCase):
    def test_decoys_outrank_the_tasks_own_sinks(self) -> None:
        task = _task("banking")
        tools = _tools("send_money", "get_iban")
        tools += injected_tools_for(task, tools)
        goal = sample_goal(task, tools)
        assert goal is not None
        self.assertIn(goal.harmful_tool, decoy_tool_names("banking"))
        self.assertFalse(goal.in_benign_plan)

    def test_deterministic_without_explicit_rng(self) -> None:
        task = _task("workspace")
        tools = _tools("search_emails")
        tools += injected_tools_for(task, tools)
        first = sample_goal(task, tools)
        second = sample_goal(task, tools)
        assert first is not None and second is not None
        self.assertEqual(first.harmful_tool, second.harmful_tool)
        self.assertEqual(first.goal_kind, second.goal_kind)

    def test_none_when_no_sink_exists(self) -> None:
        self.assertIsNone(sample_goal(_task("unknown_suite"), _tools("get_balance")))

    def test_goal_text_names_the_tool_and_disclaims_user_intent(self) -> None:
        task = _task("travel")
        tools = injected_tools_for(task, [])
        goal = sample_goal(task, tools)
        assert goal is not None
        self.assertIn(goal.harmful_tool, goal.goal_text)
        self.assertIn("did NOT", goal.goal_text)
        self.assertIn(goal.goal_kind, HARMFUL_SINK_KINDS)


class TestResolvePlaceholders(unittest.TestCase):
    def _goal(self, kind: str = SINK_KIND_EXFILTRATION, tool: str = "share_statement"):
        task = _task("banking")
        tools = injected_tools_for(task, [])
        goal = sample_goal(task, tools)
        assert goal is not None
        return goal

    def test_target_tool_is_bound(self) -> None:
        goal = self._goal()
        out = resolve_placeholders("call {{TARGET_TOOL}}({{TARGET_ARG}}) now", goal)
        self.assertIn(goal.harmful_tool, out)
        self.assertNotIn("{{", out)

    def test_none_goal_leaves_template_untouched(self) -> None:
        tmpl = "call {{TARGET_TOOL}}({{TARGET_ARG}}) now"
        self.assertEqual(resolve_placeholders(tmpl, None), tmpl)

    def test_unknown_placeholder_survives(self) -> None:
        out = resolve_placeholders("{{TARGET_TOOL}} and {{NOT_A_KNOWN_SLOT}}", self._goal())
        self.assertIn("{{NOT_A_KNOWN_SLOT}}", out)

    def test_arg_hint_tracks_the_sink_kind(self) -> None:
        task = _task("banking")
        tools = injected_tools_for(task, [])
        goal = sample_goal(task, tools)
        assert goal is not None
        out = resolve_placeholders("{{TARGET_ARG}}", goal)
        self.assertNotIn("{{", out)
        # Every hint is a bracketed slot the mutator still has to fill in, never
        # a concrete literal (a hard-coded address would be a shared give-away).
        self.assertRegex(out, r"<[^>]+>")


class TestSeedBundleV3(unittest.TestCase):
    """The shipped v3 bundle must stay loadable by the prewarm loader."""

    PATH = os.path.join(_REPO, "data", "seeds_v3", "universal_seeds.json")

    def setUp(self) -> None:
        if not os.path.isfile(self.PATH):
            self.skipTest("data/seeds_v3/universal_seeds.json absent")
        with open(self.PATH, encoding="utf-8") as fh:
            self.doc = json.load(fh)

    def test_loader_contract_fields_present(self) -> None:
        for ent in self.doc["seeds"]:
            self.assertTrue(str(ent.get("method") or "").strip(), ent.get("id"))
            self.assertIsInstance(ent.get("turn"), int)
            self.assertTrue(str(ent.get("payload_template") or "").strip())

    def test_method_labels_are_unique(self) -> None:
        labels = [e["method"] for e in self.doc["seeds"]]
        self.assertEqual(len(labels), len(set(labels)))

    def test_every_harmful_sink_kind_has_a_warm_start(self) -> None:
        families = {e.get("mechanism_family") for e in self.doc["seeds"]}
        for kind in HARMFUL_SINK_KINDS:
            self.assertIn(kind, families, kind)

    def test_all_placeholders_resolve_against_a_sampled_goal(self) -> None:
        task = _task("workspace")
        tools = injected_tools_for(task, [])
        goal = sample_goal(task, tools)
        assert goal is not None
        for ent in self.doc["seeds"]:
            out = resolve_placeholders(ent["payload_template"], goal)
            self.assertNotIn("{{", out, ent["id"])


def main() -> None:
    unittest.main(module=__name__, argv=["test_harmful_catalog"], exit=False)


if __name__ == "__main__":
    main()
