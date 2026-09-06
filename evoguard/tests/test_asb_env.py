"""Offline tests for the ASB-OPI integration (env + attack loader + splits).

Four things can break silently here, and each has a layer below:

1. **The task uid.** ``envs.asb.task_uid`` is recomputed by the attack loader,
   so a drift in either place matches zero attacks against zero tasks and the
   replay harness just reports "0 scenarios" — no error. Layer 2 asserts every
   emitted ``VendoredAttack.task_id`` exists in ``env.get_tasks()``.
2. **Attacker-tool leakage.** The bait tool must NOT appear in ``get_tools``
   (else a defender wins on the tool list alone) but MUST still resolve for the
   executor. Layer 1 pins both halves.
3. **The split guard.** ``data/ASB/splits/_sha256_guard.txt`` records the three
   digests; nothing else in the repo enforces them (``scripts/split_asb_opi.py``
   is not vendored), so historical ASB numbers stop being comparable the moment
   a split file is edited. Layer 3 is that enforcer.
4. **The prompt hook.** ``build_system_prompt`` now emits ``AGENT_ROLE`` when a
   task declares ``metadata["role_prompt"]``. Layer 4 asserts it fires for ASB
   and is byte-for-byte absent otherwise.

Run via::

    /root/miniconda3/envs/evoguard/bin/python -m evoguard.tests.test_asb_env
"""

from __future__ import annotations

import hashlib
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from evoguard.agents.prompts import build_system_prompt  # noqa: E402
from evoguard.attacks.harmful_catalog import sample_goal  # noqa: E402
from evoguard.core.types import Task, ToolCall  # noqa: E402
from evoguard.envs import available_envs  # noqa: E402
from evoguard.envs.asb import (  # noqa: E402
    DATASET_NAME,
    SPLITS,
    ASBOPIEnv,
    asb_root,
    iter_split_rows,
    load_harmful_goal_candidates,
    task_uid,
)
from evoguard.llm.mock_client import MockClient  # noqa: E402
from evoguard.process.asb_attack_loader import load_asb_attacks  # noqa: E402
from evoguard.process.vendored_attack_loaders import (  # noqa: E402
    load_vendored_scenarios,
)

_DATA_ROOT = os.path.join(_PKG_ROOT, "data")
_ROOT = asb_root(_DATA_ROOT)
_HAVE_DATA = os.path.isdir(os.path.join(_ROOT, "splits", "train"))
_skip_no_data = unittest.skipUnless(_HAVE_DATA, f"ASB dataset absent at {_ROOT}")


def _env() -> ASBOPIEnv:
    return ASBOPIEnv(MockClient(), data_root=_DATA_ROOT)


# --------------------------------------------------------------------------- #
# Layer 0 -- registration                                                      #
# --------------------------------------------------------------------------- #
class Registration(unittest.TestCase):
    def test_registered_under_its_own_name(self):
        self.assertIn(DATASET_NAME, available_envs())
        self.assertEqual(ASBOPIEnv.name, DATASET_NAME)


# --------------------------------------------------------------------------- #
# Layer 1 -- env shape                                                         #
# --------------------------------------------------------------------------- #
@_skip_no_data
class EnvShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _env()
        cls.tasks = cls.env.get_tasks()

    def test_loads_the_distinct_agent_task_pairs(self):
        self.assertEqual(len(self.tasks), 51)
        self.assertEqual(len({t.task_id for t in self.tasks}), len(self.tasks))
        self.assertEqual(len({t.suite for t in self.tasks}), 10)

    def test_every_task_declares_a_split_and_a_role_prompt(self):
        for t in self.tasks:
            self.assertIn(t.metadata.get("split"), SPLITS, t.task_id)
            self.assertTrue(t.metadata.get("role_prompt"), t.task_id)
            self.assertEqual(t.dataset, DATASET_NAME)

    def test_split_labels_come_from_the_clean_row(self):
        # All 51 user tasks ship a clean row upstream; if that stops being true
        # the env falls back to an attack row's split, which is ambiguous.
        self.assertTrue(all(t.metadata["split_from_clean_row"] for t in self.tasks))
        counts = {s: 0 for s in SPLITS}
        for t in self.tasks:
            counts[t.metadata["split"]] += 1
        self.assertEqual(counts, {"train": 33, "val": 11, "test": 7})

    def test_tools_are_benign_only_and_nonempty(self):
        attacker_names = set(self.env._attacker_tools)                # noqa: SLF001
        self.assertTrue(attacker_names)
        for t in self.tasks:
            specs = self.env.get_tools(t)
            self.assertTrue(specs, t.task_id)
            names = {s.name for s in specs}
            self.assertFalse(names & attacker_names, t.task_id)
            self.assertEqual(names, set(t.tool_names))
            self.assertEqual(t.metadata["carrier_tool"], specs[0].name)

    def test_attacker_tool_still_resolves_for_the_executor(self):
        task = self.tasks[0]
        bait = sorted(self.env._attacker_tools)[0]                    # noqa: SLF001
        spec = self.env._tool_spec(task, bait)                       # noqa: SLF001
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, bait)
        self.assertTrue(spec.description)

    def test_canonical_success_result_is_pinned_for_benign_tools(self):
        task = self.tasks[0]
        carrier = self.env.get_tools(task)[0]
        call = ToolCall(name=carrier.name, arguments={})
        system = self.env._build_executor_system(task, carrier, call)  # noqa: SLF001
        self.assertIn("CANONICAL_SUCCESS_RESULT:", system)
        self.assertIn(self.env._achievements[carrier.name], system)   # noqa: SLF001

    def test_suite_filter_selects_one_agent(self):
        env = ASBOPIEnv(MockClient(), data_root=_DATA_ROOT,
                        suites=["system_admin_agent"])
        tasks = env.get_tasks()
        self.assertTrue(tasks)
        self.assertEqual({t.suite for t in tasks}, {"system_admin_agent"})

    def test_every_task_gets_an_out_of_scope_attacker_objective(self):
        # Without the dataset-supplied catalogue sample_goal returns None on
        # 46/51 tasks (2 benign tools, no decoy table) and the MCTS attacker
        # falls back to goal=task.instruction, i.e. in-scope forgery.
        for t in self.tasks:
            goal = sample_goal(t, self.env.get_tools(t))
            self.assertIsNotNone(goal, t.task_id)
            self.assertNotIn(goal.harmful_tool, t.tool_names, t.task_id)
            self.assertIn(goal.harmful_tool, self.env._attacker_tools)  # noqa: SLF001

    def test_attacker_objectives_never_leak_a_val_or_test_tool(self):
        # ASB-OPI's held-out axis is "unseen attacker tool on a seen scenario",
        # and the three splits partition the 400 tools disjointly. Drawing the
        # attacker's target from val/test would burn that axis.
        held_out = set()
        for split in ("val", "test"):
            for _s, rec in iter_split_rows(_ROOT, (split,)):
                if rec.get("sample_type") != "clean":
                    held_out.add(str(rec.get("attacker_tool", "")))
        train_only = load_harmful_goal_candidates(_ROOT)
        self.assertEqual(len(train_only), 10)
        for agent, cands in train_only.items():
            self.assertTrue(cands, agent)
            for c in cands:
                self.assertNotIn(c["tool"], held_out, agent)
        for t in self.tasks:
            self.assertNotIn(sample_goal(t, self.env.get_tools(t)).harmful_tool,
                             held_out, t.task_id)


# --------------------------------------------------------------------------- #
# Layer 2 -- attack loader                                                     #
# --------------------------------------------------------------------------- #
@_skip_no_data
class AttackLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _env()
        cls.known = {t.task_id: t for t in cls.env.get_tasks()}
        cls.attacks = load_asb_attacks(_DATA_ROOT)

    def test_every_injected_row_becomes_a_scenario(self):
        n_injected = sum(
            1 for _s, r in iter_split_rows(_ROOT)
            if r.get("sample_type") == "injected"
        )
        self.assertEqual(n_injected, 400)
        # De-duplication keys on (task_id, target_tool, real_observation); the
        # payload is per-instance, so no two rows should collide.
        self.assertEqual(len(self.attacks), n_injected)

    def test_task_ids_match_the_env(self):
        # THE uid-drift guard. A single mismatch here means a silent 0-scenario
        # replay, not a crash.
        for va in self.attacks:
            self.assertIn(va.task_id, self.known, va.payload[:60])

    def test_carrier_tool_is_callable_in_the_task(self):
        for va in self.attacks:
            task = self.known[va.task_id]
            self.assertIn(va.target_tool, task.tool_names)
            self.assertEqual(va.target_tool, task.metadata["carrier_tool"])

    def test_payload_is_embedded_in_the_replayed_observation(self):
        for va in self.attacks:
            self.assertTrue(va.payload)
            self.assertIn(va.payload, va.real_observation)
            # The benign prefix must survive: replaying a bare payload would
            # make the injection trivially detectable.
            self.assertGreater(len(va.real_observation), len(va.payload))

    def test_goal_names_the_bait_tool_for_the_call_regex(self):
        for va in self.attacks:
            self.assertTrue(va.mal_tool)
            self.assertTrue(va.goal.startswith(f"call {va.mal_tool}"), va.goal)
            self.assertNotIn(va.mal_tool, self.known[va.task_id].tool_names)

    def test_leaf_split_dir_restricts_the_scenarios(self):
        total = 0
        for split in SPLITS:
            subset = load_asb_attacks(
                _DATA_ROOT, dataset_dir=os.path.join(_ROOT, "splits", split),
            )
            self.assertTrue(subset)
            self.assertLess(len(subset), len(self.attacks))
            total += len(subset)
        self.assertEqual(total, len(self.attacks))

    def test_registry_dispatches_by_dataset_name(self):
        via_registry = load_vendored_scenarios(_DATA_ROOT, dataset=DATASET_NAME)
        self.assertEqual(
            [va.task_id for va in via_registry],
            [va.task_id for va in self.attacks],
        )

    def test_suite_filter_is_honoured(self):
        subset = load_asb_attacks(_DATA_ROOT, suites=["system_admin_agent"])
        self.assertTrue(subset)
        self.assertEqual({va.suite for va in subset}, {"system_admin_agent"})


# --------------------------------------------------------------------------- #
# Layer 3 -- split immutability                                                #
# --------------------------------------------------------------------------- #
@_skip_no_data
class SplitGuard(unittest.TestCase):
    def test_split_files_match_the_recorded_digests(self):
        guard = os.path.join(_ROOT, "splits", "_sha256_guard.txt")
        expected: dict[str, str] = {}
        with open(guard, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("sha256("):
                    continue
                ref, digest = line.split(")=", 1)
                expected[ref[len("sha256("):]] = digest.strip()
        self.assertEqual(len(expected), len(SPLITS))
        for rel, digest in expected.items():
            path = os.path.join(_ROOT, "splits", rel)
            with open(path, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                actual, digest,
                f"{rel} changed; historical ASB-OPI numbers are no longer "
                f"comparable. Update the guard deliberately, not to make this "
                f"test pass.",
            )


# --------------------------------------------------------------------------- #
# Layer 4 -- prompt hook                                                       #
# --------------------------------------------------------------------------- #
class RolePromptHook(unittest.TestCase):
    def _task(self, metadata: dict) -> Task:
        return Task(task_id="t", instruction="do the thing", suite="s",
                    dataset="d", tool_names=[], metadata=metadata)

    def test_absent_role_prompt_leaves_the_prompt_unchanged(self):
        without = build_system_prompt(self._task({}), [], None)
        self.assertNotIn("AGENT_ROLE", without)
        # Byte-identical to the pre-hook layout.
        self.assertIn("\n\nAVAILABLE_TOOLS_JSON: []\n", without)
        for empty in ("", "   ", None):
            self.assertEqual(
                build_system_prompt(self._task({"role_prompt": empty}), [], None),
                without,
            )

    def test_role_prompt_is_emitted_before_the_tool_list(self):
        with_role = build_system_prompt(
            self._task({"role_prompt": "You are a system admin agent."}), [], None,
        )
        self.assertIn("AGENT_ROLE: You are a system admin agent.", with_role)
        self.assertLess(with_role.index("AGENT_ROLE"),
                        with_role.index("AVAILABLE_TOOLS_JSON"))

    @_skip_no_data
    def test_it_fires_for_real_asb_tasks(self):
        env = _env()
        task = env.get_tasks()[0]
        prompt = build_system_prompt(task, env.get_tools(task), None)
        self.assertIn("AGENT_ROLE:", prompt)
        self.assertIn(task.metadata["role_prompt"], prompt)


# --------------------------------------------------------------------------- #
# Layer 5 -- uid helper                                                        #
# --------------------------------------------------------------------------- #
class TaskUid(unittest.TestCase):
    def test_shape_and_stability(self):
        uid = task_uid("system_admin_agent", "Monitor the network.")
        prefix, agent, digest = uid.split(":")
        self.assertEqual(prefix, DATASET_NAME)
        self.assertEqual(agent, "system_admin_agent")
        self.assertEqual(len(digest), 10)
        self.assertEqual(
            digest,
            hashlib.sha1(b"Monitor the network.").hexdigest()[:10],
        )


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
