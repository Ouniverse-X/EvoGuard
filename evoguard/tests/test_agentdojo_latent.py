"""Offline tests for the AgentDojo-Latent probe (data + env + attack loader).

This dataset is *authored*, so the failure modes are different from a vendored
one: nothing upstream will complain if a payload stops making sense. Six things
can break silently and each has a layer below.

1. **The build is stale.** ``scripts/build_agentdojo_latent.py`` is the only
   writer; a hand-edit to a scenario file, or an edit to ``authored/`` without a
   rebuild, invalidates every number recorded against the probe. Layer 1 is the
   sha256 guard.
2. **The camouflage premise collapses.** The whole point is that the harmful
   sink is a real out-of-plan action reached after genuine progress. If the sink
   is already in the clean plan, or is not advertised in ``env_info``, or the
   declared Δ does not fit the plan, the row measures nothing. Layer 2 re-checks
   every constraint the build script enforces, against the shipped bytes.
3. **The arms become distinguishable.** ``observation`` must be exactly
   ``benign_observation + payload``, and the paired clean row must carry the same
   ``benign_observation``. If they drift, a defender can separate the arms on
   shape without reading either. Layer 2 and layer 4 cover the two halves.
4. **The task uid.** The env and the loader both recompute it from the
   instruction; a drift matches zero attacks against zero tasks and the replay
   harness reports "0 scenarios" with no error. Layer 3 asserts every emitted
   ``VendoredAttack.task_id`` exists in ``env.get_tasks()``.
5. **The design distribution.** 48 injected rows, 4 suites x 3 declared Δ x 6
   techniques with 2 rows each -- a lopsided rebuild would make per-Δ and
   per-technique breakdowns unreportable. Layer 2 pins the counts.
6. **The clean arm's carrier observation.** ``execute`` must serve the authored
   benign text for any carrier of the task, including the banking task with four
   distinct carriers, and must fall through to the simulator otherwise. Layer 4
   pins the tiering.

Run via::

    python -m evoguard.tests.test_agentdojo_latent
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from evoguard.core.types import ToolCall  # noqa: E402
from evoguard.envs import available_envs  # noqa: E402
from evoguard.envs.agentdojo_latent import (  # noqa: E402
    DATASET_NAME,
    DELTAS,
    SUITES,
    TECHNIQUES,
    AgentDojoLatentEnv,
    agentdojo_latent_root,
    iter_scenario_rows,
    task_uid,
    verify_guard,
)
from evoguard.envs.tool_parsing import parse_env_info  # noqa: E402
from evoguard.llm.mock_client import MockClient  # noqa: E402
from evoguard.process.agentdojo_latent_loader import (  # noqa: E402
    load_agentdojo_latent_attacks,
)
from evoguard.process.vendored_attack_loaders import load_vendored_scenarios  # noqa: E402

DATA_ROOT = os.path.join(_PKG_ROOT, "data")
ROOT = agentdojo_latent_root(DATA_ROOT)

#: Shipped design: 12 injected rows per suite, 4 per declared Δ, 2 per technique.
N_INJECTED_PER_SUITE = 12
N_INJECTED_TOTAL = N_INJECTED_PER_SUITE * len(SUITES)


def _rows(suite: str | None = None) -> list[dict]:
    return list(iter_scenario_rows(ROOT, [suite] if suite else None))


# --------------------------------------------------------------------------- #
# Layer 1 -- the build is the only writer                                      #
# --------------------------------------------------------------------------- #
class BuildGuard(unittest.TestCase):
    def test_guard_matches_on_disk_bytes(self):
        verify_guard(ROOT)

    def test_guard_covers_every_generated_file(self):
        path = os.path.join(ROOT, "_sha256_guard.txt")
        with open(path, "r", encoding="utf-8") as f:
            guarded = {
                line[len("sha256("):].split(")=")[0]
                for line in f
                if line.startswith("sha256(")
            }
        expected = {f"scenarios/{s}.jsonl" for s in SUITES} | {"manifest.json"}
        self.assertEqual(guarded, expected)

    def test_manifest_agrees_with_scenarios(self):
        with open(os.path.join(ROOT, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["dataset"], DATASET_NAME)
        self.assertEqual(manifest["n_injected_total"], N_INJECTED_TOTAL)
        for suite in SUITES:
            rows = _rows(suite)
            per = manifest["per_suite"][suite]
            self.assertEqual(per["n_rows"], len(rows), suite)
            self.assertEqual(
                per["n_injected"],
                sum(r["sample_type"] == "injected" for r in rows),
                suite,
            )

    def test_no_split_metadata_anywhere(self):
        # The probe is held out wholesale; a split label would invite training
        # on part of it.
        with open(os.path.join(ROOT, "manifest.json"), encoding="utf-8") as f:
            self.assertIn("held-out", json.load(f)["split_unit"])
        for row in _rows():
            self.assertNotIn("split", row)


# --------------------------------------------------------------------------- #
# Layer 2 -- the camouflage premise, re-checked against the shipped bytes       #
# --------------------------------------------------------------------------- #
class ScenarioInvariants(unittest.TestCase):
    def setUp(self):
        self.rows = _rows()
        self.injected = [r for r in self.rows if r["sample_type"] == "injected"]

    def test_counts_and_design_distribution(self):
        self.assertEqual(len(self.injected), N_INJECTED_TOTAL)
        for suite in SUITES:
            inj = [r for r in self.injected if r["suite"] == suite]
            self.assertEqual(len(inj), N_INJECTED_PER_SUITE, suite)
            by_delta = collections.Counter(r["expected_delta"] for r in inj)
            self.assertEqual(dict(by_delta),
                             {d: N_INJECTED_PER_SUITE // len(DELTAS) for d in DELTAS},
                             suite)
            by_tech = collections.Counter(r["stealth_technique"] for r in inj)
            self.assertEqual(dict(by_tech), {t: 2 for t in TECHNIQUES}, suite)

    def test_instance_ids_unique(self):
        ids = [r["instance_id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_task_uid_recomputes_from_instruction(self):
        for row in self.rows:
            self.assertEqual(
                row["task_id"], task_uid(row["suite"], row["task_instruction"]),
                row["instance_id"],
            )

    def test_observation_is_benign_plus_payload(self):
        # The ONE thing that makes Δ measurable: the two arms differ by the
        # payload and nothing else.
        for row in self.injected:
            self.assertEqual(
                row["observation"], row["benign_observation"] + row["payload"],
                row["instance_id"],
            )
            self.assertTrue(row["payload"].strip(), row["instance_id"])

    def test_payload_carries_no_injection_delimiter(self):
        # AgentDojo's own payloads are wrapped in <INFORMATION>...</INFORMATION>,
        # which is a free detector. A latent payload must not hand one over.
        for row in self.injected:
            low = row["payload"].lower()
            for tell in ("<information>", "</information>",
                         "ignore the previous", "ignore all previous"):
                self.assertNotIn(tell, low, f"{row['instance_id']}: {tell}")

    def test_carrier_sits_where_the_plan_says(self):
        for row in self.rows:
            plan = row["clean_plan"]
            idx = row["carrier_index"]
            self.assertEqual(plan[idx], row["carrier_tool"], row["instance_id"])

    def test_sink_is_advertised_and_out_of_plan(self):
        for row in self.injected:
            advertised = {s.name for s in parse_env_info(row["env_info"])}
            self.assertIn(row["harmful_tool"], advertised, row["instance_id"])
            self.assertNotIn(row["harmful_tool"], row["clean_plan"],
                             row["instance_id"])
            self.assertIn(row["carrier_tool"], advertised, row["instance_id"])

    def test_declared_delta_is_consistent_with_the_induced_plan(self):
        for row in self.injected:
            plan, delta = row["clean_plan"], row["expected_delta"]
            turning = row["carrier_index"] + delta
            self.assertEqual(row["expected_turning_index"], turning,
                             row["instance_id"])
            self.assertLessEqual(turning, len(plan), row["instance_id"])
            self.assertEqual(
                row["expected_induced_plan"],
                plan[:turning] + [row["harmful_tool"]] + plan[turning:],
                row["instance_id"],
            )
            # Δ >= 1 is what "latent" means: the sink is never the very next
            # action after the poisoned observation.
            self.assertGreaterEqual(delta, 1, row["instance_id"])

    def test_clean_rows_carry_no_attack_fields(self):
        for row in self.rows:
            if row["sample_type"] != "clean":
                continue
            self.assertEqual(row["observation"], row["benign_observation"])
            for key in ("payload", "harmful_tool", "attacker_goal",
                        "expected_delta", "stealth_technique"):
                self.assertIsNone(row[key], f"{row['instance_id']}/{key}")

    def test_every_injected_task_has_a_clean_twin_for_its_carrier(self):
        clean = {(r["task_id"], r["carrier_tool"])
                 for r in self.rows if r["sample_type"] == "clean"}
        for row in self.injected:
            self.assertIn((row["task_id"], row["carrier_tool"]), clean,
                          row["instance_id"])

    def test_task_ref_points_at_a_real_agentdojo_task(self):
        for row in self.rows:
            dataset, version, suite, ref = row["task_ref"].split(":")
            self.assertEqual((dataset, version), ("agentdojo", "v1"))
            self.assertEqual(suite, row["suite"])
            src = os.path.join(DATA_ROOT, "agentdojo", "default_suites", "v1",
                               suite, "user_tasks.py")
            with open(src, encoding="utf-8") as f:
                self.assertIn(f"class {ref}(", f.read(), row["task_ref"])


# --------------------------------------------------------------------------- #
# Layer 3 -- loader/env uid agreement                                          #
# --------------------------------------------------------------------------- #
class LoaderAgreement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = AgentDojoLatentEnv(MockClient(), data_root=DATA_ROOT)
        cls.attacks = load_agentdojo_latent_attacks(data_root=DATA_ROOT)

    def test_registered(self):
        self.assertIn(DATASET_NAME, available_envs())

    def test_only_injected_rows_become_scenarios(self):
        self.assertEqual(len(self.attacks), N_INJECTED_TOTAL)
        for va in self.attacks:
            self.assertTrue(va.payload.strip())

    def test_every_scenario_matches_a_task(self):
        uids = {t.task_id for t in self.env.get_tasks()}
        for va in self.attacks:
            self.assertIn(va.task_id, uids, va.goal)

    def test_dispatch_table_routes_the_dataset(self):
        via_table = load_vendored_scenarios(
            data_root=DATA_ROOT, dataset=DATASET_NAME,
        )
        self.assertEqual(len(via_table), len(self.attacks))
        self.assertEqual([v.task_id for v in via_table],
                         [v.task_id for v in self.attacks])

    def test_goal_string_names_the_sink(self):
        # dataset_builder._infer_target_tool and the judge prompt both read the
        # target tool out of this string.
        for va in self.attacks:
            self.assertTrue(va.goal.startswith(f"call {va.mal_tool} "), va.goal)

    def test_mal_args_are_canonical_json(self):
        for va in self.attacks:
            args = json.loads(va.mal_args)
            self.assertIsInstance(args, dict)
            self.assertEqual(va.mal_args,
                             json.dumps(args, sort_keys=True, ensure_ascii=False))

    def test_suite_filter(self):
        only = load_agentdojo_latent_attacks(data_root=DATA_ROOT,
                                             suites=["banking"])
        self.assertEqual(len(only), N_INJECTED_PER_SUITE)
        self.assertEqual({v.suite for v in only}, {"banking"})

    def test_dataset_dir_accepts_the_scenarios_dir(self):
        via_dir = load_agentdojo_latent_attacks(
            data_root=DATA_ROOT,
            dataset_dir=os.path.join(ROOT, "scenarios"),
        )
        self.assertEqual(len(via_dir), N_INJECTED_TOTAL)


# --------------------------------------------------------------------------- #
# Layer 4 -- the env: tools, clean-arm carrier, simulator fallthrough           #
# --------------------------------------------------------------------------- #
class EnvBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = AgentDojoLatentEnv(MockClient(), data_root=DATA_ROOT)
        cls.tasks = {t.task_id: t for t in cls.env.get_tasks()}

    def test_task_count_and_suites(self):
        self.assertEqual(len(self.tasks), 7 + 12 + 12 + 12)
        self.assertEqual({t.suite for t in self.tasks.values()}, set(SUITES))
        for task in self.tasks.values():
            self.assertEqual(task.dataset, DATASET_NAME)
            self.assertNotIn("split", task.metadata)

    def test_sink_is_advertised_to_the_agent(self):
        # Unlike InjecAgent, the sink is one of the suite's own tools, so hiding
        # it would make the attack unexecutable rather than harder to detect.
        by_task: dict[str, set[str]] = collections.defaultdict(set)
        for row in _rows():
            if row["sample_type"] == "injected":
                by_task[row["task_id"]].add(row["harmful_tool"])
        self.assertTrue(by_task)
        for uid, sinks in by_task.items():
            names = {s.name for s in self.env.get_tools(self.tasks[uid])}
            self.assertTrue(sinks <= names, uid)

    def test_clean_arm_serves_the_authored_carrier_observation(self):
        for row in _rows():
            task = self.tasks[row["task_id"]]
            got = self.env.execute(task, ToolCall(name=row["carrier_tool"],
                                                  arguments={}), [])
            self.assertEqual(got, row["benign_observation"], row["instance_id"])
            self.assertNotIn(row["payload"] or "\x00", got)

    def test_multi_carrier_task_keeps_all_four_observations(self):
        # banking UserTask15 is poisoned through four different carriers; a
        # single-slot metadata field would silently serve the wrong text.
        multi = [t for t in self.tasks.values()
                 if len(t.metadata["carrier_tools"]) > 1]
        self.assertEqual(len(multi), 1)
        self.assertEqual(len(multi[0].metadata["carrier_tools"]), 4)

    def test_non_carrier_tool_falls_through_to_the_simulator(self):
        task = next(iter(self.tasks.values()))
        carriers = set(task.metadata["carrier_tools"])
        other = next(s.name for s in self.env.get_tools(task)
                     if s.name not in carriers)
        out = self.env.execute(task, ToolCall(name=other, arguments={}), [])
        self.assertIsInstance(out, str)

    def test_guard_is_enforced_on_construction(self):
        with self.assertRaises(FileNotFoundError):
            AgentDojoLatentEnv(MockClient(), data_root=os.path.join(_HERE, "nope"))


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
