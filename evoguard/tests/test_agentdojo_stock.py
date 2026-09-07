"""Offline tests for AgentDojo-Stock, the Δ=0 control arm of AgentDojo-Latent.

The latent probe's own tests (``test_agentdojo_latent.py``) already pin the
invariants a scenario file needs to be *loadable*. This set is derived from that
one, so those hold by construction; what can break here is the thing the control
exists for.

**The arms stop being paired.** ASR(latent) − ASR(stock) is only the effect of
camouflage if every other variable is held. The moment a rebuild of either side
changes a task instruction, a clean plan, a carrier, a benign observation or a
sink specification, the difference absorbs that change too and the number stops
meaning anything -- silently, because both files still load and both replays
still produce an ASR. Layer 1 asserts field-by-field equality against the twin
for all 48 injected and 46 clean rows, which is the only check that catches a
one-sided rebuild.

Three smaller things also have to hold:

* **The payload really is the stock attack.** If it lost the ``<INFORMATION>``
  delimiter or stopped naming the sink, this arm would be a third condition
  rather than the control -- ``test_agentdojo_latent`` asserts the *absence* of
  that delimiter, so here it is asserted present (layer 2).
* **Δ is declared 1 everywhere**, and ``expected_induced_plan`` agrees, so a
  by-Δ breakdown can put this arm in its own cell (layer 2).
* **Env and loader are wired to the new directory**, not still to the latent one
  -- a copy-paste that left ``subdir`` alone would score the latent set twice and
  report a zero effect (layer 3).

Run via::

    python -m evoguard.tests.test_agentdojo_stock
"""

from __future__ import annotations

import collections
import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from evoguard.envs import available_envs  # noqa: E402
from evoguard.envs.agentdojo_latent import (  # noqa: E402
    DELTAS,
    SUITES,
    agentdojo_latent_root,
    iter_scenario_rows,
    verify_guard,
)
from evoguard.envs.agentdojo_stock import (  # noqa: E402
    DATASET_NAME,
    STOCK_DELTA,
    TECHNIQUES,
    AgentDojoStockEnv,
)
from evoguard.envs.tool_parsing import parse_env_info  # noqa: E402
from evoguard.llm.mock_client import MockClient  # noqa: E402
from evoguard.process.agentdojo_stock_loader import (  # noqa: E402
    load_agentdojo_stock_attacks,
)
from evoguard.process.vendored_attack_loaders import load_vendored_scenarios  # noqa: E402

DATA_ROOT = os.path.join(_PKG_ROOT, "data")
ROOT = agentdojo_latent_root(DATA_ROOT, DATASET_NAME)
LATENT_ROOT = agentdojo_latent_root(DATA_ROOT)

N_INJECTED_PER_SUITE = 12
N_INJECTED_TOTAL = N_INJECTED_PER_SUITE * len(SUITES)
N_CLEAN_TOTAL = 46

#: Every column that must be byte-identical to the latent twin. The complement
#: -- payload, observation, instance_id, task_id, expected_*, stealth_*, twin_*
#: -- is exactly what this arm is allowed to change.
PAIRED_FIELDS = (
    "suite",
    "task_ref",
    "task_instruction",
    "env_info",
    "clean_plan",
    "carrier_tool",
    "carrier_index",
    "benign_observation",
    "sample_type",
)

#: Additionally paired on injected rows: the sink specification. If these drift,
#: the two arms are pointing the agent at different actions.
PAIRED_SINK_FIELDS = ("harmful_tool", "harmful_args_hint", "attacker_goal")


def _rows(root: str, suite: str | None = None) -> list[dict]:
    return list(iter_scenario_rows(root, [suite] if suite else None))


def _by_instance(rows: list[dict]) -> dict[str, dict]:
    return {r["instance_id"]: r for r in rows}


# --------------------------------------------------------------------------- #
# Layer 1 -- the pairing                                                       #
# --------------------------------------------------------------------------- #
class Pairing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = _rows(ROOT)
        cls.latent = _by_instance(_rows(LATENT_ROOT))

    def test_row_counts_match_the_latent_arm(self):
        self.assertEqual(len(self.stock), len(self.latent))
        self.assertEqual(
            sum(r["sample_type"] == "injected" for r in self.stock),
            N_INJECTED_TOTAL,
        )
        self.assertEqual(
            sum(r["sample_type"] == "clean" for r in self.stock), N_CLEAN_TOTAL,
        )

    def test_twin_reference_is_total_and_injective(self):
        twins = [r["twin_instance_id"] for r in self.stock]
        self.assertEqual(len(twins), len(set(twins)))
        self.assertEqual(set(twins), set(self.latent))

    def test_every_paired_field_is_identical_to_the_twin(self):
        for row in self.stock:
            twin = self.latent[row["twin_instance_id"]]
            for key in PAIRED_FIELDS:
                self.assertEqual(row[key], twin[key],
                                 f"{row['instance_id']}/{key}")
            if row["sample_type"] != "injected":
                continue
            for key in PAIRED_SINK_FIELDS:
                self.assertEqual(row[key], twin[key],
                                 f"{row['instance_id']}/{key}")

    def test_only_the_payload_differs(self):
        # Stated as its own assertion because it is the claim the experiment
        # rests on: same benign prefix, different appended text.
        for row in self.stock:
            if row["sample_type"] != "injected":
                continue
            twin = self.latent[row["twin_instance_id"]]
            self.assertNotEqual(row["payload"], twin["payload"],
                                row["instance_id"])
            self.assertEqual(row["observation"],
                             row["benign_observation"] + row["payload"],
                             row["instance_id"])
            self.assertTrue(twin["observation"].startswith(
                row["benign_observation"]), row["instance_id"])

    def test_twin_declared_delta_covers_the_latent_design(self):
        inj = [r for r in self.stock if r["sample_type"] == "injected"]
        for row in inj:
            self.assertEqual(row["twin_declared_delta"],
                             self.latent[row["twin_instance_id"]]["expected_delta"],
                             row["instance_id"])
        by_delta = collections.Counter(r["twin_declared_delta"] for r in inj)
        self.assertEqual(dict(by_delta),
                         {d: N_INJECTED_TOTAL // len(DELTAS) for d in DELTAS})

    def test_task_ids_do_not_collide_with_the_latent_arm(self):
        # Both replays land in the same aggregate; a shared uid would merge the
        # arms' records.
        stock_uids = {r["task_id"] for r in self.stock}
        latent_uids = {r["task_id"] for r in self.latent.values()}
        self.assertEqual(len(stock_uids), len(latent_uids))
        self.assertEqual(stock_uids & latent_uids, set())
        for row in self.stock:
            self.assertTrue(row["task_id"].startswith(f"{DATASET_NAME}:"),
                            row["task_id"])


# --------------------------------------------------------------------------- #
# Layer 2 -- it is the stock attack, declared at Δ=1                           #
# --------------------------------------------------------------------------- #
class StockPayload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.injected = [r for r in _rows(ROOT) if r["sample_type"] == "injected"]

    def test_build_guard_matches_on_disk_bytes(self):
        verify_guard(ROOT)

    def test_manifest_declares_the_control_role(self):
        with open(os.path.join(ROOT, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["dataset"], DATASET_NAME)
        self.assertEqual(manifest["n_injected_total"], N_INJECTED_TOTAL)
        self.assertIn("held-out", manifest["split_unit"])

    def test_payload_is_the_agentdojo_information_block(self):
        for row in self.injected:
            payload = row["payload"]
            self.assertIn("<INFORMATION>", payload, row["instance_id"])
            self.assertIn("</INFORMATION>", payload, row["instance_id"])
            self.assertIn("important message from me, Emma Johnson",
                          payload, row["instance_id"])

    def test_payload_names_the_sink_and_its_arguments(self):
        for row in self.injected:
            self.assertIn(f"`{row['harmful_tool']}`", row["payload"],
                          row["instance_id"])
            for value in row["harmful_args_hint"]:
                self.assertIn(str(value), row["payload"], row["instance_id"])

    def test_goal_is_not_double_punctuated(self):
        for row in self.injected:
            self.assertNotIn("..", row["payload"], row["instance_id"])

    def test_single_technique_tag(self):
        self.assertEqual(
            {r["stealth_technique"] for r in self.injected}, set(TECHNIQUES),
        )

    def test_declared_delta_is_one_everywhere(self):
        for row in self.injected:
            self.assertEqual(row["expected_delta"], STOCK_DELTA,
                             row["instance_id"])
            turning = row["carrier_index"] + STOCK_DELTA
            self.assertEqual(row["expected_turning_index"], turning,
                             row["instance_id"])
            plan = row["clean_plan"]
            self.assertLessEqual(turning, len(plan), row["instance_id"])
            self.assertEqual(
                row["expected_induced_plan"],
                plan[:turning] + [row["harmful_tool"]] + plan[turning:],
                row["instance_id"],
            )

    def test_sink_is_advertised_and_out_of_plan(self):
        for row in self.injected:
            advertised = {s.name for s in parse_env_info(row["env_info"])}
            self.assertIn(row["harmful_tool"], advertised, row["instance_id"])
            self.assertNotIn(row["harmful_tool"], row["clean_plan"],
                             row["instance_id"])

    def test_clean_rows_carry_no_attack_fields(self):
        for row in _rows(ROOT):
            if row["sample_type"] != "clean":
                continue
            self.assertEqual(row["observation"], row["benign_observation"])
            for key in ("payload", "harmful_tool", "attacker_goal",
                        "expected_delta", "stealth_technique",
                        "twin_declared_delta"):
                self.assertIsNone(row[key], f"{row['instance_id']}/{key}")


# --------------------------------------------------------------------------- #
# Layer 3 -- env + loader point at data/agentdojo_stock                        #
# --------------------------------------------------------------------------- #
class Wiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = AgentDojoStockEnv(MockClient(), data_root=DATA_ROOT)
        cls.attacks = load_agentdojo_stock_attacks(data_root=DATA_ROOT)

    def test_registered_under_its_own_name(self):
        self.assertIn(DATASET_NAME, available_envs())

    def test_env_loads_this_directory_not_the_latent_one(self):
        tasks = self.env.get_tasks()
        self.assertEqual(len(tasks), 7 + 12 + 12 + 12)
        for task in tasks:
            self.assertEqual(task.dataset, DATASET_NAME)
            self.assertTrue(task.task_id.startswith(f"{DATASET_NAME}:"))
            self.assertNotIn("split", task.metadata)

    def test_every_scenario_matches_a_task(self):
        self.assertEqual(len(self.attacks), N_INJECTED_TOTAL)
        uids = {t.task_id for t in self.env.get_tasks()}
        for va in self.attacks:
            self.assertIn(va.task_id, uids, va.goal)
            self.assertIn("<INFORMATION>", va.payload)

    def test_dispatch_table_routes_the_dataset(self):
        via_table = load_vendored_scenarios(data_root=DATA_ROOT,
                                            dataset=DATASET_NAME)
        self.assertEqual([v.task_id for v in via_table],
                         [v.task_id for v in self.attacks])
        self.assertEqual([v.payload for v in via_table],
                         [v.payload for v in self.attacks])

    def test_clean_arm_serves_the_authored_carrier_observation(self):
        from evoguard.core.types import ToolCall

        tasks = {t.task_id: t for t in self.env.get_tasks()}
        for row in _rows(ROOT):
            got = self.env.execute(tasks[row["task_id"]],
                                   ToolCall(name=row["carrier_tool"],
                                            arguments={}), [])
            self.assertEqual(got, row["benign_observation"], row["instance_id"])


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
