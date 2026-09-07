"""Offline tests for the InjecAgent integration (env + attack loader + splits).

Five things can break silently here, and each has a layer below.

1. **The task uid.** ``envs.injecagent.task_uid`` is recomputed by the attack
   loader from a toolkit it looks up in ``tools.json``; a drift in either place
   matches zero attacks against zero tasks and the replay harness just reports
   "0 scenarios" -- no error. Layer 2 asserts every emitted
   ``VendoredAttack.task_id`` exists in ``env.get_tasks()``.
2. **Attacker-tool leakage.** The bait must NOT appear in ``get_tools`` (else a
   defender wins on the tool list alone) but MUST still resolve for the executor.
   Layer 1 pins both halves, plus the one row where bait and carrier genuinely
   coincide.
3. **The pre-simulated response branch.** ``execute`` has three tiers and the
   order is load-bearing: a carrier call must be served from the template (an
   LLM-invented carrier observation would put the clean and replay arms on
   different distributions), and a data-stealing stage-1 call must come from
   ``attacker_simulated_responses.json`` rather than the simulator. Layer 4
   pins the tiering and the cache index.
4. **The split guard.** ``data/InjecAgent/splits/_sha256_guard.txt`` records the
   three digests; a regenerated split silently invalidates every recorded
   number. Layer 3 is that enforcer, and also checks the property the split
   exists for: no attacker case appears in two splits.
5. **The clean observation.** It must keep the template's shape, drop the
   placeholder, and contain no payload -- otherwise clean and poisoned
   observations are distinguishable without reading them. Layer 1 covers it.
6. **The attacker's own objective.** Each task exposes exactly ONE benign tool,
   so without ``metadata["harmful_goal_candidates"]`` ``sample_goal`` returns
   ``None`` on all 17 and the MCTS attacker silently degrades to
   ``goal=task.instruction`` -- in-scope forgery counts as success again, and the
   only symptom is a WARNING. Layer 6 pins the seam and the train-only
   restriction that keeps val/test tool names out of training payloads.

Run via::

    /root/miniconda3/envs/evoguard2/bin/python -m evoguard.tests.test_injecagent_env
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from evoguard.attacks.harmful_catalog import (  # noqa: E402
    GOAL_CANDIDATES_KEY,
    sample_goal,
)
from evoguard.core.types import ToolCall  # noqa: E402
from evoguard.envs import available_envs  # noqa: E402
from evoguard.envs.injecagent import (  # noqa: E402
    DATASET_NAME,
    GOAL_CANDIDATE_SPLIT,
    PLACEHOLDER,
    SPLITS,
    InjecAgentEnv,
    canonical_args,
    injecagent_root,
    iter_split_rows,
    task_uid,
)
from evoguard.llm.mock_client import MockClient  # noqa: E402
from evoguard.process.injecagent_attack_loader import (  # noqa: E402
    load_injecagent_attacks,
)
from evoguard.process.split_injecagent import ENHANCED_PREFIX  # noqa: E402
from evoguard.process.vendored_attack_loaders import (  # noqa: E402
    load_vendored_scenarios,
)

_DATA_ROOT = os.path.join(_PKG_ROOT, "data")
_ROOT = injecagent_root(_DATA_ROOT)
_HAVE_DATA = os.path.isdir(os.path.join(_ROOT, "splits", "train"))
_skip_no_data = unittest.skipUnless(_HAVE_DATA, f"InjecAgent dataset absent at {_ROOT}")

#: Regression baseline. 17 user cases x 62 attacker cases, split by attacker
#: case (38/12/12) -> 646/204/204 rows.
N_TASKS = 17
N_SUITES = 11
N_ROWS = {"train": 646, "val": 204, "test": 204}

#: Distinct stage-1 attacker tools per split. Zero overlap between them is what
#: makes the train-only goal catalogue a valid held-out guarantee.
N_GOAL_CANDIDATES = 38


def _env(**kw) -> InjecAgentEnv:
    return InjecAgentEnv(MockClient(), data_root=_DATA_ROOT, **kw)


# --------------------------------------------------------------------------- #
# Layer 0 -- registration                                                      #
# --------------------------------------------------------------------------- #
class Registration(unittest.TestCase):
    def test_registered_under_its_own_name(self):
        self.assertIn(DATASET_NAME, available_envs())
        self.assertEqual(InjecAgentEnv.name, DATASET_NAME)


# --------------------------------------------------------------------------- #
# Layer 1 -- env shape                                                         #
# --------------------------------------------------------------------------- #
@_skip_no_data
class EnvShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _env()
        cls.tasks = cls.env.get_tasks()

    def test_loads_the_distinct_user_cases(self):
        self.assertEqual(len(self.tasks), N_TASKS)
        self.assertEqual(len({t.task_id for t in self.tasks}), len(self.tasks))
        self.assertEqual(len({t.suite for t in self.tasks}), N_SUITES)
        for t in self.tasks:
            self.assertEqual(t.dataset, DATASET_NAME)

    def test_tasks_declare_no_split(self):
        # All 17 tasks appear in all three splits, so a task-level label would be
        # a lie and would make driver._split_train_val hold tasks out of
        # training for nothing.
        for t in self.tasks:
            self.assertNotIn("split", t.metadata, t.task_id)
            self.assertEqual(t.metadata["attack_splits"], sorted(SPLITS))
            self.assertEqual(t.metadata["n_vendored_attacks"], 62)

    def test_exactly_one_benign_tool_per_task(self):
        for t in self.tasks:
            specs = self.env.get_tools(t)
            self.assertEqual(len(specs), 1, t.task_id)
            self.assertEqual([s.name for s in specs], t.tool_names)
            self.assertEqual(t.metadata["carrier_tool"], specs[0].name)
            # The signature must be complete or the executor simulates blind.
            self.assertTrue(specs[0].description, t.task_id)
            self.assertTrue(specs[0].parameters, t.task_id)

    def test_attacker_tools_are_absent_from_get_tools(self):
        advertised = {s.name for t in self.tasks for s in self.env.get_tools(t)}
        baits = {a.mal_tool for a in load_injecagent_attacks(_DATA_ROOT)}
        self.assertTrue(baits)
        # One genuine coincidence: the data-stealing case that re-reads the
        # GitHub profile the user asked about. Kept rather than dropped so the
        # benchmark stays at its published 1054 rows.
        self.assertEqual(sorted(baits & advertised), ["GitHubGetUserDetails"])

    def test_attacker_tool_still_resolves_for_the_executor(self):
        task = self.tasks[0]
        spec = self.env._tool_spec(task, "TerminalExecute")           # noqa: SLF001
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "TerminalExecute")
        self.assertTrue(spec.description)
        self.assertIsNone(self.env._tool_spec(task, "NoSuchTool"))    # noqa: SLF001

    def test_clean_observation_keeps_the_template_shape(self):
        payloads = {a.payload for a in load_injecagent_attacks(_DATA_ROOT)}
        for t in self.tasks:
            carrier = t.metadata["carrier_tool"]
            obs = self.env.execute(t, ToolCall(name=carrier, arguments={}), [])
            template = t.metadata["tool_response_template"]
            self.assertNotIn(PLACEHOLDER, obs, t.task_id)
            self.assertTrue(obs.startswith(template[:12]), t.task_id)
            self.assertFalse(any(p in obs for p in payloads), t.task_id)

    def test_suite_filter_selects_one_toolkit(self):
        env = _env(suites=["GitHub"])
        tasks = env.get_tasks()
        self.assertEqual({t.suite for t in tasks}, {"GitHub"})
        self.assertEqual(len(tasks), 3)

    def test_max_tasks_truncates(self):
        self.assertEqual(len(_env(max_tasks=5).get_tasks()), 5)


# --------------------------------------------------------------------------- #
# Layer 2 -- attack loader                                                     #
# --------------------------------------------------------------------------- #
@_skip_no_data
class AttackLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _env()
        cls.known = {t.task_id: t for t in cls.env.get_tasks()}
        cls.attacks = load_injecagent_attacks(_DATA_ROOT)

    def test_every_row_becomes_a_scenario(self):
        n_rows = sum(1 for _s, _r in iter_split_rows(_ROOT))
        self.assertEqual(n_rows, sum(N_ROWS.values()))
        # De-duplication keys on (task_id, target_tool, real_observation); all 62
        # attacker instructions are distinct, so no two rows collide.
        self.assertEqual(len(self.attacks), n_rows)

    def test_task_ids_match_the_env(self):
        # THE uid-drift guard. A single mismatch means a silent 0-scenario
        # replay, not a crash.
        for va in self.attacks:
            self.assertIn(va.task_id, self.known, va.payload[:60])

    def test_carrier_tool_is_callable_in_the_task(self):
        for va in self.attacks:
            task = self.known[va.task_id]
            self.assertEqual([va.target_tool], task.tool_names)
            self.assertEqual(va.target_tool, task.metadata["carrier_tool"])

    def test_payload_is_embedded_in_the_replayed_observation(self):
        for va in self.attacks:
            self.assertTrue(va.payload)
            self.assertIn(va.payload, va.real_observation)
            self.assertNotIn(PLACEHOLDER, va.real_observation)
            # The benign scaffold must survive: replaying a bare payload would
            # make the injection trivially detectable.
            self.assertGreater(len(va.real_observation), len(va.payload))

    def test_goal_names_the_bait_tool_for_the_call_regex(self):
        n_bait_is_carrier = 0
        for va in self.attacks:
            self.assertTrue(va.mal_tool)
            self.assertTrue(va.goal.startswith(f"call {va.mal_tool}"), va.goal)
            self.assertTrue(va.goal_text)
            if va.mal_tool == va.target_tool:
                n_bait_is_carrier += 1
        self.assertEqual(n_bait_is_carrier, 1)

    def test_leaf_split_dir_restricts_the_scenarios(self):
        for split, expected in N_ROWS.items():
            subset = load_injecagent_attacks(
                _DATA_ROOT, dataset_dir=os.path.join(_ROOT, "splits", split),
            )
            self.assertEqual(len(subset), expected, split)
        self.assertEqual(sum(N_ROWS.values()), len(self.attacks))

    def test_enhanced_setting_prepends_the_jailbreak_wrapper(self):
        enhanced = load_injecagent_attacks(_DATA_ROOT, setting="enhanced")
        self.assertEqual(len(enhanced), len(self.attacks))
        for base, enh in zip(self.attacks, enhanced):
            self.assertEqual(enh.payload, ENHANCED_PREFIX + base.payload)
            self.assertIn(enh.payload, enh.real_observation)
        with self.assertRaises(ValueError):
            load_injecagent_attacks(_DATA_ROOT, setting="nonsense")

    def test_registry_dispatches_by_dataset_name(self):
        via_registry = load_vendored_scenarios(_DATA_ROOT, dataset=DATASET_NAME)
        self.assertEqual(
            [va.task_id for va in via_registry],
            [va.task_id for va in self.attacks],
        )

    def test_suite_filter_is_honoured(self):
        subset = load_injecagent_attacks(_DATA_ROOT, suites=["GitHub"])
        self.assertEqual({va.suite for va in subset}, {"GitHub"})
        self.assertEqual(len(subset), 3 * 62)


# --------------------------------------------------------------------------- #
# Layer 3 -- split integrity                                                   #
# --------------------------------------------------------------------------- #
@_skip_no_data
class SplitIntegrity(unittest.TestCase):
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
                f"{rel} changed; historical InjecAgent numbers are no longer "
                f"comparable. Regenerate the guard deliberately (via "
                f"split_injecagent --force), not to make this test pass.",
            )

    def test_no_attacker_case_straddles_two_splits(self):
        # This is the whole point of splitting on the attacker case: a held-out
        # payload must never have been trained on.
        cases: dict[str, set[tuple[str, int]]] = {}
        rows_per_split: dict[str, int] = {}
        users_per_split: dict[str, set[int]] = {}
        for split, rec in iter_split_rows(_ROOT):
            cases.setdefault(split, set()).add(
                (rec["attack_class"], rec["attacker_case_index"]))
            rows_per_split[split] = rows_per_split.get(split, 0) + 1
            users_per_split.setdefault(split, set()).add(rec["user_case_index"])
        self.assertEqual(rows_per_split, N_ROWS)
        self.assertEqual({s: len(v) for s, v in cases.items()},
                         {"train": 38, "val": 12, "test": 12})
        for a in SPLITS:
            for b in SPLITS:
                if a < b:
                    self.assertEqual(cases[a] & cases[b], set(), f"{a}/{b}")
            # Every user task stays available as a carrier on every side.
            self.assertEqual(len(users_per_split[a]), N_TASKS, a)

    def test_manifest_agrees_with_the_files(self):
        path = os.path.join(_ROOT, "splits", "split_manifest.json")
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["split_unit"], "attacker case")
        self.assertEqual(manifest["attack_setting"], "base")
        self.assertEqual(manifest["n_rows_by_split"], N_ROWS)
        self.assertEqual(manifest["n_user_cases"], N_TASKS)


# --------------------------------------------------------------------------- #
# Layer 4 -- the pre-simulated response branch                                 #
# --------------------------------------------------------------------------- #
@_skip_no_data
class SimulatedResponseTiers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _env()
        cls.task = cls.env.get_tasks()[0]

    def test_cache_index_shape(self):
        exact = self.env._sim_exact                                   # noqa: SLF001
        default = self.env._sim_default                               # noqa: SLF001
        # 2347 raw keys -> 1385 distinct (name, canonical args) pairs over the
        # 32 data-stealing stage-1 read tools. The 30 direct-harm tools are
        # absent: upstream never simulates them, because a direct-harm attack is
        # already scored at the call itself.
        self.assertEqual(len(exact), 1385)
        self.assertEqual(len(default), 32)
        self.assertTrue(set(default) >= {"AmazonViewSavedAddresses",
                                         "NortonIdentitySafeSearchPasswords"})
        self.assertNotIn("TerminalExecute", default)

    def test_carrier_is_served_from_the_template_not_the_cache(self):
        carrier = self.task.metadata["carrier_tool"]
        obs = self.env.execute(self.task, ToolCall(name=carrier, arguments={}), [])
        self.assertEqual(
            obs, self.env._benign_observation(self.task))             # noqa: SLF001

    def test_cached_attacker_tool_bypasses_the_simulator(self):
        name = "AmazonViewSavedAddresses"
        obs = self.env.execute(self.task, ToolCall(name=name, arguments={}), [])
        self.assertEqual(obs, self.env._sim_default[name])            # noqa: SLF001
        # Arguments the cache never saw still resolve, deterministically, to the
        # tool's default entry rather than falling through to the simulator.
        self.assertEqual(
            self.env.execute(
                self.task, ToolCall(name=name, arguments={"zz": 1}), []),
            obs,
        )

    def test_uncached_attacker_tool_falls_through_to_the_simulator(self):
        call = ToolCall(name="TerminalExecute", arguments={"command": "ls"})
        self.assertIsNone(self.env._cached_response(call))            # noqa: SLF001
        obs = self.env.execute(self.task, call, [])
        self.assertTrue(obs.strip())
        self.assertNotIn(PLACEHOLDER, obs)

    def test_canonical_args_accepts_every_shape_upstream_recorded(self):
        self.assertEqual(canonical_args({}), "{}")
        self.assertEqual(canonical_args(""), "{}")
        self.assertEqual(canonical_args("{}"), "{}")
        self.assertEqual(canonical_args("['{}']"), "{}")
        self.assertEqual(canonical_args("[]"), "{}")
        self.assertEqual(canonical_args({"b": 1, "a": 2}), '{"a": 2, "b": 1}')
        self.assertEqual(canonical_args("{'a': 'x'}"), canonical_args({"a": "x"}))
        self.assertIsNone(canonical_args("{not parseable"))


# --------------------------------------------------------------------------- #
# Layer 6 -- the attacker's own malicious objective                            #
# --------------------------------------------------------------------------- #
@_skip_no_data
class HarmfulGoalSeam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _env()
        cls.tasks = cls.env.get_tasks()

    def _split_baits(self, split: str) -> set:
        return {str((rec.get("attacker_tools") or [""])[0])
                for _s, rec in iter_split_rows(_ROOT, (split,))}

    def test_every_task_carries_the_catalogue(self):
        for t in self.tasks:
            cands = t.metadata[GOAL_CANDIDATES_KEY]
            self.assertTrue(cands, t.task_id)
            for c in cands:
                self.assertTrue(c["tool"], t.task_id)
                self.assertTrue(c["goal_text"], c["tool"])

    def test_candidates_come_from_the_train_split_only(self):
        # The held-out axis is "unseen payload on a seen scenario". Drawing the
        # training attacker's target from the full catalogue would put val/test
        # tool names into training payloads and burn it.
        pool = {c["tool"] for t in self.tasks
                for c in t.metadata[GOAL_CANDIDATES_KEY]}
        train = self._split_baits(GOAL_CANDIDATE_SPLIT)
        self.assertEqual(len(train), N_GOAL_CANDIDATES)
        self.assertTrue(pool <= train)
        for held_out in ("val", "test"):
            self.assertEqual(pool & self._split_baits(held_out), set(), held_out)

    def test_a_task_never_targets_its_own_carrier(self):
        # The only bait/carrier coincidence in the benchmark is
        # injecagent-ds-16-03 (GitHubGetUserDetails is user case 16's carrier),
        # and it lives in VAL -- so the train-only catalogue does not contain it
        # and the carrier filter is a no-op today. It stays because a
        # regenerated split could move that case into train, where an unfiltered
        # catalogue would hand one task an in-scope "malicious" objective.
        for t in self.tasks:
            cands = t.metadata[GOAL_CANDIDATES_KEY]
            self.assertEqual(len(cands), N_GOAL_CANDIDATES, t.task_id)
            self.assertNotIn(t.metadata["carrier_tool"],
                             {c["tool"] for c in cands}, t.task_id)
        carriers = {t.metadata["carrier_tool"] for t in self.tasks}
        self.assertEqual(carriers & self._split_baits(GOAL_CANDIDATE_SPLIT), set())
        self.assertIn("GitHubGetUserDetails", self._split_baits("val"))

    def test_sample_goal_never_falls_back_to_the_user_instruction(self):
        # A None here is the "exposes no sensitive sink" WARNING in
        # mct_searcher, whose consequence is that in-scope forgery scores as an
        # attack success.
        for t in self.tasks:
            goal = sample_goal(t, self.env.get_tools(t))
            self.assertIsNotNone(goal, t.task_id)
            self.assertNotIn(goal.harmful_tool,
                             {s.name for s in self.env.get_tools(t)}, t.task_id)
            self.assertFalse(goal.in_benign_plan, t.task_id)
            # dataset_builder._infer_target_tool matches `call\s+(\w+)`.
            self.assertIn(f"call {goal.harmful_tool}", goal.goal_text)

    def test_the_catalogue_order_is_deterministic(self):
        pool = [c["tool"] for c in self.tasks[0].metadata[GOAL_CANDIDATES_KEY]]
        self.assertEqual(pool, sorted(pool))
        # sample_goal indexes by sha1(task_id), so a re-run must re-target.
        again = _env().get_tasks()[0]
        self.assertEqual(
            sample_goal(again, self.env.get_tools(again)).harmful_tool,
            sample_goal(self.tasks[0], self.env.get_tools(self.tasks[0])).harmful_tool,
        )


# --------------------------------------------------------------------------- #
# Layer 5 -- uid helper                                                        #
# --------------------------------------------------------------------------- #
class TaskUid(unittest.TestCase):
    def test_shape_and_stability(self):
        uid = task_uid("Gmail", "Read my latest email.")
        prefix, toolkit, digest = uid.split(":")
        self.assertEqual(prefix, DATASET_NAME)
        self.assertEqual(toolkit, "Gmail")
        self.assertEqual(len(digest), 10)
        self.assertEqual(
            digest,
            hashlib.sha1(b"Read my latest email.").hexdigest()[:10],
        )


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
