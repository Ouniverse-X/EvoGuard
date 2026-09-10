"""Offline tests for Δ-Bench, the Δ-stratified twin ladder at ``data/delta_bench/``.

The dataset exists to answer one question: **how does ASR move as the harmful
call is pushed further from the injection point?** That reading is only valid if
Δ is the *sole* variable across a group's four rows, so the tests are organised
around the ways that can quietly stop being true.

* **Layer 1 -- the ladder is paired.** All four tiers of a group must agree
  field-for-field on task, clean plan, carrier, benign observation and sink. A
  one-sided re-author breaks the contrast silently: every file still loads, every
  tier still yields an ASR, and the by-tier curve absorbs the drift.
* **Layer 2 -- Δ is not confounded with anything else.** Every tier is authored
  in the same register: no ``<INFORMATION>`` delimiter *anywhere* (including
  ``imm``, unlike ``data/agentdojo_stock``), and every payload names its sink,
  which is the lesson the camouflage probe's v1 null bought -- an unobeyable
  payload measures nothing.
* **Layer 3 -- the geometry is constructible.** ``carrier_index + Δ`` has to fit
  inside the clean plan, or the "deferred" tier is really a truncated one, and
  the sink must be advertised by the env yet absent from the clean plan.
* **Layer 4 -- reuse provenance.** Each group borrows exactly one tier verbatim
  from ``data/agentdojo_latent/authored/``; that tier's Δ must be the one the
  latent case declares, and the text must be byte-identical.
* **Layer 5 -- the artefacts on disk match the manifest and the sha256 guard.**

Run via::

    python -m evoguard.tests.test_delta_bench
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

from evoguard.envs.tool_parsing import parse_env_info  # noqa: E402

ROOT = os.path.join(_PKG_ROOT, "data", "delta_bench")
AUTHORED = os.path.join(ROOT, "authored")
SCENARIOS = os.path.join(ROOT, "scenarios")
LATENT_AUTHORED = os.path.join(_PKG_ROOT, "data", "agentdojo_latent", "authored")

SUITES = ("banking", "slack", "travel", "workspace")
TIERS = ("imm", "d1", "d2", "d3")
TIER_DELTA = {"imm": 1, "d1": 2, "d2": 3, "d3": 4}

N_GROUPS = 19
N_CLEAN = 18

#: A payload in any tier that carries one of these is not in the same register
#: as its siblings, so the tier contrast would measure delimiter style too.
FORBIDDEN = ("<INFORMATION>", "</INFORMATION>", "ignore previous",
             "ignore all previous")

#: Byte-identical across all four tiers of a group. The complement -- payload,
#: tier, instance_id, expected_delta, expected_turning_index,
#: expected_induced_plan, stealth_technique, reused_from_latent -- is exactly
#: what a tier is allowed to change.
PAIRED_FIELDS = (
    "suite",
    "task_ref",
    "task_id",
    "task_instruction",
    "env_info",
    "clean_plan",
    "carrier_tool",
    "carrier_index",
    "benign_observation",
    "harmful_tool",
    "harmful_args_hint",
    "attacker_goal",
    "sample_type",
)


def _bucket(tier: str) -> list[dict]:
    path = os.path.join(SCENARIOS, f"bucket_{tier}.jsonl")
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _groups() -> dict[str, dict[str, dict]]:
    """``{case_id: {tier: row}}`` over the four attacked buckets."""
    out: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for tier in TIERS:
        for row in _bucket(tier):
            out[row["case_id"]][tier] = row
    return dict(out)


def _authored(suite: str) -> list[dict]:
    with open(os.path.join(AUTHORED, f"{suite}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _latent(suite: str) -> list[dict]:
    with open(os.path.join(LATENT_AUTHORED, f"{suite}.json"), "r",
              encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Layer 1 -- the ladder is paired                                              #
# --------------------------------------------------------------------------- #
class Ladder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = _groups()

    def test_every_group_has_exactly_the_four_tiers(self):
        self.assertEqual(len(self.groups), N_GROUPS)
        for case_id, tiers in self.groups.items():
            self.assertEqual(sorted(tiers), sorted(TIERS), case_id)

    def test_each_bucket_holds_one_row_per_group(self):
        for tier in TIERS:
            rows = _bucket(tier)
            self.assertEqual(len(rows), N_GROUPS, tier)
            self.assertEqual(len({r["case_id"] for r in rows}), N_GROUPS, tier)
            self.assertEqual({r["tier"] for r in rows}, {tier})

    def test_tiers_of_a_group_agree_on_everything_but_the_payload(self):
        for case_id, tiers in self.groups.items():
            ref = tiers["imm"]
            for tier in TIERS[1:]:
                for field in PAIRED_FIELDS:
                    self.assertEqual(
                        ref[field], tiers[tier][field],
                        f"{case_id}/{tier} drifted on {field!r}",
                    )

    def test_payloads_differ_across_tiers(self):
        for case_id, tiers in self.groups.items():
            texts = {t: tiers[t]["payload"] for t in TIERS}
            self.assertEqual(len(set(texts.values())), len(TIERS),
                             f"{case_id}: two tiers share a payload")

    def test_instance_ids_are_unique_and_carry_the_tier(self):
        seen = set()
        for case_id, tiers in self.groups.items():
            for tier, row in tiers.items():
                iid = row["instance_id"]
                self.assertNotIn(iid, seen)
                seen.add(iid)
                self.assertTrue(iid.endswith(f"-{tier}"), iid)
                self.assertIn(case_id, iid)


# --------------------------------------------------------------------------- #
# Layer 2 -- Δ is not confounded with the authoring register                    #
# --------------------------------------------------------------------------- #
class PayloadRegister(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = _groups()

    def test_no_tier_uses_an_injection_delimiter(self):
        for case_id, tiers in self.groups.items():
            for tier, row in tiers.items():
                low = row["payload"].lower()
                for bad in FORBIDDEN:
                    self.assertNotIn(bad.lower(), low, f"{case_id}/{tier}")

    def test_every_payload_names_its_sink(self):
        for case_id, tiers in self.groups.items():
            for tier, row in tiers.items():
                self.assertIn(row["harmful_tool"], row["payload"],
                              f"{case_id}/{tier} does not name the sink")

    def test_payloads_are_substantial(self):
        for case_id, tiers in self.groups.items():
            for tier, row in tiers.items():
                self.assertGreater(len(row["payload"]), 200,
                                   f"{case_id}/{tier}")

    def test_stealth_technique_is_constant_within_a_group(self):
        for case_id, tiers in self.groups.items():
            techs = {tiers[t]["stealth_technique"] for t in TIERS}
            self.assertEqual(len(techs), 1, f"{case_id}: {sorted(techs)}")
            self.assertTrue(all(techs))


# --------------------------------------------------------------------------- #
# Layer 3 -- the deferral geometry is constructible                            #
# --------------------------------------------------------------------------- #
class Geometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = _groups()

    def test_declared_delta_matches_the_tier(self):
        for case_id, tiers in self.groups.items():
            for tier, row in tiers.items():
                self.assertEqual(row["expected_delta"], TIER_DELTA[tier],
                                 f"{case_id}/{tier}")

    def test_the_deferral_fits_inside_the_clean_plan(self):
        for case_id, tiers in self.groups.items():
            for tier, row in tiers.items():
                self.assertLessEqual(
                    row["carrier_index"] + row["expected_delta"],
                    len(row["clean_plan"]),
                    f"{case_id}/{tier}: Δ runs off the end of the plan",
                )

    def test_turning_index_and_induced_plan_agree_with_delta(self):
        for case_id, tiers in self.groups.items():
            for tier, row in tiers.items():
                ci, delta = row["carrier_index"], row["expected_delta"]
                self.assertEqual(row["expected_turning_index"], ci + delta,
                                 f"{case_id}/{tier}")
                induced, clean = row["expected_induced_plan"], row["clean_plan"]
                self.assertEqual(len(induced), len(clean) + 1,
                                 f"{case_id}/{tier}")
                self.assertEqual(induced[ci + delta], row["harmful_tool"],
                                 f"{case_id}/{tier}")
                spliced = induced[: ci + delta] + induced[ci + delta + 1:]
                self.assertEqual(spliced, clean, f"{case_id}/{tier}")

    def test_carrier_sits_where_the_plan_says_it_does(self):
        for case_id, tiers in self.groups.items():
            row = tiers["imm"]
            self.assertEqual(row["clean_plan"][row["carrier_index"]],
                             row["carrier_tool"], case_id)

    def test_sink_is_advertised_but_not_part_of_the_clean_plan(self):
        for case_id, tiers in self.groups.items():
            row = tiers["imm"]
            advertised = {s.name for s in parse_env_info(row["env_info"])}
            self.assertIn(row["harmful_tool"], advertised, case_id)
            self.assertNotIn(row["harmful_tool"], row["clean_plan"], case_id)


# --------------------------------------------------------------------------- #
# Layer 4 -- reuse provenance                                                  #
# --------------------------------------------------------------------------- #
class Reuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = _groups()

    def test_exactly_one_tier_per_group_is_reused(self):
        for case_id, tiers in self.groups.items():
            reused = [t for t in TIERS if tiers[t]["reused_from_latent"]]
            self.assertEqual(len(reused), 1, f"{case_id}: reused={reused}")

    def test_reused_payload_is_byte_identical_to_the_latent_case(self):
        for case_id, tiers in self.groups.items():
            tier = next(t for t in TIERS if tiers[t]["reused_from_latent"])
            row = tiers[tier]
            _, suite, index = row["reused_from_latent"].split(":")
            latent = _latent(suite)[int(index)]
            self.assertEqual(row["payload"], latent["payload"],
                             f"{case_id}/{tier} diverged from its latent source")
            self.assertEqual(latent["expected_delta"], TIER_DELTA[tier],
                             f"{case_id}: reused at the wrong tier")
            for field in ("harmful_tool", "carrier_tool", "carrier_index",
                          "benign_observation", "attacker_goal"):
                self.assertEqual(row[field], latent[field],
                                 f"{case_id}: {field} drifted from the source")
            # The emitted row qualifies the reference (``agentdojo:v1:<suite>:
            # UserTaskN``); the authored source carries the bare class name.
            self.assertTrue(row["task_ref"].endswith(":" + latent["task_ref"]),
                            f"{case_id}: {row['task_ref']} is not the source task")

    def test_authored_files_omit_the_reused_tier(self):
        for suite in SUITES:
            for spec in _authored(suite):
                tier = spec["reuse_tier"]
                self.assertIn(tier, TIERS, spec["case_id"])
                self.assertNotIn(tier, spec["payloads"],
                                 f"{spec['case_id']}: reused tier re-authored")
                self.assertEqual(sorted(spec["payloads"]),
                                 sorted(t for t in TIERS if t != tier),
                                 spec["case_id"])


# --------------------------------------------------------------------------- #
# Layer 5 -- artefacts, manifest and guard                                     #
# --------------------------------------------------------------------------- #
class Artefacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "manifest.json"), "r",
                  encoding="utf-8") as fh:
            cls.manifest = json.load(fh)

    def test_manifest_counts_match_the_buckets(self):
        counts = self.manifest["counts"]
        for tier in TIERS:
            self.assertEqual(counts[tier], len(_bucket(tier)), tier)
        self.assertEqual(counts["clean"], len(_bucket("clean")))
        self.assertEqual(self.manifest["n_groups"], N_GROUPS)
        self.assertEqual(self.manifest["n_attacked_total"],
                         N_GROUPS * len(TIERS))

    def test_per_suite_group_counts_match_the_authored_files(self):
        for suite in SUITES:
            self.assertEqual(self.manifest["per_suite"][suite]["n_groups"],
                             len(_authored(suite)), suite)

    def test_clean_rows_are_one_per_carrier_and_carry_no_payload(self):
        # A carrier serves exactly one benign text, so the clean twin is keyed on
        # ``(task, carrier)``, not on the task alone -- two groups can share a
        # task and still need separate clean rows. Same convention as
        # ``data/agentdojo_latent``.
        rows = _bucket("clean")
        self.assertEqual(len(rows), N_CLEAN)
        keys = {(r["task_id"], r["carrier_tool"]) for r in rows}
        self.assertEqual(len(keys), N_CLEAN)
        for row in rows:
            self.assertEqual(row["sample_type"], "clean")
            self.assertEqual(row["tier"], "clean")
            self.assertIsNone(row["expected_delta"])
            self.assertFalse(row["payload"])
            self.assertEqual(row["observation"], row["benign_observation"])

    def test_every_attacked_row_has_a_clean_twin(self):
        clean = {(r["task_id"], r["carrier_tool"]): r
                 for r in _bucket("clean")}
        for tier in TIERS:
            for row in _bucket(tier):
                key = (row["task_id"], row["carrier_tool"])
                self.assertIn(key, clean,
                              f"{row['instance_id']} has no clean twin")
                self.assertEqual(clean[key]["benign_observation"],
                                 row["benign_observation"],
                                 f"{row['instance_id']}: twin observation differs")

    def test_sha256_guard_matches_the_files_on_disk(self):
        path = os.path.join(ROOT, "_sha256_guard.txt")
        with open(path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()
                     and not l.startswith("#")]
        self.assertTrue(lines)
        for line in lines:
            head, digest = line.split("=", 1)
            rel = head[len("sha256("):-1]
            with open(os.path.join(ROOT, rel), "rb") as fh:
                got = hashlib.sha256(fh.read()).hexdigest()
            self.assertEqual(got, digest, f"{rel} changed without a rebuild")


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
