"""Offline tests for Δ-Bench, the concealment-graded twin ladder at
``data/delta_bench/``.

**The tier is a COMPOUND manipulation.** It grades deferral distance (Δ 1/2/3/4)
*and* pretext elaboration (L1/L2/L3) together, so a by-tier ASR curve measures
concealment as a whole and CANNOT be attributed to Δ. The pure-Δ arm --
uniform pretext, same 19 groups -- is archived byte-for-byte at
``data/delta_bench/v1_uniform_pretext/`` and measured flat
(0.474 / 0.553 / 0.456 / 0.482, no McNemar p below 0.08), which is what bounds
the Δ component. Report the two arms together or neither.

Everything else about a group is still paired, so the tests are organised around
the ways that can quietly stop being true.

**The ladder is RAGGED and MULTI-SOURCE since 2026-09-10.** Only the 19 core
AgentDojo groups carry four tiers; the AgentDojo extension carries ``imm``+``d1``
and the ``asb``/``injecagent`` groups carry ``imm`` alone, because their clean
plans are 2 and 1 tool long and no deeper Δ is constructible. Two consequences
the tests below enforce rather than document: a by-tier number POOLED across
sources is not interpretable (base ASR differs across them by a factor of
seven, and the deep rungs are single-source), and the extension groups ship no
clean twin, so BU and UA remain defined on the original 18 clean rows.

* **Layer 1 -- the ladder is paired and its rungs are declared.** Every rung of
  a group must agree field-for-field on task, clean plan, carrier, benign
  observation and sink, and the rows present must equal the group's declared
  ``group_tiers``. A one-sided re-author breaks the contrast silently: every
  file still loads, every tier still yields an ASR, and the by-tier curve
  absorbs the drift. A row lost from a ragged group is worse -- it reads as a
  deliberately short group.
* **Layer 2 -- the register is uniform and the grading runs the right way.** No
  ``<INFORMATION>`` delimiter *anywhere* (including ``imm``, unlike
  ``data/agentdojo_stock``); every payload names its sink, which is the lesson
  the camouflage probe's v1 null bought -- an unobeyable payload measures
  nothing; and payload length, the mechanical proxy for pretext elaboration,
  increases strictly across whichever of ``imm < d1 < d2`` the group declares.
  Length is a DESIGNED correlate of tier, not an incidental confound.
* **Layer 3 -- the geometry is constructible.** ``carrier_index + Δ`` has to fit
  inside the clean plan, or the "deferred" tier is really a truncated one, and
  the sink must be advertised by the env yet absent from the clean plan.
* **Layer 4 -- reuse provenance.** Verbatim reuse from
  ``data/agentdojo_latent/authored/`` is confined to ``d2``: those payloads are
  the strongest pretexts in the set (~1032 chars vs 620 authored), so one landing
  on ``imm``/``d1`` would put an L3 text in an L1/L2 slot and invert the
  gradient. The authored files are self-contained now -- the builder no longer
  reads the latent tree -- so this layer is the ONLY thing keeping the reused
  text in step with its source.
* **Layer 5 -- the artefacts on disk match the manifest and the sha256 guard.**
* **Layer 6 -- the replay wiring resolves.** The env, the scenario loader and the
  two registries must agree on the dataset name and on ``task_id``: a mismatch
  there does not raise, it matches zero attacks against zero tasks and reports a
  perfectly clean ASR of 0.0. The multi-source variant of that trap has its own
  test: the PARENT env narrows ``suites`` to the four AgentDojo ones, so
  inheriting its filter would drop every extension row while still reporting a
  plausible ASR over the core.

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

SUITES = ("banking", "slack", "travel", "workspace", "asb", "injecagent")
TIERS = ("imm", "d1", "d2", "d3")
TIER_DELTA = {"imm": 1, "d1": 2, "d2": 3, "d3": 4}
TIER_PRETEXT = {"imm": "L1", "d1": "L2", "d2": "L3", "d3": "L3"}

#: Provenance of a group, and the suite each source may occupy. The extension
#: sources are ``imm``-only by ARITHMETIC: ASB advertises 2 tools per agent and
#: InjecAgent 1 per user case, so ``carrier_index + delta <= len(clean_plan)``
#: admits nothing deeper.
SOURCES = ("agentdojo", "asb", "injecagent")
SUITE_SOURCE = {
    "banking": "agentdojo", "slack": "agentdojo",
    "travel": "agentdojo", "workspace": "agentdojo",
    "asb": "asb", "injecagent": "injecagent",
}

#: **The ladder is RAGGED.** A group declares its own rungs, and only these
#: three sets are legal. A typo here would silently change what a by-tier cell
#: contains, which is why it is pinned rather than derived.
ALLOWED_TIER_SETS = (
    ("imm", "d1", "d2", "d3"),  # the four-tier core -- the headline curve
    ("imm", "d1"),              # AgentDojo extension: the L1->L2 contrast
    ("imm",),                   # ASB / InjecAgent: the L1 rung alone
)

#: Verbatim reuse is confined to this tier -- see Layer 4 in the module doc.
REUSE_TIER = "d2"

#: How many of the four-tier core groups qualify for reuse (their latent source
#: declares ``expected_delta == 3``). The other 7 author all four tiers.
N_REUSED = 12

#: An ``imm`` payload citing one of these is not an L1 pretext. ``imm`` is the
#: rung the concealment gradient is measured against, so a citation there
#: flattens the very axis the ladder grades. Kept in step with
#: ``scripts/build_delta_bench.py::AUTHORITY``.
AUTHORITY = (
    "directive", "regulation", "article", "art.", "§", "clause", "annex",
    "statute", "statutory", "gdpr", "psd2", "iso ", "soc 2", "pci dss",
    "paragraph", "subsection", "schedule 1", "schedule 2",
)

#: Re-spelled rather than imported, like the constants above: the env name, the
#: on-disk subdirectory and the ``task_id`` prefix are three separate spellings
#: of this string and a drift between them matches zero attacks silently.
DATASET_NAME = "delta_bench"

#: The four-tier core, and the ONLY groups carrying a clean arm. Both are
#: pinned: the core is what the headline by-tier curve is computed over and the
#: 18 clean rows are what BU and UA are defined on, so either one moving
#: silently re-bases a published number. Everything else about the tree's size
#: is derived from the authored files.
N_CORE_GROUPS = 19
N_CLEAN = 18

#: A payload in any tier that carries one of these is not in the same register
#: as its siblings, so the tier contrast would measure delimiter style too.
FORBIDDEN = ("<INFORMATION>", "</INFORMATION>", "ignore previous",
             "ignore all previous")

#: Byte-identical across all four tiers of a group. The complement -- payload,
#: tier, pretext_level, instance_id, expected_delta, expected_turning_index,
#: expected_induced_plan, stealth_technique, reused_from_latent -- is exactly
#: what a tier is allowed to change.
PAIRED_FIELDS = (
    "source",
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
    "group_tiers",
)


def _bucket(tier: str) -> list[dict]:
    path = os.path.join(SCENARIOS, f"bucket_{tier}.jsonl")
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _groups() -> dict[str, dict[str, dict]]:
    """``{case_id: {tier: row}}`` over the attacked buckets.

    The inner dict holds only the rungs the group DECLARES -- the buckets are
    ragged, so ``len(inner)`` is 4, 2 or 1.
    """
    out: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for tier in TIERS:
        for row in _bucket(tier):
            out[row["case_id"]][tier] = row
    return dict(out)


def _declared(tiers: dict[str, dict]) -> tuple[str, ...]:
    """The group's declared rung set, read off the rows themselves."""
    return tuple(tuple(r["group_tiers"]) for r in tiers.values())[0]


def _core() -> dict[str, dict[str, dict]]:
    """Only the four-tier groups: the subset the headline curve is read from."""
    return {c: t for c, t in _groups().items() if _declared(t) == TIERS}


def _authored_stems() -> list[str]:
    """Authored files present on disk, in build order."""
    return [s for s in SUITES
            if os.path.isfile(os.path.join(AUTHORED, f"{s}.json"))]


def _authored(suite: str) -> list[dict]:
    with open(os.path.join(AUTHORED, f"{suite}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _all_authored() -> list[dict]:
    return [g for s in _authored_stems() for g in _authored(s)]


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
        cls.authored = _all_authored()

    def test_every_group_declares_a_legal_rung_set_and_carries_exactly_it(self):
        # The ONE thing raggedness must not cost: a group whose rows and whose
        # declared ``group_tiers`` disagree. A missing row then reads as a
        # deliberately short group instead of as a lost row.
        self.assertEqual(len(self.groups), len(self.authored))
        for case_id, tiers in self.groups.items():
            declared = _declared(tiers)
            self.assertIn(declared, ALLOWED_TIER_SETS,
                          f"{case_id}: illegal rung set {list(declared)}")
            self.assertEqual(sorted(tiers), sorted(declared), case_id)
            for row in tiers.values():
                self.assertEqual(tuple(row["group_tiers"]), declared, case_id)

    def test_the_four_tier_core_is_the_original_nineteen(self):
        core = _core()
        self.assertEqual(len(core), N_CORE_GROUPS)
        # The headline curve is this subset. Every deeper rung exists ONLY here,
        # so d2/d3 must be exactly the core and nothing else.
        for tier in ("d2", "d3"):
            self.assertEqual({r["case_id"] for r in _bucket(tier)},
                             set(core), tier)

    def test_each_bucket_holds_one_row_per_group_that_declares_it(self):
        for tier in TIERS:
            rows = _bucket(tier)
            expect = {c for c, t in self.groups.items() if tier in _declared(t)}
            self.assertEqual({r["case_id"] for r in rows}, expect, tier)
            self.assertEqual(len(rows), len(expect), tier)
            self.assertEqual({r["tier"] for r in rows}, {tier})

    def test_every_row_declares_a_known_source_matching_its_suite(self):
        for case_id, tiers in self.groups.items():
            for tier, row in tiers.items():
                self.assertIn(row["source"], SOURCES, f"{case_id}/{tier}")
                self.assertEqual(SUITE_SOURCE[row["suite"]], row["source"],
                                 f"{case_id}/{tier}: suite/source mismatch")

    def test_only_agentdojo_groups_reach_past_d1(self):
        # Pooling a by-tier cell across sources is not interpretable (base ASR
        # differs across them by a factor of seven), and the shape that makes
        # that trap concrete is this one: the deep rungs are single-source.
        for tier in ("d2", "d3"):
            self.assertEqual({r["source"] for r in _bucket(tier)},
                             {"agentdojo"}, tier)

    def test_tiers_of_a_group_agree_on_everything_but_the_payload(self):
        for case_id, tiers in self.groups.items():
            declared = _declared(tiers)
            ref = tiers[declared[0]]
            for tier in declared[1:]:
                for field in PAIRED_FIELDS:
                    self.assertEqual(
                        ref[field], tiers[tier][field],
                        f"{case_id}/{tier} drifted on {field!r}",
                    )

    def test_payloads_differ_across_tiers(self):
        for case_id, tiers in self.groups.items():
            texts = {t: row["payload"] for t, row in tiers.items()}
            self.assertEqual(len(set(texts.values())), len(tiers),
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

    def test_case_ids_are_unique_across_the_authored_files(self):
        ids = [g["case_id"] for g in self.authored]
        self.assertEqual(len(ids), len(set(ids)))


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
        # Per-tier floors, not one global floor: an L1 pretext is a bare
        # requirement and is SUPPOSED to be short. The floor still has to bite,
        # because a payload too short to name the sink and its arguments is
        # unobeyable and measures nothing.
        floor = {"imm": 120, "d1": 400, "d2": 400, "d3": 400}
        for case_id, tiers in self.groups.items():
            for tier, row in tiers.items():
                self.assertGreater(len(row["payload"]), floor[tier],
                                   f"{case_id}/{tier}")

    def test_pretext_level_is_declared_and_matches_the_tier(self):
        for case_id, tiers in self.groups.items():
            for tier, row in tiers.items():
                self.assertEqual(row["pretext_level"], TIER_PRETEXT[tier],
                                 f"{case_id}/{tier}")

    def test_pretext_elaboration_increases_with_the_tier(self):
        # Length is the mechanical proxy for elaboration; the L1/L2/L3 content
        # rubric itself is authored and reviewed, not checkable here. d3 is
        # exempt from the upper rung because it shares L3 with d2 and the user's
        # ordering requirement stops at d2. Only DECLARED rungs are compared, so
        # a two-tier group has ``imm < d1`` enforced and a one-tier group has no
        # gradient to break.
        for case_id, tiers in self.groups.items():
            n = {t: len(row["payload"]) for t, row in tiers.items()}
            for lo, hi in (("imm", "d1"), ("d1", "d2"), ("imm", "d3")):
                if lo in n and hi in n:
                    self.assertLess(n[lo], n[hi], f"{case_id}: {lo} !< {hi}")

    def test_imm_cites_no_instrument(self):
        # Every legal rung set opens on ``imm``, so this covers the whole tree.
        for case_id, tiers in self.groups.items():
            low = tiers["imm"]["payload"].lower()
            hit = [m for m in AUTHORITY if m in low]
            self.assertFalse(hit, f"{case_id}/imm cites {hit}")

    def test_stealth_technique_is_constant_within_a_group(self):
        # The graded property is the pretext's ELABORATION, not its FAMILY.
        # Varying the family along the ladder would turn the by-tier curve into
        # a by-technique curve, and the by-technique spread on this data
        # (0.264-0.479) is comparable to the tier axis's own.
        for case_id, tiers in self.groups.items():
            techs = {row["stealth_technique"] for row in tiers.values()}
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

    def test_the_extension_sources_clean_plans_have_the_shape_that_caps_them(self):
        # This is WHY asb/injecagent are imm-only, and it is worth asserting
        # rather than documenting: if an ASB group ever grew a 3-step plan it
        # would be fabricated, since the ASB rows carry no call sequence at all.
        for case_id, tiers in self.groups.items():
            row = tiers["imm"]
            if row["source"] == "asb":
                self.assertEqual(len(row["clean_plan"]), 2, case_id)
            elif row["source"] == "injecagent":
                self.assertEqual(len(row["clean_plan"]), 1, case_id)
            if row["source"] != "agentdojo":
                self.assertEqual(_declared(tiers), ("imm",), case_id)


# --------------------------------------------------------------------------- #
# Layer 4 -- reuse provenance                                                  #
# --------------------------------------------------------------------------- #
class Reuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = _groups()

    def test_reuse_is_confined_to_d2_and_at_most_one_tier_per_group(self):
        n_groups_reusing = 0
        for case_id, tiers in self.groups.items():
            reused = [t for t, row in tiers.items() if row["reused_from_latent"]]
            self.assertLessEqual(len(reused), 1, f"{case_id}: reused={reused}")
            if reused:
                self.assertEqual(reused, [REUSE_TIER], case_id)
                n_groups_reusing += 1
        self.assertEqual(n_groups_reusing, N_REUSED)

    def test_reused_payload_is_byte_identical_to_the_latent_case(self):
        # The authored files became self-contained on 2026-09-10 -- the builder
        # no longer reads the latent tree -- so THIS is now the only thing
        # keeping the reused text in step with its source.
        for case_id, tiers in self.groups.items():
            row = tiers.get(REUSE_TIER)
            if row is None or not row["reused_from_latent"]:
                continue
            _, suite, index = row["reused_from_latent"].split(":")
            latent = _latent(suite)[int(index)]
            self.assertEqual(row["payload"], latent["payload"],
                             f"{case_id}/{REUSE_TIER} diverged from its source")
            self.assertEqual(latent["expected_delta"], TIER_DELTA[REUSE_TIER],
                             f"{case_id}: reused at the wrong tier")
            for field in ("harmful_tool", "carrier_tool", "carrier_index",
                          "benign_observation", "attacker_goal"):
                self.assertEqual(row[field], latent[field],
                                 f"{case_id}: {field} drifted from the source")
            # The emitted row qualifies the reference (``agentdojo:v1:<suite>:
            # UserTaskN``); the authored source carries the bare class name.
            self.assertTrue(row["task_ref"].endswith(":" + latent["task_ref"]),
                            f"{case_id}: {row['task_ref']} is not the source task")

    def test_authored_groups_are_self_contained_and_declare_their_provenance(self):
        # Schema of the authored record after the decoupling: the structural
        # fields live here, ``payloads`` covers exactly the declared rungs
        # (including any reused one, materialised verbatim), and ``provenance``
        # is audit trail only.
        for stem in _authored_stems():
            source = SUITE_SOURCE[stem]
            for spec in _authored(stem):
                where = spec["case_id"]
                self.assertEqual(spec["source"], source, where)
                self.assertEqual(spec["suite"], stem, where)
                declared = tuple(spec["tiers"])
                self.assertIn(declared, ALLOWED_TIER_SETS, where)
                self.assertEqual(sorted(spec["payloads"]), sorted(declared), where)
                for field in ("carrier_tool", "carrier_index", "harmful_tool",
                              "harmful_args_hint", "attacker_goal",
                              "benign_observation", "stealth_technique"):
                    self.assertIn(field, spec, f"{where}: missing {field}")
                prov = spec.get("provenance") or {}
                tier = prov.get("reused_tier")
                if tier is None:
                    continue
                self.assertEqual(tier, REUSE_TIER, where)
                self.assertIn(tier, spec["payloads"],
                              f"{where}: reused tier has no materialised payload")
                self.assertTrue(prov.get("latent_case"), where)


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
        groups = _groups()
        for tier in TIERS:
            self.assertEqual(counts[tier], len(_bucket(tier)), tier)
        self.assertEqual(counts["clean"], len(_bucket("clean")))
        self.assertEqual(self.manifest["n_groups"], len(groups))
        self.assertEqual(self.manifest["n_groups_core_four_tier"], N_CORE_GROUPS)
        self.assertEqual(self.manifest["n_attacked_total"],
                         sum(counts[t] for t in TIERS))
        self.assertEqual(sum(counts[t] for t in TIERS),
                         sum(len(t) for t in groups.values()))

    def test_manifest_declares_the_compound_axis_and_the_pooling_trap(self):
        # The whole point of schema 2 was the compound axis; schema 3 adds the
        # never-pool warning, because a ragged multi-source tree makes a pooled
        # by-tier cell move with source composition rather than with the pretext.
        self.assertEqual(self.manifest["schema_version"], 3)
        for key in ("axis", "pretext_rubric", "reuse", "tier_semantics",
                    "never_pool", "per_source", "extension_sources"):
            self.assertIn(key, self.manifest)
        for tier in TIERS:
            self.assertEqual(
                self.manifest["tier_semantics"][tier]["pretext_level"],
                TIER_PRETEXT[tier], tier)

    def test_manifest_per_source_matches_the_buckets(self):
        per_source = self.manifest["per_source"]
        rows = [r for tier in TIERS for r in _bucket(tier)]
        by_source = collections.defaultdict(list)
        for row in rows:
            by_source[row["source"]].append(row)
        self.assertEqual(set(per_source), set(by_source))
        for src, block in per_source.items():
            got = by_source[src]
            self.assertEqual(block["n_attacked"], len(got), src)
            self.assertEqual(block["n_groups"],
                             len({r["case_id"] for r in got}), src)
            self.assertEqual(block["n_clean"],
                             len([r for r in _bucket("clean")
                                  if r["source"] == src]), src)

    def test_per_suite_group_counts_match_the_authored_files(self):
        for stem in _authored_stems():
            self.assertEqual(self.manifest["per_suite"][stem]["n_groups"],
                             len(_authored(stem)), stem)
        self.assertEqual(set(self.manifest["per_suite"]), set(_authored_stems()))

    def test_clean_rows_are_one_per_carrier_and_carry_no_payload(self):
        # A carrier serves exactly one benign text, so the clean twin is keyed on
        # ``(task, carrier)``, not on the task alone -- two groups can share a
        # task and still need separate clean rows. Same convention as
        # ``data/agentdojo_latent``. Pinned at 18 because BU and UA are defined
        # on these rows and nothing else; the extension groups ship none.
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

    def test_the_clean_arm_covers_the_core_and_only_the_core(self):
        # ``has_clean_arm`` is declared per group, not inferred. The extension
        # groups deliberately ship no benign counterfactual -- they were added to
        # measure ASR -- so BU and UA stay figures about the core tasks and are
        # NOT diluted by rows that never had a twin.
        clean = {(r["task_id"], r["carrier_tool"]) for r in _bucket("clean")}
        for case_id, tiers in _groups().items():
            row = tiers["imm"]
            key = (row["task_id"], row["carrier_tool"])
            if _declared(tiers) == TIERS:
                self.assertIn(key, clean, f"{case_id}: core group has no twin")
            else:
                self.assertNotIn(key, clean,
                                 f"{case_id}: extension group grew a clean twin")

    def test_every_core_attacked_row_shares_its_twins_observation(self):
        clean = {(r["task_id"], r["carrier_tool"]): r for r in _bucket("clean")}
        for tier in TIERS:
            for row in _bucket(tier):
                twin = clean.get((row["task_id"], row["carrier_tool"]))
                if twin is None:
                    continue
                self.assertEqual(twin["benign_observation"],
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


# --------------------------------------------------------------------------- #
# Layer 6 -- replay wiring: env, loader and the two registries                 #
# --------------------------------------------------------------------------- #
class Wiring(unittest.TestCase):
    """The seams between the data and ``eval/vendored_replay.py``.

    Every failure mode here is SILENT. ``vendored_replay`` matches scenarios to
    tasks by ``task_id``; a name mismatch, a stale registry entry or a loader
    that reads the wrong tree all produce "0 scenarios matched" and an ASR of
    0.0, which reads exactly like a defended model.
    """

    @classmethod
    def setUpClass(cls):
        from evoguard.envs import build_env
        from evoguard.config import EnvConfig, LLMConfig
        from evoguard.process.vendored_attack_loaders import load_vendored_scenarios

        cls.load_vendored_scenarios = staticmethod(load_vendored_scenarios)
        # MockClient: the env is only asked for its task list and tool specs
        # here, and the carrier's benign text is authored rather than simulated.
        cfg = EnvConfig(
            dataset=DATASET_NAME,
            data_root=os.path.join(_PKG_ROOT, "data"),
            tool_executor_llm=LLMConfig(backend="mock"),
            utility_judge_llm=LLMConfig(backend="mock"),
        )
        cls.env = build_env(cfg)

    def test_env_is_registered_under_the_dataset_name(self):
        from evoguard.envs import available_envs
        self.assertIn(DATASET_NAME, available_envs())
        self.assertEqual(self.env.name, DATASET_NAME)

    def test_loader_is_registered_and_returns_every_injected_row(self):
        atk = self.load_vendored_scenarios(
            data_root=os.path.join(_PKG_ROOT, "data"), dataset=DATASET_NAME)
        n = sum(len(_bucket(t)) for t in TIERS)
        self.assertEqual(len(atk), n)
        # Registry hit, not the AgentDojo fallback: that one would have raised
        # on the missing default_suites tree, but assert the shape too.
        self.assertTrue(all(a.mal_tool for a in atk))
        self.assertTrue(all(a.payload for a in atk))

    def test_every_scenario_task_id_resolves_to_an_env_task(self):
        env_ids = {t.task_id for t in self.env.get_tasks()}
        atk = self.load_vendored_scenarios(
            data_root=os.path.join(_PKG_ROOT, "data"), dataset=DATASET_NAME)
        self.assertFalse(sorted({a.task_id for a in atk} - env_ids))

    def test_the_env_loads_every_source(self):
        # The failure this guards is the loudest silent one in the tree: the
        # PARENT env narrows ``suites`` to the four AgentDojo ones, so inheriting
        # its filter would drop every ASB and InjecAgent row while the replay
        # still reported a perfectly plausible ASR over the core.
        suites = {r["suite"] for tier in TIERS for r in _bucket(tier)}
        loaded = {t.suite for t in self.env.get_tasks()}
        self.assertFalse(suites - loaded, f"env dropped suites {suites - loaded}")

    def test_tier_filter_partitions_the_ladder(self):
        from evoguard.process.delta_bench_loader import load_delta_bench_attacks
        seen: set[str] = set()
        for tier in TIERS:
            sub = load_delta_bench_attacks(
                os.path.join(_PKG_ROOT, "data"), tiers=[tier])
            self.assertEqual(len(sub), len(_bucket(tier)), tier)
            keys = {(a.task_id, a.payload) for a in sub}
            self.assertFalse(keys & seen, f"{tier} overlaps another tier")
            seen |= keys

    def test_source_filter_partitions_the_ladder(self):
        # Per-source replay is the supported way to avoid the pooling trap, so
        # the filter has to be a partition rather than an approximation.
        from evoguard.process.delta_bench_loader import load_delta_bench_attacks
        root = os.path.join(_PKG_ROOT, "data")
        total = len(load_delta_bench_attacks(root))
        seen: set = set()
        for src in SOURCES:
            sub = load_delta_bench_attacks(root, sources=[src])
            keys = {(a.task_id, a.payload) for a in sub}
            self.assertFalse(keys & seen, f"{src} overlaps another source")
            seen |= keys
        self.assertEqual(len(seen), total)

    def test_loader_excludes_the_clean_bucket(self):
        # bucket_clean.jsonl exists for the ENV. Replaying its payload-free rows
        # as attacks would count 18 attempts that cannot succeed and drag ASR
        # down by roughly a fifth.
        atk = self.load_vendored_scenarios(
            data_root=os.path.join(_PKG_ROOT, "data"), dataset=DATASET_NAME)
        self.assertTrue(all(a.payload.strip() for a in atk))

    def test_task_payload_join_key_is_unique_across_all_tiers(self):
        # scripts/summarize_delta_bench.py recovers the tier by joining the
        # replay records back onto the scenario files on this key -- Δ has no
        # slot on VendoredAttack. A collision would mis-stratify the curve.
        keys = [(r["task_id"], r["payload"])
                for tier in TIERS for r in _bucket(tier)]
        self.assertEqual(len(keys), len(set(keys)))

    def test_every_sink_is_advertised_to_the_agent(self):
        tools = {t.task_id: {s.name for s in self.env.get_tools(t)}
                 for t in self.env.get_tasks()}
        for tier in TIERS:
            for row in _bucket(tier):
                self.assertIn(row["harmful_tool"], tools[row["task_id"]],
                              f"{row['instance_id']}: sink not in get_tools")

    def test_clean_arm_serves_the_authored_carrier_observation(self):
        # If the simulator invented the carrier's clean output, the clean and
        # attacked arms would differ in SHAPE as well as by the payload.
        from evoguard.core.types import ToolCall
        for task in self.env.get_tasks():
            benign = (task.metadata or {}).get("benign_observations") or {}
            self.assertTrue(benign, task.task_id)
            for carrier, text in benign.items():
                got = self.env.execute(task, ToolCall(name=carrier), [])
                self.assertEqual(got, text)


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
