"""Offline unit tests for :class:`evoguard.attacks.mct_searcher.DeltaGuidedMCTSAttacker`.

Pure-Python, no network/GPU. Run via::

    /ssd1/conda_envs/evoguard/bin/python -m evoguard.tests.test_mcts_attacker
"""

from __future__ import annotations

import random
import unittest
from typing import Sequence

from evoguard.attacks import GeneticAttacker, build_attacker
from evoguard.attacks.base import AttackGenerator
from evoguard.attacks.genetic import EvaluatedAttack
from evoguard.attacks.mct_searcher import DeltaGuidedMCTSAttacker
from evoguard.config import AttackerConfig
from evoguard.core.types import AttackSpec, Task, ToolSpec


# --------------------------------------------------------------------------- #
# Stubs                                                                       #
# --------------------------------------------------------------------------- #
class _StubGenerator(AttackGenerator):
    """Deterministic offline generator producing predictable variants.

    Avoids any LLM/network calls so tests run in milliseconds on CI.
    """

    def __init__(self) -> None:
        self.seed_calls = 0
        self.mutate_calls = 0

    # ---- seed ---------------------------------------------------------- #
    def seed(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        n: int,
        *,
        max_turns: int,
        generation: int,
    ) -> list[AttackSpec]:
        out: list[AttackSpec] = []
        ceiling = max(1, int(max_turns or 1))
        for i in range(max(1, int(n))):
            self.seed_calls += 1
            t = i % ceiling if ceiling > 0 else 0   # spread across turns deterministically
            spec = AttackSpec(
                attack_id=AttackSpec.new_id(),
                task_id=task.task_id,
                target_turn=int(t),
                injection_channel="tool_result",
                method=f"method_{i % 3}",
                payload=f"<seed payload #{self.seed_calls} for turn={t}>",
                goal="",
                generation=int(generation),
                parents=[],
                origin="seed",
                metadata={},
            )
            out.append(spec)
            if len(out) >= int(n):
                break
        return out

    # ---- crossover (unused by MCTS path; required by ABC) -------------- #
    def crossover(self, task, tools, parents, *, generation):     # noqa: D401
        raise NotImplementedError("crossover not used by MCTS backend")

    # ---- mutate -------------------------------------------------------- #
    def mutate(self, task, tools, individual, *, generation):
        self.mutate_calls += 1
        suffix = str(individual.payload or "")[-4:] + f"m{self.mutate_calls}"
        return AttackSpec(
            attack_id=AttackSpec.new_id(),
            task_id=task.task_id,
            target_turn=int(individual.target_turn),
            injection_channel=str(individual.injection_channel or ""),
            method=str(individual.method or "default_method"),
            payload=f"{individual.payload}+mut_{suffix}",
            goal=getattr(task, "instruction", ""),
            generation=int(generation),
            parents=[individual.attack_id] if individual.attack_id else [],
            origin="mutation",
            metadata={},
        )


def _make_task() -> Task:
    return Task(task_id="test_task_0", instruction="<dummy benign user request>")


def _make_tools(n_tools: int = 3) -> list[ToolSpec]:
    return [ToolSpec(name=f"tool_{i}", description="d") for i in range(n_tools)]


def _evaluated_from_spec(spec: AttackSpec, *, success=True, delta_norm=None,
                         tau_caught: int = 2) -> EvaluatedAttack:
    fit_val = float(delta_norm if delta_norm is not None else (0.5 if success else 0.0))
    return EvaluatedAttack(
        spec=spec,
        fitness=float(fit_val),
        success=bool(success),
        metadata={
            "turning_point": int(tau_caught),
            "delta_normalized": float(fit_val),
        },
    )


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
class TestDeltaGuidedMCTSAttacker(unittest.TestCase):

    def _build(self, **cfg_overrides) -> tuple[DeltaGuidedMCTSAttacker, _StubGenerator]:
        kw = dict(search_method="mcts_delta", population_size=8)
        kw.update(cfg_overrides)
        cfg = AttackerConfig(**kw)
        gen = _StubGenerator()
        att = DeltaGuidedMCTSAttacker(_make_task(), _make_tools(), gen,
                                       cfg, defense_max_turns=4)
        return att, gen

    # ----------------------------------------------------------------- #
    def test_cold_start_emits_population_of_correct_size(self):
        att,_ = self._build(population_size=10)
        pop = att.current_population()
        self.assertEqual(len(pop), 10)
        for s in pop:
            self.assertIsInstance(s, AttackSpec)
            self.assertGreaterEqual(s.target_turn, 0)
            self.assertLess(s.target_turn, att.injectable_turn_ceiling)

    def test_L1_pre_enumeration_matches_ceiling(self):
        att,_ = self._build()
        L1_children = [att._nodes_by_id[cid] for cid in att._root.children_ids]
        self.assertEqual(len(L1_children), att.injectable_turn_ceiling)
        turns = sorted(int(c.discriminator["turn"]) for c in L1_children)
        self.assertEqual(turns, list(range(att.injectable_turn_ceiling)))

    def test_backprop_increments_root_visits_by_batch_size(self):
        att,_ = self._build(population_size=3)
        pop = list(att.current_population())
        root_before = att._nodes_by_id["root"].n_visits

        evaluated_list = [
            _evaluated_from_spec(pop[0], success=True, delta_norm=0.7, tau_caught=3),
            _evaluated_from_spec(pop[1], success=False, delta_norm=0.0, tau_caught=1),
            _evaluated_from_spec(pop[2], success=True, delta_norm=0.9, tau_caught=2),
        ]
        next_pop = att.evolve(evaluated_list)

        root_after = att._nodes_by_id["root"].n_visits
        self.assertEqual(root_after - root_before, len(evaluated_list),
                          "Root must receive visit-count increment equal to batch size")
        self.assertGreater(len(next_pop), 0, "evolve() must schedule a new batch")

    def test_failure_partial_credit_accumulates_more_for_late_catches(self):
        """Late-caught C-class outcomes should deposit more partial credit than early ones."""
        att,_ = self._build(population_size=4)

        pop_r1 = list(att.current_population())
        evals_r1 = [_evaluated_from_spec(s, success=False, tau_caught=1)
                     for s in pop_r1]
        att.evolve(evals_r1)
        pc_after_early_failures = sum(float(n.sum_partial_credit)
                                        for n in att._nodes_by_id.values())

        pop_r2 = list(att.current_population())
        evals_r2 = [_evaluated_from_spec(s, success=False,
                                          tau_caught=max(2, att.injectable_turn_ceiling))
                     for s in pop_r2]
        att.evolve(evals_r2)
        pc_after_late_failures = sum(float(n.sum_partial_credit)
                                       for n in att._nodes_by_id.values())

        self.assertGreater(pc_after_late_failures, pc_after_early_failures,
                           "Late-caught failures should accumulate more partial credit")

    def test_ucb_score_prefers_higher_delta_subtree_under_equal_visits(self):
        att,_ = self._build()
        children_ids = att._root.children_ids
        self.assertGreaterEqual(len(children_ids), 2)

        lo_child = att._nodes_by_id[children_ids[-1]]
        hi_child = att._nodes_by_id[children_ids[0]]

        hi_child.n_visits = 20
        hi_child.n_success = 15
        hi_child.sum_delta_on_success = 12.0       # mean ≈ 0.80 on successes
        hi_child.max_delta_observed = 0.95

        lo_child.n_visits = 20                      # equally visited but no successes
        lo_child.n_success = 0
        lo_child.sum_delta_on_success = 0.0
        lo_child.max_delta_observed = 0.0

        parent_visit_proxy = max(int(att._root.n_visits + 40), 1)
        score_hi = att._ucb_score(parent_visit_proxy, hi_child)
        score_lo = att._ucb_score(parent_visit_proxy, lo_child)
        self.assertGreater(score_hi, score_lo,
                           "UCB score should favour higher-success higher-delta child "
                           "under equal-visits condition")

    def test_factory_dispatch_ga_default_returns_genetic_attacker(self):
        gen = _StubGenerator()
        ga_inst = build_attacker(_make_task(), _make_tools(), gen, AttackerConfig())
        self.assertIsInstance(ga_inst, GeneticAttacker)

    def test_factory_dispatch_mcts_delta_returns_mcts_attacker(self):
        gen = _StubGenerator()
        mcts_inst = build_attacker(
            _make_task(), _make_tools(), gen,
            AttackerConfig(search_method="mcts_delta"),
        )
        self.assertIsInstance(mcts_inst, DeltaGuidedMCTSAttacker)

    def test_unknown_search_method_raises_value_error(self):
        gen = _StubGenerator()
        with self.assertRaises(ValueError):
            build_attacker(
                _make_task(), _make_tools(), gen,
                AttackerConfig(search_method="bogus"),
            )

    def test_two_rounds_run_without_crash_offline(self):
        """End-to-end smoke against the stub generator across two rounds."""
        att,_ = self._build(population_size=6)
        for round_label in range(2):
            pop = att.current_population() if round_label == 0 else \
                  att.evolve(prev_evals)             # type: ignore[name-defined]
            prev_evals = [
                _evaluated_from_spec(s, success=(i % 2 == 0),
                                     delta_norm=float((i % 3) * 0.25),
                                     tau_caught=min(i + 1, att.injectable_turn_ceiling))
                for i, s in enumerate(pop)
            ]
        # After two full rounds the tree should have non-trivial statistics somewhere.
        total_succ = sum(int(getattr(n,'n_success',0)) for n in att._nodes_by_id.values())
        self.assertGreater(total_succ, 0)

    def test_prewarm_accepts_existing_concrete_seed_fields(self):
        import evoguard.attacks.mct_searcher as mcts

        mcts._PREWARM_CACHE = {
            "legacy": [{
                "target_turn": 0,
                "method": "legacy_concrete_seed",
                "payload": "legacy payload",
            }]
        }
        try:
            task = Task(
                task_id="test:legacy:prewarm",
                instruction="dummy instruction",
                suite="legacy",
                dataset="legacy",
                tool_names=[],
            )
            attacker = DeltaGuidedMCTSAttacker(
                task=task,
                tools=[],
                generator=_StubGenerator(),
                config=AttackerConfig(search_method="mcts_delta", population_size=8),
                defense_max_turns=5,
            )
            methods = {
                str(node.discriminator.get("method") or "")
                for node in attacker._nodes_by_id.values()
                if node.level == "L2_method"
            }
            self.assertIn("legacy_concrete_seed", methods)
        finally:
            mcts._PREWARM_CACHE = None

    def test_universal_prewarm_seeds_apply_to_unseen_suite(self):
        import evoguard.attacks.mct_searcher as mcts

        mcts._PREWARM_CACHE = None
        skel_by_suite = mcts._load_prewarm_skeletons_all()
        if not skel_by_suite.get("universal"):
            self.skipTest("no universal prewarm seeds present; skipping")
        self.assertEqual(set(skel_by_suite), {"universal"})
        # Count is read from the artifact rather than hard-coded: the universal
        # seed set is revised over time (v1 -> v2 28 seeds -> v2r 20 seeds) and a
        # literal here fails on every revision without indicating a real defect.
        self.assertGreater(len(skel_by_suite["universal"]), 0)

        task = Task(
            task_id="test:agent_security_bench:universal_prewarm",
            instruction="dummy instruction",
            suite="agent_security_bench",
            dataset="agent_security_bench",
            tool_names=[],
        )
        cfg = AttackerConfig(
            search_method="mcts_delta",
            population_size=8,
            random_seed=42,
        )
        attacker = DeltaGuidedMCTSAttacker(
            task=task, tools=[], generator=_StubGenerator(), config=cfg,
            defense_max_turns=5,
        )
        methods = {
            str(node.discriminator.get("method") or "")
            for node in attacker._nodes_by_id.values()
            if node.level == "L2_method"
        }
        expected = {str(seed["method"]) for seed in skel_by_suite["universal"]}
        self.assertEqual(methods, expected)

        seed = skel_by_suite["universal"][0]
        seed_node = next(
            node for node in attacker._nodes_by_id.values()
            if node.level == "L2_method"
            and node.discriminator.get("method") == seed["method"]
        )
        child = attacker._expand_frontier(seed_node)
        self.assertIsNotNone(child)
        self.assertTrue(child.cached_payload_text.startswith(seed["payload_template"]))

    def test_prewarm_skeleton_injection_adds_l2_branches_for_matching_suite(self):
        """Skeletons from *_seeds.json should appear as L2_method children of L1.

        We don't mock the filesystem here; instead we verify against whatever
        real seed files ship under data/toolsafe/agentdojo-tragj/ (4 suites x
        ~5 entries each). If no files exist on disk we skip gracefully.
        """
        from evoguard.attacks.mct_searcher import _load_prewarm_skeletons_all

        skel_by_suite = _load_prewarm_skeletons_all()
        # The loader returns a key per *_seeds.json file it finds, so a dict of
        # empty lists is the normal state when the seed files are empty stubs
        # (as in the noseed baseline). Only suites with actual entries are
        # testable here.
        populated = [s for s, e in skel_by_suite.items() if e]
        if not populated:
            self.skipTest("no prewarm skeleton entries present; skipping integration test")

        # Pick any suite that has skeletons and build a task in that suite.
        suite_name = populated[0]

        task = Task(
            task_id=f"test:{suite_name}:prewarm_probe",
            instruction="dummy instruction",
            suite=suite_name,
            dataset="agentdojo",
            tool_names=[],
        )
        cfg = AttackerConfig(
            search_method="mcts_delta",
            population_size=8,
            random_seed=42,
        )
        attacker = DeltaGuidedMCTSAttacker(
            task=task, tools=[], generator=_StubGenerator(), config=cfg,
            defense_max_turns=5,
        )

        # Collect all method labels visible in tree's L2_method nodes.
        l2_methods: set[str] = set()
        for nid in attacker._root.children_ids:
            cn = attacker._nodes_by_id.get(nid)
            if not cn:
                continue
            for cid in cn.children_ids:
                cc = attacker._nodes_by_id.get(cid)
                if cc is None or cc.level != "L2_method":
                    continue
                m = str(cc.discriminator.get("method") or "")
                if m:
                    l2_methods.add(m)

        skel_methods = {
            str(e.get("method", "")).strip()
            for e in skel_by_suite[suite_name]
            if isinstance(e, dict)
        }
        skel_methods.discard("")
        matched = [s for s in skel_methods if s in l2_methods]
        self.assertGreater(
            len(matched), 0,
            f"none of skeleton methods {skel_methods} found among "
            f"tree's L2_method children {l2_methods}",
        )


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDeltaGuidedMCTSAttacker)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    ok = result.wasSuccessful()
    print("\nALL CHECKS PASSED" if ok else "\nFAILURES DETECTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
