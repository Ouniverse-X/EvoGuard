"""Unit tests for :class:`GeneticAttacker._inject_turn_ceiling`.

These pin down the policy described in ``docs/genetic.md`` / discussion thread:

* The controller's hard loop bound (:class:`DefenseConfig.max_turns`,
  passed in here as ``defense_max_turns``) is the ONLY structural cap applied
  to ``AttackSpec.target_turn``.
* The previous secondary bound based on tool count has been removed because LLM
  agents are free to call tools repeatedly -- ``len(tools)`` was never more than
  a mock-only heuristic that silently shrank the reachable search space under
  real backends.
* A defensive fallback to ``len(tools)`` survives ONLY when no
  ``defense_max_turns`` information is available at construction time.

Run::

    python -m evoguard.tests.test_attacker_ceiling

The script exits non-zero on any assertion failure so it can be wired into CI.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evoguard.attacks.genetic import GeneticAttacker
from evoguard.config import AttackerConfig
from evoguard.core.types import AttackSpec, Task, ToolSpec


def _build(n_tools: int, *, defense_max_turns):
    """Construct a real GeneticAttacker without invoking any LLM."""
    from evoguard.attacks.base import AttackGenerator

    class _NoOpGenerator(AttackGenerator):
        def seed(self, *a, **k): return []
        def crossover(self, *a, **k): raise AssertionError("crossover must be lazy at init time")
        def mutate(self, *a, **k): raise AssertionError("mutate must be lazy at init time")

    task = Task(task_id="t-test", instruction="dummy", suite="test")
    tools = [ToolSpec(name=f"tool_{i}", description="d") for i in range(n_tools)]
    cfg = AttackerConfig(population_size=2)
    return GeneticAttacker(
        task=task,
        tools=tools,
        generator=_NoOpGenerator(),
        config=cfg,
        rng=__import__("random").Random(0),
        defense_max_turns=defense_max_turns,
    )


def main() -> int:
    failures = []

    # ------------------------------------------------------------------ #
    # Case 1: defense_max_turns larger than len(tools).
    #         Ceiling MUST equal max_turns alone (NOT min()).
    # ------------------------------------------------------------------ #
    ga = _build(n_tools=3, defense_max_turns=10)
    expected1 = 10                  # NOT min(10,3)==3 anymore
    got1 = ga.injectable_turn_ceiling
    if got1 != expected1:
        failures.append(f"[case1] ceiling={got1}, want {expected1} "
                        f"(n_tools=3,max_turns=10)")

    # ------------------------------------------------------------------ #
    # Case 2: defense_max_turns smaller than len(tools). Still just max_turns.
    # ------------------------------------------------------------------ #
    ga = _build(n_tools=8, defense_max_turns=5)
    expected2 = 5                   # equals old behaviour coincidentally;
                                    # included to lock the rule going forward
    got2 = ga.injectable_turn_ceiling
    if got2 != expected2:
        failures.append(f"[case2] ceiling={got2}, want {expected2} "
                        f"(n_tools=8,max_turns=5)")

    # ------------------------------------------------------------------ #
    # Case 3: No defense_max_turns supplied -> fallback to len(tools).
    # ------------------------------------------------------------------ #
    ga = _build(n_tools=7, defense_max_turns=None)
    expected3 = 7
    got3 = ga.injectable_turn_ceiling
    if got3 != expected3:
        failures.append(f"[case3] ceiling={got3}, want {expected3} "
                        f"(fallback path broken; n_tools=7,no max_turns)")

    # ------------------------------------------------------------------ #
    # Case 4: sanitize_spec respects the wider ceiling.
    # Old impl would clamp target_turn>=len(tools); new one allows up to
    # max_turns-1 inclusive.
    # ------------------------------------------------------------------ #
    ga = _build(n_tools=2, defense_max_turns=6)
    spec_high = AttackSpec(attack_id="a-hi", task_id="t-test",
                           target_turn=5, injection_channel="tool_result",
                           method="m", payload="p")
    out_hi = ga.sanitize_spec(spec_high)
    if out_hi.target_turn != 5:
        failures.append(f"[case4 high] target_turn={out_hi.target_turn} after clamp,"
                        f"want 5 preserved")

    spec_over = AttackSpec(attack_id="a-over", task_id="t-test",
                           target_turn=99, injection_channel="tool_result",
                           method="m", payload="p")
    out_over = ga.sanitize_spec(spec_over)
    if out_over.target_turn != 5:           # max_turns-1 == 6-1 == 5
        failures.append(f"[case4 over] target_turn={out_over.target_turn}"
                        f" after clamp,want 5 (=max_turns-1)")
    spec_neg = AttackSpec(attack_id="a-neg", task_id="t-test",
                          target_turn=-3, injection_channel="tool_result",
                          method="m", payload="p")
    out_neg = ga.sanitize_spec(spec_neg)
    if out_neg.target_turn != 0:
        failures.append(f"[case4 neg] target_turn={out_neg.target_turn}"
                        f" after clamp,want 0")

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    print("=== evoguard.tests.test_attacker_ceiling ===")
    if failures:
        print(f"{len(failures)} FAIL(S):")
        for s in failures:
            print(" x ", s)
        return 1
    print("all assertions ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
