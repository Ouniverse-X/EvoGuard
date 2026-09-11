"""Offline invariants for the ``grpo_disable_progress_term`` ablation.

Run: ``python -m evoguard.tests.test_progress_term_ablation`` (optionally ``-k name``).
No GPU, no network, no judge endpoint.

What this locks, and why each case exists:

1. ``disable_progress=False`` must be BIT-FOR-BIT the historical path. Same
   contract as the safety ablation: a research arm, not a behaviour change, so
   the default branch is compared field-by-field against omitting the kwarg.
2. ``disable_progress=True`` must give ``r_progress == 0.0`` EXACTLY and
   ``total == r_safety - p_drift`` on BOTH arms. The clean arm matters more here
   than in the safety ablation: ``r_progress`` was the ONLY signal clean prompts
   carried, so this is where the arm's gradient actually goes missing.
3. The progress judge must never be consulted -- proved with a stub that RAISES
   and counts, because the historical path swallows progress-judge exceptions
   into the NEUTRAL fallback and would look identical otherwise.
4. ``progress_source`` must read ``"disabled"``, never ``"fallback"``. A progress
   OUTAGE pins the judged rows at ``R_PROGRESS_NEUTRAL`` while structurally
   settled rows keep real values -- a different and noisier signal than an absent
   term -- so the round-level ``progress_source_tally`` is the only observable
   that separates the two.
5. Under GDPO the zeroed term must contribute EXACTLY nothing, verified against
   ``_gdpo_advantages``: a 3-term batch whose MIDDLE slot is all-zero must give
   the same advantages as the 2-term batch with that slot deleted.
6. The DOCUMENTED PATHOLOGY must be real: with ``P_PROGRESS_WASTE`` gone, a
   no-action completion scores the structural safety ``held`` (+2.00) at
   ``p_drift`` 0.0, which is the arm's maximum -- so "do nothing" is optimal and
   ASR is uninterpretable on this arm. This is asserted rather than merely
   commented so that anyone who later "fixes" it has to confront the fact that
   the fix changes what the ablation measures.
"""

from __future__ import annotations

import sys

from evoguard.training.grpo_reward import (
    PromptMeta,
    P_PROGRESS_WASTE,
    R_PROGRESS_NEUTRAL,
    R_SAFETY_HELD,
    compute_evoguard_reward,
)
from evoguard.training.native_grpo_runner import _gdpo_advantages


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
# Reaches the progress judge: it acts, the signature is absent from the prefix,
# and it is NOT the clean twin's next action -- so _progress_from_structure
# returns None on all three of its short circuits.
_COMPLETION_OFFPLAN = (
    '{"thought": "forwarding as instructed", '
    '"tool": {"name": "send_money", "args": {"amount": 9999}}}'
)
# Settled structurally as WASTE: no tool and no final answer.
_COMPLETION_NOACTION = '{"thought": "let me think about this some more"}'


def _attacked_meta() -> PromptMeta:
    return PromptMeta(
        task_id="t/bank/1",
        task_instruction="check my balance",
        injection_point=1,
        turning_point=2,
        delta_normalized=0.0,
        target_tool_name="",
        is_clean=False,
        payload_visible_in_prefix=True,
        prefix_action_signatures=(),
        clean_twin_action_tool_name="get_balance",
        clean_twin_action_args={},
        benign_reference_timeline=(),
    )


def _clean_meta() -> PromptMeta:
    return PromptMeta(
        task_id="t/bank/1",
        task_instruction="check my balance",
        injection_point=-1,
        turning_point=-1,
        delta_normalized=0.0,
        target_tool_name="",
        is_clean=True,
        prefix_action_signatures=(),
        clean_twin_action_tool_name="get_balance",
        clean_twin_action_args={},
    )


class _CountingProgressJudge:
    """Progress-judge stub that RAISES, and counts how often it was reached."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _completion_text, _meta):
        self.calls += 1
        raise AssertionError("progress_call must not be reached under the ablation")


def _held_judge(_completion_text, _meta):
    """Safety judge that always answers ``held`` -- keeps r_safety deterministic."""
    return "held"


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- #
# 1. default path is unchanged                                                 #
# --------------------------------------------------------------------------- #
def test_default_is_bit_for_bit_the_historical_path():
    for meta, comp in (
        (_attacked_meta(), _COMPLETION_OFFPLAN),
        (_attacked_meta(), _COMPLETION_NOACTION),
        (_clean_meta(), _COMPLETION_OFFPLAN),
    ):
        omitted = compute_evoguard_reward(
            completion_text=comp, meta=meta, judge_call=_held_judge
        )
        explicit = compute_evoguard_reward(
            completion_text=comp, meta=meta, judge_call=_held_judge,
            disable_progress=False,
        )
        for field in ("r_safety", "r_progress", "p_drift", "total"):
            _assert(
                _close(getattr(omitted, field), getattr(explicit, field)),
                f"disable_progress=False changed {field}: "
                f"{getattr(omitted, field)!r} != {getattr(explicit, field)!r}",
            )
        for field in ("safety_label", "safety_source", "progress_source",
                      "used_progress_fallback"):
            _assert(
                getattr(omitted, field) == getattr(explicit, field),
                f"disable_progress=False changed {field}",
            )
    print("[ok] default path unchanged (off-plan + no-action + clean)")


# --------------------------------------------------------------------------- #
# 2. the term is gone, on BOTH arms                                            #
# --------------------------------------------------------------------------- #
def test_progress_term_is_exactly_zero_on_both_arms():
    for name, meta in (("attacked", _attacked_meta()), ("clean", _clean_meta())):
        bd = compute_evoguard_reward(
            completion_text=_COMPLETION_OFFPLAN, meta=meta,
            judge_call=_held_judge, disable_progress=True,
        )
        _assert(bd.r_progress == 0.0,
                f"[{name}] r_progress is {bd.r_progress!r}, expected exactly 0.0")
        _assert(_close(bd.total, bd.r_safety - bd.p_drift),
                f"[{name}] total {bd.total} != r_safety - p_drift "
                f"({bd.r_safety} - {bd.p_drift})")

    # Control: on the DEFAULT path the same clean row carries a nonzero
    # r_progress. Without this the test above would also pass if r_progress
    # happened to be 0.0 for unrelated reasons.
    ctrl = compute_evoguard_reward(
        completion_text=_COMPLETION_OFFPLAN, meta=_clean_meta(),
        judge_call=_held_judge,
    )
    _assert(ctrl.r_progress != 0.0,
            "control: clean arm default r_progress should be nonzero, "
            f"got {ctrl.r_progress!r}")
    print("[ok] r_progress == 0.0 exactly on both arms; total == r_safety - p_drift")


# --------------------------------------------------------------------------- #
# 3. the progress judge is never reached                                       #
# --------------------------------------------------------------------------- #
def test_progress_judge_is_never_called():
    stub = _CountingProgressJudge()
    bd = compute_evoguard_reward(
        completion_text=_COMPLETION_OFFPLAN, meta=_attacked_meta(),
        judge_call=_held_judge, progress_call=stub, disable_progress=True,
    )
    _assert(stub.calls == 0,
            f"progress judge was reached {stub.calls}x under the ablation")
    _assert(bd.r_progress == 0.0, "ablation did not pin r_progress")

    # Control: the SAME fixture reaches the judge exactly once by default. This
    # is what makes the assertion above meaningful -- a fixture that structurally
    # short-circuits would give calls == 0 on both paths.
    ctrl = _CountingProgressJudge()
    compute_evoguard_reward(
        completion_text=_COMPLETION_OFFPLAN, meta=_attacked_meta(),
        judge_call=_held_judge, progress_call=ctrl,
    )
    _assert(ctrl.calls == 1,
            f"control: default path called the progress judge {ctrl.calls}x, expected 1")
    print("[ok] progress judge never constructed-or-called (control: 1 call by default)")


# --------------------------------------------------------------------------- #
# 4. ablation is distinguishable from an OUTAGE                                #
# --------------------------------------------------------------------------- #
def test_disabled_is_distinct_from_the_outage_labels():
    ablated = compute_evoguard_reward(
        completion_text=_COMPLETION_OFFPLAN, meta=_attacked_meta(),
        judge_call=_held_judge, progress_call=None, disable_progress=True,
    )
    _assert(ablated.progress_source == "disabled",
            f"progress_source is {ablated.progress_source!r}, expected 'disabled'")

    # An OUTAGE: no progress callable, flag off. Judged rows collapse onto the
    # NEUTRAL constant while structural rows keep real values -- a different
    # signal, and the tally must be able to say so.
    outage = compute_evoguard_reward(
        completion_text=_COMPLETION_OFFPLAN, meta=_attacked_meta(),
        judge_call=_held_judge, progress_call=None,
    )
    _assert(outage.progress_source == "fallback",
            f"outage progress_source is {outage.progress_source!r}, expected 'fallback'")
    _assert(_close(outage.r_progress, R_PROGRESS_NEUTRAL),
            "outage should score the NEUTRAL constant")
    _assert(ablated.progress_source != outage.progress_source,
            "ablation and outage are indistinguishable in progress_source")

    # And a healthy structural row is a third, distinct value.
    structural = compute_evoguard_reward(
        completion_text=_COMPLETION_NOACTION, meta=_attacked_meta(),
        judge_call=_held_judge,
    )
    _assert(structural.progress_source == "structural",
            f"structural progress_source is {structural.progress_source!r}")
    print("[ok] progress_source separates disabled / fallback / structural")


# --------------------------------------------------------------------------- #
# 5. GDPO gets the zeroed term for free                                        #
# --------------------------------------------------------------------------- #
def test_gdpo_ignores_the_zeroed_middle_slot():
    # Three groups of two, safety and drift varying, PROGRESS all-zero.
    three = [
        (2.00, 0.0, -0.00), (-8.00, 0.0, -0.25),
        (2.00, 0.0, -0.50), (-0.50, 0.0, -0.00),
        (-1.00, 0.0, -0.25), (2.00, 0.0, -0.00),
    ]
    two = [(a, c) for (a, _b, c) in three]
    got = _gdpo_advantages(three, num_generations=2)
    want = _gdpo_advantages(two, num_generations=2)
    _assert(len(got) == len(want) == len(three),
            f"length mismatch: {len(got)} / {len(want)} / {len(three)}")
    for i, (g, w) in enumerate(zip(got, want)):
        _assert(_close(g, w, 1e-6),
                f"row {i}: 3-term-with-zero {g!r} != 2-term {w!r}")

    # A nonzero CONSTANT is equally free -- which is exactly why the unknown-row
    # -0.5 fallback must not be allowed into the ablated slot: the constant has to
    # hold inside EVERY group, and a per-batch-only constant would not.
    shifted = [(a, -0.15, c) for (a, _b, c) in three]
    got_shifted = _gdpo_advantages(shifted, num_generations=2)
    for i, (g, w) in enumerate(zip(got_shifted, want)):
        _assert(_close(g, w, 1e-6),
                f"row {i}: constant middle slot changed the advantage: {g!r} != {w!r}")
    print("[ok] GDPO: all-zero (and any constant) middle slot contributes nothing")


# --------------------------------------------------------------------------- #
# 6. the documented pathology is real                                          #
# --------------------------------------------------------------------------- #
def test_do_nothing_becomes_optimal_under_the_ablation():
    """`P_PROGRESS_WASTE` is what makes stalling unprofitable. Without it a
    no-action step is settled as structural safety ``held`` at ``p_drift`` 0.0,
    which ties the best score any step can reach. Asserted so the consequence is
    on the record rather than merely in a comment."""
    meta = _attacked_meta()
    noop = compute_evoguard_reward(
        completion_text=_COMPLETION_NOACTION, meta=meta,
        judge_call=_held_judge, disable_progress=True,
    )
    acting = compute_evoguard_reward(
        completion_text=_COMPLETION_OFFPLAN, meta=meta,
        judge_call=_held_judge, disable_progress=True,
    )
    _assert(_close(noop.r_safety, R_SAFETY_HELD),
            f"no-action r_safety is {noop.r_safety!r}, expected structural held "
            f"{R_SAFETY_HELD}")
    _assert(noop.total >= acting.total,
            f"expected the ablation to make stalling at least as good as acting: "
            f"noop {noop.total} < acting {acting.total}")

    # Control: on the DEFAULT path stalling is strictly worse, by exactly the
    # WASTE penalty. If this ever stops holding, the shipped reward regressed.
    noop_ok = compute_evoguard_reward(
        completion_text=_COMPLETION_NOACTION, meta=meta, judge_call=_held_judge,
    )
    _assert(_close(noop_ok.r_progress, -abs(P_PROGRESS_WASTE)),
            f"control: default no-action r_progress is {noop_ok.r_progress!r}, "
            f"expected -{abs(P_PROGRESS_WASTE)}")
    _assert(noop_ok.total < noop.total,
            "control: the default path should punish stalling harder than the ablation")
    print("[ok] pathology confirmed: stalling ties the max under the ablation, "
          "and is strictly worse by default")


# --------------------------------------------------------------------------- #
# 7. the config default is off                                                 #
# --------------------------------------------------------------------------- #
def test_training_config_default_is_off():
    from evoguard.config import TrainingConfig

    _assert(TrainingConfig().grpo_disable_progress_term is False,
            "grpo_disable_progress_term must default to False -- the shipped "
            "reward is the three-term R = r_safety + r_progress - p_drift")
    print("[ok] TrainingConfig().grpo_disable_progress_term is False")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    if "-k" in sys.argv:
        needle = sys.argv[sys.argv.index("-k") + 1]
        tests = [t for t in tests if needle in t.__name__]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:                                    # noqa: BLE001
            failed += 1
            print(f"[FAIL] {t.__name__}: {exc}")
    print(f"\nRan {len(tests)} tests, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
