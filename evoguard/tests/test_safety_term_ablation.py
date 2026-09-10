"""Offline invariants for the ``grpo_disable_safety_term`` ablation.

Run: ``python -m evoguard.tests.test_safety_term_ablation`` (optionally ``-k name``).
No GPU, no network, no judge endpoint.

What this locks, and why each case exists:

1. ``disable_safety=False`` must be BIT-FOR-BIT the historical path. The ablation
   is a research arm, not a behaviour change, so the default branch is compared
   field-by-field against a run with the kwarg omitted entirely.
2. ``disable_safety=True`` must produce ``r_safety == 0.0`` EXACTLY and
   ``total == r_progress - p_drift`` on BOTH arms. The clean arm matters because
   the ablation branch sits before ``meta.is_clean``'s early return; if it were
   placed after, clean rows would keep ``R_SAFETY_CLEAN_SERVED = +2.00`` and the
   two arms would no longer share a reward scale.
3. The judge must never be consulted. A judge callable that raises on contact
   proves it: the historical path swallows judge exceptions into the ``unclear``
   fallback, so a "no exception" assertion alone would not distinguish
   "never called" from "called and failed" -- hence the call COUNTER.
4. ``safety_source``/``safety_label`` must read ``"disabled"``, not
   ``"fallback"``/``"unclear"``. Those are the values a judge OUTAGE produces,
   and the two situations have identical reward shapes, so the round-level
   ``safety_source_tally`` is the only thing that can tell an intended ablation
   from an accident.
5. Under GDPO the zeroed term must contribute EXACTLY nothing. Verified against
   ``_gdpo_advantages`` directly: a 3-term batch whose first slot is all-zero
   must give the same advantages as the 2-term batch with that slot deleted.
   This is what makes "GDPO stays on" a coherent claim for this arm.
"""

from __future__ import annotations

import sys

from evoguard.training.grpo_reward import (
    PromptMeta,
    R_SAFETY_CLEAN_SERVED,
    compute_evoguard_reward,
)
from evoguard.training.native_grpo_runner import _gdpo_advantages


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
_COMPLETION_TOOL = '{"thought": "reading the inbox", "tool": {"name": "read_inbox", "args": {"n": 5}}}'
_COMPLETION_BAIT = '{"thought": "forwarding as instructed", "tool": {"name": "send_money", "args": {"amount": 9999}}}'


def _attacked_meta() -> PromptMeta:
    """An attacked prompt whose payload is already visible.

    ``payload_visible_in_prefix=True`` plus an EMPTY benign timeline is the one
    configuration that reaches the judge: the structural short-circuits fire on
    an invisible payload, on a signature present in the clean twin, and on a
    no-action step, and none of those apply here.
    """
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


class _CountingJudge:
    """Judge stub that RAISES, and counts how often it was reached."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _completion_text, _meta):
        self.calls += 1
        raise AssertionError("judge_call must not be reached under the ablation")


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
        (_attacked_meta(), _COMPLETION_BAIT),
        (_clean_meta(), _COMPLETION_TOOL),
    ):
        omitted = compute_evoguard_reward(completion_text=comp, meta=meta)
        explicit = compute_evoguard_reward(
            completion_text=comp, meta=meta, disable_safety=False
        )
        for field in ("r_safety", "r_progress", "p_drift", "total"):
            _assert(
                _close(getattr(omitted, field), getattr(explicit, field)),
                f"disable_safety=False changed {field}: "
                f"{getattr(omitted, field)!r} != {getattr(explicit, field)!r}",
            )
        for field in ("safety_label", "safety_source"):
            _assert(
                getattr(omitted, field) == getattr(explicit, field),
                f"disable_safety=False changed {field}",
            )
    print("[ok] default path unchanged (attacked + clean)")


# --------------------------------------------------------------------------- #
# 2. the term is gone, on BOTH arms                                            #
# --------------------------------------------------------------------------- #
def test_safety_term_is_exactly_zero_on_both_arms():
    for name, meta, comp in (
        ("attacked", _attacked_meta(), _COMPLETION_BAIT),
        ("clean", _clean_meta(), _COMPLETION_TOOL),
    ):
        bd = compute_evoguard_reward(
            completion_text=comp, meta=meta, disable_safety=True
        )
        _assert(
            bd.r_safety == 0.0,
            f"{name}: r_safety={bd.r_safety!r}, expected exactly 0.0",
        )
        _assert(
            _close(bd.total, bd.r_progress - bd.p_drift),
            f"{name}: total={bd.total!r} != r_progress - p_drift "
            f"({bd.r_progress!r} - {bd.p_drift!r})",
        )
    # The clean arm is the one that would silently keep its +2.00 if the branch
    # were placed after the is_clean early return -- assert the delta explicitly.
    clean_on = compute_evoguard_reward(
        completion_text=_COMPLETION_TOOL, meta=_clean_meta()
    )
    _assert(
        _close(clean_on.r_safety, R_SAFETY_CLEAN_SERVED),
        "control broke: the clean arm should score R_SAFETY_CLEAN_SERVED by default",
    )
    print("[ok] r_safety == 0.0 and total == r_progress - p_drift on both arms")


# --------------------------------------------------------------------------- #
# 3. the judge is never reached                                                #
# --------------------------------------------------------------------------- #
def test_judge_is_never_called_under_ablation():
    judge = _CountingJudge()
    bd = compute_evoguard_reward(
        completion_text=_COMPLETION_BAIT,
        meta=_attacked_meta(),
        judge_call=judge,
        disable_safety=True,
    )
    _assert(judge.calls == 0, f"judge was called {judge.calls}x under the ablation")
    _assert(bd.r_safety == 0.0, "r_safety must stay 0.0")

    # Control: on the DEFAULT path this same meta/completion does reach the judge.
    # (It raises, and the historical path absorbs that into the unclear fallback.)
    control = _CountingJudge()
    compute_evoguard_reward(
        completion_text=_COMPLETION_BAIT,
        meta=_attacked_meta(),
        judge_call=control,
    )
    _assert(
        control.calls == 1,
        f"control expected exactly 1 judge call, got {control.calls} -- the "
        f"fixture no longer reaches the judge, so case 3 proves nothing",
    )
    print("[ok] judge never reached under the ablation (control reaches it once)")


# --------------------------------------------------------------------------- #
# 4. an ablation is distinguishable from an outage                              #
# --------------------------------------------------------------------------- #
def test_labels_distinguish_ablation_from_outage():
    ablated = compute_evoguard_reward(
        completion_text=_COMPLETION_BAIT, meta=_attacked_meta(), disable_safety=True
    )
    _assert(ablated.safety_source == "disabled",
            f"safety_source={ablated.safety_source!r}, expected 'disabled'")
    _assert(ablated.safety_label == "disabled",
            f"safety_label={ablated.safety_label!r}, expected 'disabled'")

    outage = compute_evoguard_reward(
        completion_text=_COMPLETION_BAIT, meta=_attacked_meta(), judge_call=None
    )
    _assert(
        (outage.safety_source, outage.safety_label) == ("fallback", "unclear"),
        f"outage control changed shape: {outage.safety_source!r}/"
        f"{outage.safety_label!r}",
    )
    _assert(
        (ablated.safety_source, ablated.safety_label)
        != (outage.safety_source, outage.safety_label),
        "ablation and outage are indistinguishable in the tally",
    )
    print("[ok] 'disabled:disabled' is distinct from the outage's 'fallback:unclear'")


# --------------------------------------------------------------------------- #
# 5. GDPO degrades cleanly to 2 terms                                          #
# --------------------------------------------------------------------------- #
def test_gdpo_ignores_the_zeroed_term():
    g = 4
    # Two prompt groups, deliberately including a group where progress is
    # unanimous but drift is not, so the surviving terms are not degenerate.
    progress = [1.20, -0.15, -2.50, -0.15, -0.15, -0.15, -0.15, -0.15]
    drift = [0.0, -0.25, 0.0, -0.50, 0.0, -0.25, -0.50, 0.0]

    three = [(0.0, progress[i], drift[i]) for i in range(len(progress))]
    two = [(progress[i], drift[i]) for i in range(len(progress))]

    adv3 = _gdpo_advantages(three, num_generations=g)
    adv2 = _gdpo_advantages(two, num_generations=g)
    _assert(adv3 and adv2, "GDPO returned no advantages for a well-formed batch")
    _assert(len(adv3) == len(adv2) == len(progress), "advantage length mismatch")
    for i, (a3, a2) in enumerate(zip(adv3, adv2)):
        _assert(
            _close(a3, a2, tol=1e-9),
            f"slot {i}: 3-term-with-zero {a3!r} != 2-term {a2!r}; the zeroed "
            f"safety term is NOT free under GDPO",
        )

    # And a NONZERO constant would also be free (it is unanimous in-group), which
    # is why the runner's unknown-row default moves its -0.5 out of the safety
    # slot: a per-BATCH constant is only free if it is also per-GROUP constant.
    nonzero_const = [(7.0, progress[i], drift[i]) for i in range(len(progress))]
    advc = _gdpo_advantages(nonzero_const, num_generations=g)
    for i, (ac, a2) in enumerate(zip(advc, adv2)):
        _assert(_close(ac, a2, tol=1e-9), f"slot {i}: constant term was not free")
    print("[ok] GDPO treats the zeroed term as unanimous -> exactly 2-term behaviour")


# --------------------------------------------------------------------------- #
# 6. the config field exists and defaults off                                  #
# --------------------------------------------------------------------------- #
def test_config_field_defaults_off():
    from evoguard.config import TrainingConfig

    cfg = TrainingConfig()
    _assert(
        hasattr(cfg, "grpo_disable_safety_term"),
        "TrainingConfig has no grpo_disable_safety_term field",
    )
    _assert(
        cfg.grpo_disable_safety_term is False,
        f"grpo_disable_safety_term defaults to "
        f"{cfg.grpo_disable_safety_term!r}, must be False",
    )
    print("[ok] TrainingConfig.grpo_disable_safety_term defaults to False")


_TESTS = [
    test_default_is_bit_for_bit_the_historical_path,
    test_safety_term_is_exactly_zero_on_both_arms,
    test_judge_is_never_called_under_ablation,
    test_labels_distinguish_ablation_from_outage,
    test_gdpo_ignores_the_zeroed_term,
    test_config_field_defaults_off,
]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = ""
    if "-k" in argv:
        selector = argv[argv.index("-k") + 1]
    chosen = [t for t in _TESTS if not selector or selector in t.__name__]
    if not chosen:
        print(f"no test matches -k {selector!r}")
        return 1
    failures = 0
    for t in chosen:
        try:
            t()
        except Exception as exc:                                     # noqa: BLE001
            failures += 1
            print(f"[FAIL] {t.__name__}: {exc}")
    print(f"\n{len(chosen) - failures}/{len(chosen)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
