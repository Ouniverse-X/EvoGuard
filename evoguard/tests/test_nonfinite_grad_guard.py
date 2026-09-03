"""Guard against NaN gradients reaching the LoRA weights.

Locks ``native_grpo_runner._install_nonfinite_grad_guard``. The failure it
prevents killed r2 of run 20260903_002823: ``'grad_norm': nan`` at inner step 17
with a finite loss, ``Trainer`` clipped every gradient by that NaN norm, the
optimizer wrote NaN into the adapter, and the next rollout aborted inside
``generate()`` on ``probability tensor contains either inf, nan or element < 0``.

CPU-only: the guard reads ``p.grad`` through ``torch._foreach_norm``, which needs
no device.
"""

from __future__ import annotations

import unittest

import torch

from evoguard.training.native_grpo_runner import _install_nonfinite_grad_guard


class _FakeOptimizer:
    """Minimal ``torch.optim.Optimizer`` surface the guard touches."""

    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]
        self.n_steps_run = 0

    def step(self, *args, **kwargs):          # noqa: ANN002,ANN003,D401
        self.n_steps_run += 1
        return "stepped"


class _FakeTrainer:
    def __init__(self, params):
        self._opt = _FakeOptimizer(params)

    def create_optimizer(self):
        return self._opt


class TestNonFiniteGradGuard(unittest.TestCase):
    def _harness(self):
        p = torch.nn.Parameter(torch.zeros(4))
        trainer = _FakeTrainer([p])
        diag: dict = {}
        _install_nonfinite_grad_guard(trainer, diag)
        opt = trainer.create_optimizer()
        return p, opt, diag

    def test_finite_gradients_step_normally(self):
        p, opt, diag = self._harness()
        p.grad = torch.ones(4)
        self.assertEqual(opt.step(), "stepped")
        self.assertEqual(opt.n_steps_run, 1)
        self.assertEqual(diag["nonfinite_grad_skips"], 0)

    def test_nan_gradient_skips_the_step(self):
        p, opt, diag = self._harness()
        p.grad = torch.tensor([1.0, float("nan"), 0.0, 0.0])
        self.assertIsNone(opt.step())
        self.assertEqual(opt.n_steps_run, 0)
        self.assertEqual(diag["nonfinite_grad_skips"], 1)

    def test_inf_gradient_skips_the_step(self):
        p, opt, diag = self._harness()
        p.grad = torch.tensor([float("inf"), 0.0, 0.0, 0.0])
        self.assertIsNone(opt.step())
        self.assertEqual(opt.n_steps_run, 0)
        self.assertEqual(diag["nonfinite_grad_skips"], 1)

    def test_guard_does_not_zero_the_gradients(self):
        # Trainer calls model.zero_grad() itself right after optimizer.step()
        # (transformers trainer.py:2752). Zeroing here as well would be harmless
        # today but hides which component owns the reset.
        p, opt, _diag = self._harness()
        p.grad = torch.tensor([float("nan"), 0.0, 0.0, 0.0])
        opt.step()
        self.assertIsNotNone(p.grad)
        self.assertTrue(torch.isnan(p.grad[0]))

    def test_recovers_after_a_skip(self):
        p, opt, diag = self._harness()
        p.grad = torch.full((4,), float("nan"))
        opt.step()
        p.grad = torch.ones(4)
        self.assertEqual(opt.step(), "stepped")
        self.assertEqual(opt.n_steps_run, 1)
        self.assertEqual(diag["nonfinite_grad_skips"], 1)

    def test_no_gradients_yet_steps(self):
        # First accumulation boundary of a resumed run can legitimately have
        # params with grad=None; that is not a NaN and must not be skipped.
        _p, opt, diag = self._harness()
        self.assertEqual(opt.step(), "stepped")
        self.assertEqual(diag["nonfinite_grad_skips"], 0)

    def test_install_is_idempotent_on_the_same_optimizer(self):
        p, opt, diag = self._harness()
        first = opt.step
        opt2 = None
        # A second create_optimizer() call must not wrap the guard twice.
        opt2 = _FakeTrainer([p]).create_optimizer()
        self.assertIsNot(first, opt2.step)
        p.grad = torch.full((4,), float("nan"))
        opt.step()
        opt.step()
        self.assertEqual(diag["nonfinite_grad_skips"], 2)
        self.assertEqual(opt.n_steps_run, 0)


def main() -> None:
    unittest.main(module=__name__, argv=["test_nonfinite_grad_guard"], exit=False)


if __name__ == "__main__":
    unittest.main()
