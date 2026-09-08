"""Locks the fix for the colocate second-engine crash.

Both r2 of run 20260907_220147 and r3 of 20260908_000734 died with
``it->second->use_count > 0 INTERNAL ASSERT FAILED at
"CUDACachingAllocator.cpp":2731`` from ``capture_begin`` while the SECOND colocate
vLLM engine of the process was capturing CUDA graphs. The second death had 44.45
GiB free, so this is not an out-of-memory failure -- it is mempool-id reuse.

``test_reused_pool_after_use_count_zero_asserts`` reproduces the crash in torch
alone, and ``test_fresh_pool_handle_is_usable`` shows the escape. Both need a GPU
and are skipped without one. ``test_reset_retires_the_cached_handle`` needs no GPU
and is the one that actually guards ``_reset_vllm_global_graph_pool``.

Run: python -m evoguard.tests.test_colocate_graph_pool
"""
from __future__ import annotations

import sys
import types
import unittest

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:                                                       # noqa: BLE001
    torch = None                                                        # type: ignore[assignment]
    _HAS_CUDA = False

from evoguard.training.native_grpo_runner import _reset_vllm_global_graph_pool


def _capture_into(pool, pinned):
    """Capture a graph into ``pool`` and keep one of its tensors alive.

    Retaining the output is what makes the private pool unfreeable, which is the
    precondition for the assert: ``releasePool`` drops use_count to 0 but leaves
    the entry in ``graph_pools`` until all of its blocks are gone.
    """
    graph = torch.cuda.CUDAGraph()
    static_in = torch.zeros(1024, 1024, device="cuda")
    with torch.cuda.graph(graph, pool=pool):
        out = static_in * 2.0 + 1.0
    pinned.append(out)
    return graph


@unittest.skipUnless(_HAS_CUDA, "needs a CUDA device")
class TestMempoolIdReuse(unittest.TestCase):
    def test_reused_pool_after_use_count_zero_asserts(self):
        pinned: list = []
        pool = torch.cuda.graph_pool_handle()
        del_me = _capture_into(pool, pinned)
        del del_me                      # releasePool -> use_count 0, entry retained
        torch.cuda.synchronize()
        torch.cuda.empty_cache()        # cannot erase: `pinned` still holds a block

        with self.assertRaises(RuntimeError) as ctx:
            _capture_into(pool, pinned)
        self.assertIn("use_count > 0", str(ctx.exception))

    def test_fresh_pool_handle_is_usable(self):
        """The same sequence survives if the second capture gets a NEW handle.

        This is exactly what `_reset_vllm_global_graph_pool` arranges for the next
        engine, by clearing the handle vLLM memoises on its platform class.
        """
        pinned: list = []
        first = torch.cuda.graph_pool_handle()
        del_me = _capture_into(first, pinned)
        del del_me
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        second = torch.cuda.graph_pool_handle()
        self.assertNotEqual(first, second)
        graph = _capture_into(second, pinned)
        self.assertIsNotNone(graph)


class TestResetGlobalGraphPool(unittest.TestCase):
    """No GPU needed: `vllm.platforms` is stubbed to the shape vLLM 0.19.1 has."""

    def setUp(self):
        self._saved = sys.modules.get("vllm.platforms")

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("vllm.platforms", None)
        else:
            sys.modules["vllm.platforms"] = self._saved

    @staticmethod
    def _install_stub():
        mod = types.ModuleType("vllm.platforms")

        class Platform:
            _global_graph_pool = None

        class CudaPlatform(Platform):
            pass

        # get_global_graph_pool() does `cls = self.__class__; cls._global_graph_pool
        # = ...`, so the handle lands on the SUBCLASS and shadows the base.
        CudaPlatform._global_graph_pool = (0, 1)
        mod.Platform = Platform
        mod._current_platform = CudaPlatform()
        sys.modules["vllm.platforms"] = mod
        return mod, Platform, CudaPlatform

    def test_reset_retires_the_cached_handle(self):
        _mod, base, subclass = self._install_stub()
        _reset_vllm_global_graph_pool()
        self.assertIsNone(subclass._global_graph_pool)
        self.assertIsNone(base._global_graph_pool)

    def test_reset_clears_a_handle_cached_on_the_base_class(self):
        mod, base, subclass = self._install_stub()
        subclass._global_graph_pool = None
        base._global_graph_pool = (0, 7)
        _reset_vllm_global_graph_pool()
        self.assertIsNone(base._global_graph_pool)

    def test_no_vllm_import_is_a_noop(self):
        sys.modules.pop("vllm.platforms", None)
        _reset_vllm_global_graph_pool()          # must not raise

    def test_missing_current_platform_still_clears_base(self):
        mod, base, _subclass = self._install_stub()
        mod._current_platform = None
        base._global_graph_pool = (0, 3)
        _reset_vllm_global_graph_pool()
        self.assertIsNone(base._global_graph_pool)


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
