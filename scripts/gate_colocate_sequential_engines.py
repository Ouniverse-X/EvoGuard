"""End-to-end gate for the colocate second-engine fix. NEEDS a free GPU + weights.

Builds N vLLM engines sequentially in ONE process through exactly the runner's
release path, which is what production does across GRPO rounds. Before the
`_reset_vllm_global_graph_pool` fix, engine #2 died in CUDA-graph capture with
`use_count > 0 INTERNAL ASSERT` regardless of how much memory was free.

Also prints free memory before each build, because the fix does NOT address the
~19 GiB/round that stays resident -- that budget is what decides how many rounds a
single process can do, and it is worth seeing the two facts side by side.

Usage (GPU must be EMPTY -- vLLM profiles whole-card free memory):
    CUDA_VISIBLE_DEVICES=3 python scripts/gate_colocate_sequential_engines.py \
        --model /root/paddlejob/workspace/yangxiao/models/Meta-Llama-3.1-8B-Instruct \
        --engines 3
Exit 0 = every engine built. Exit 1 = a build failed (message says which).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _free_gib() -> float:
    import torch

    free, _total = torch.cuda.mem_get_info(0)
    return free / 2**30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engines", type=int, default=3)
    ap.add_argument("--mem-util", type=float, default=0.25)
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    from evoguard.training.native_grpo_runner import (
        _enter_colocate_env,
        _release_colocate_engine,
        _teardown_colocate_process_group,
    )

    # The engine is built inside GRPOTrainer.__init__ in production, so the env has
    # to be in place before the constructor; here there is no trainer, so set it up
    # once and leave it for every build.
    _enter_colocate_env()

    from vllm import LLM

    class _Holder:
        """Stands in for the trainer: `_release_colocate_engine` reads `.llm`."""

        def __init__(self, llm):
            self.llm = llm

    trace: list[tuple[int, float, str]] = []
    for i in range(1, args.engines + 1):
        before = _free_gib()
        print(f"[gate] engine {i}: free before build = {before:.2f} GiB", flush=True)
        try:
            llm = LLM(
                model=args.model,
                gpu_memory_utilization=args.mem_util,
                max_model_len=args.max_len,
                distributed_executor_backend="external_launcher",
                enable_sleep_mode=False,
            )
        except Exception as exc:                                        # noqa: BLE001
            print(f"[gate] engine {i} FAILED to build: {type(exc).__name__}: {exc}",
                  flush=True)
            trace.append((i, before, f"FAILED {type(exc).__name__}"))
            _print_trace(trace)
            return 1

        out = llm.generate(["The capital of France is"])
        text = out[0].outputs[0].text.strip().replace("\n", " ")[:40]
        print(f"[gate] engine {i} generated: {text!r}", flush=True)

        _release_colocate_engine(_Holder(llm))
        if os.environ.get("EVOGUARD_GATE_SKIP_POOL_RESET") == "1":
            # CONTROL ARM: undo the fix, restoring vLLM's memoised mempool id so the
            # next engine captures into the retired pool exactly as it did before
            # `_reset_vllm_global_graph_pool` existed. Engine 2 is expected to die
            # here with `use_count > 0`. Gate-only; nothing in production does this.
            import vllm.platforms as _vp

            type(_vp._current_platform)._global_graph_pool = (0, 1)
            print("[gate] CONTROL: restored the stale mempool id (0, 1)", flush=True)
        del llm
        trace.append((i, before, "ok"))

    _teardown_colocate_process_group()
    _print_trace(trace)
    print("[gate] PASS: all engines built sequentially in one process", flush=True)
    return 0


def _print_trace(trace) -> None:
    print("\n[gate] free-before-build trace:", flush=True)
    for idx, free, status in trace:
        print(f"  engine {idx}: {free:6.2f} GiB  {status}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
