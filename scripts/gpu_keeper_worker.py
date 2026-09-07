#!/usr/bin/env python
"""One GPU-holding worker. Launched per card by scripts/gpu_keeper.sh.

Holds ~`--target-mib` of VRAM as ballast and keeps SM utilisation near
`--util` with a duty-cycled fp16 matmul, so the card looks (and is) busy
rather than merely allocated.

Backs off in two independent ways, checked every `--poll` seconds:

  * PAUSE FLAG -- `<flag-dir>/pause_all.flag` or `<flag-dir>/pause_<gpu>.flag`.
    Written by `gpu_keeper.sh pause -g N`. This is the path an operator MUST
    use before starting anything real on the card: vLLM and the GRPO colocate
    engine both profile WHOLE-CARD free memory at construction, so they fail
    to start rather than wait if the ballast is still resident.
  * FOREIGN MEMORY -- a rise in the card's `memory.used` that this process did
    not cause. Above `--foreign-limit-mib` for two consecutive polls the
    ballast is dropped and the worker idles until the card is ours again. This
    is the safety net for a tenant that appears without anyone calling pause;
    it cannot help a tenant that needs the memory at ITS startup instant, which
    is why the flag exists.

Foreign memory is measured as a DELTA against a startup baseline, not by
filtering `--query-compute-apps` on our own pid: this box runs under
/opt/baidu-cgpu, which reports host-namespace pids there, so no pid we can see
from inside ever matches and the worker classified its own 71 GiB as foreign.

Release is a full teardown (drop tensors + empty_cache), so `nvidia-smi` shows
the card back at its foreign-only footprint within a poll interval.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

MIB = 1024 * 1024


def _card_used_mib(gpu_index: int) -> int:
    """Total memory.used on `gpu_index`, every tenant included."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--format=csv,noheader,nounits",
             "--query-gpu=memory.used"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return int(float(out.splitlines()[0].strip()))
    except Exception:                                        # noqa: BLE001
        return 0


class Holder:
    """Owns every GPU allocation this worker makes, so release is one call."""

    def __init__(self, target_mib: int, matmul_n: int) -> None:
        import torch
        self.torch = torch
        self.target_mib = int(target_mib)
        self.matmul_n = int(matmul_n)
        self.ballast: list = []
        self.a = self.b = self.c = None

    @property
    def held_mib(self) -> int:
        if self.a is None:
            return 0
        work = 3 * self.matmul_n * self.matmul_n * 2 // MIB
        return len(self.ballast) * 1024 + work

    @property
    def reserved_mib(self) -> int:
        """What the caching allocator holds from the driver, ballast included."""
        try:
            return int(self.torch.cuda.memory_reserved() // MIB)
        except Exception:                                    # noqa: BLE001
            return self.held_mib

    def touch_cuda(self) -> None:
        """Create the CUDA context so the startup baseline includes it."""
        self.torch.zeros(1, device="cuda")
        self.torch.cuda.synchronize()

    def acquire(self) -> None:
        """Idempotent: fills up to the target in 1 GiB chunks, stops on OOM."""
        torch = self.torch
        if self.a is None:
            n = self.matmul_n
            self.a = torch.randn((n, n), device="cuda", dtype=torch.float16)
            self.b = torch.randn((n, n), device="cuda", dtype=torch.float16)
            self.c = torch.empty((n, n), device="cuda", dtype=torch.float16)
        while self.held_mib + 1024 <= self.target_mib:
            try:
                self.ballast.append(
                    torch.empty(1024 * MIB, dtype=torch.uint8, device="cuda")
                )
            except RuntimeError:                             # card genuinely full
                break

    def release(self) -> None:
        self.ballast.clear()
        self.a = self.b = self.c = None
        try:
            self.torch.cuda.empty_cache()
        except Exception:                                    # noqa: BLE001
            pass

    def burn(self, seconds: float, util: float) -> None:
        """Duty-cycled matmul: busy for `util` of the wall time, idle for the rest."""
        if self.a is None:
            time.sleep(seconds)
            return
        torch = self.torch
        deadline = time.time() + seconds
        util = min(max(util, 0.01), 1.0)
        while time.time() < deadline:
            t0 = time.time()
            for _ in range(8):
                torch.matmul(self.a, self.b, out=self.c)
            torch.cuda.synchronize()
            busy = time.time() - t0
            if util < 1.0:
                time.sleep(busy * (1.0 / util - 1.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True,
                    help="PHYSICAL index, used for nvidia-smi. Torch always "
                         "sees cuda:0 because the launcher pins "
                         "CUDA_VISIBLE_DEVICES to this one card.")
    ap.add_argument("--target-mib", type=int, default=71680)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--matmul-n", type=int, default=8192)
    ap.add_argument("--foreign-limit-mib", type=int, default=4096,
                    help="Tolerance for the foreign-memory delta. Generous on "
                         "purpose: nvidia-smi's memory.used lags our own "
                         "alloc/free, while any real tenant worth yielding to "
                         "(a 7B vLLM, a trainer) takes tens of GiB.")
    ap.add_argument("--poll", type=float, default=2.0,
                    help="Pause-flag responsiveness. Cheap (two os.path.exists), "
                         "so it also bounds how long a `pause` call blocks.")
    ap.add_argument("--foreign-poll", type=float, default=10.0,
                    help="Foreign-memory check cadence. Deliberately slower than "
                         "--poll: each check forks nvidia-smi for ~0.1-0.2 s, "
                         "which at --poll cadence alone costs ~7 points of the "
                         "utilisation the worker exists to produce.")
    ap.add_argument("--flag-dir", default="rounds/gpu_keeper")
    args = ap.parse_args()

    pid = os.getpid()
    flag_dir = os.path.abspath(args.flag_dir)
    os.makedirs(flag_dir, exist_ok=True)
    pause_files = (os.path.join(flag_dir, "pause_all.flag"),
                   os.path.join(flag_dir, f"pause_{args.gpu}.flag"))
    state_file = os.path.join(flag_dir, f"state_{args.gpu}.json")

    stopping = {"now": False}

    def _stop(signum, _frame):
        stopping["now"] = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    holder = Holder(args.target_mib, args.matmul_n)
    holder.touch_cuda()
    # Baseline is taken AFTER the CUDA context exists and BEFORE any ballast, so
    # it absorbs both our own context (~300-500 MiB) and whatever was already on
    # the card. Foreign is then the rise above it that our allocator cannot
    # account for -- which is what makes the measurement pid-namespace-proof.
    baseline_mib = _card_used_mib(args.gpu)
    print(f"[keeper gpu{args.gpu}] pid={pid} target={args.target_mib}MiB "
          f"util={args.util} baseline_used={baseline_mib}MiB flag_dir={flag_dir}",
          flush=True)

    last_report = 0.0
    last_foreign_check = 0.0
    foreign = 0
    card_used = baseline_mib
    foreign_strikes = 0
    while not stopping["now"]:
        paused = any(os.path.exists(p) for p in pause_files)
        now = time.time()
        if now - last_foreign_check >= args.foreign_poll:
            last_foreign_check = now
            card_used = _card_used_mib(args.gpu)
            foreign = max(0, card_used - baseline_mib - holder.reserved_mib)
            # Two strikes, because memory.used lags our own alloc/free by a poll
            # or two and a single noisy sample would make the worker flap
            # between holding and releasing 70 GiB.
            foreign_strikes = (foreign_strikes + 1
                               if foreign > args.foreign_limit_mib else 0)
        blocked = paused or foreign_strikes >= 2
        reason = "paused" if paused else ("foreign" if blocked else "holding")

        if blocked:
            if holder.held_mib:
                print(f"[keeper gpu{args.gpu}] backing off ({reason}, "
                      f"foreign={foreign}MiB) -- releasing {holder.held_mib}MiB",
                      flush=True)
                holder.release()
            time.sleep(args.poll)
        else:
            holder.acquire()
            holder.burn(args.poll, args.util)

        now = time.time()
        # State file is rewritten EVERY iteration, not on the 60 s log cadence:
        # `gpu_keeper.sh pause` polls held_mib here to decide when the card is
        # actually free, and a 60 s-stale file would report a hold that is long
        # gone (or worse, miss one that is not).
        payload = {"gpu": args.gpu, "pid": pid, "state": reason,
                   "held_mib": holder.held_mib, "foreign_mib": foreign,
                   "card_used_mib": card_used, "baseline_mib": baseline_mib,
                   "target_mib": args.target_mib, "util": args.util,
                   "ts": int(now)}
        try:
            with open(state_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except OSError:
            pass
        if now - last_report >= 60:
            last_report = now
            print(f"[keeper gpu{args.gpu}] {reason} held={holder.held_mib}MiB "
                  f"foreign={foreign}MiB", flush=True)

    holder.release()
    print(f"[keeper gpu{args.gpu}] stopped, memory released", flush=True)
    try:
        os.remove(state_file)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
