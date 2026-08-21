"""GPU 占卡进程：在指定 GPU 上分配大块显存防止被其他作业抢占，等待实验接管。

kill 进程（或 kill $(cat rounds/gpu_holder.pid)）即可立即释放。
"""
import os
import sys
import time

import torch

gpus = [int(x) for x in os.environ.get("HOLDER_GPUS", "1,2").split(",") if x.strip()]
hold_gb = float(os.environ.get("HOLDER_GB_PER_GPU", "33"))
n_elems = int(hold_gb * (1024 ** 3) / 2)  # float16 = 2 bytes/elem

held = []
for g in gpus:
    try:
        t = torch.empty(n_elems, dtype=torch.float16, device=f"cuda:{g}")
        held.append((g, t))
        print(f"[gpu_holder] occupied GPU{g}: {hold_gb}GB held", flush=True)
    except RuntimeError as e:
        print(f"[gpu_holder] WARN GPU{g} alloc failed: {e}", flush=True)

print(f"[gpu_holder] holding {len(held)} GPU(s); kill PID {os.getpid()} to release", flush=True)
print(f"[gpu_holder] holding GPUs: {[g for g, _ in held]}", flush=True)

while True:
    time.sleep(3600)
