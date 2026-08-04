"""Aggressively grab an idle GPU before competing jobs steal it.
Tries each GPU in parallel-friendly tight loop; first successful alloc wins."""
import os, sys, time

import torch

HOLD_GB = float(os.environ.get("HOLDER_GB_PER_GPU", "32"))
N_ELEMS = int(HOLD_GB * (1024**3) / 2)

# Try multiple rounds because contention is volatile.
MAX_ROUNDS = int(os.environ.get("GRAB_MAX_ROUNDS", "60"))
SLEEP_BETWEEN_ROUNDS_SEC = float(os.environ.get("GRAB_SLEEP", "2"))

held_gpu = None
held_tensor = None

for rnd in range(MAX_ROUNDS):
    free_info = []
    for g in range(torch.cuda.device_count()):
        try:
            free_b, _total_b = torch.cuda.mem_get_info(g)
            free_info.append((g, free_b))
        except Exception as e:
            print(f"[grab] err reading gpu{g}: {e}", flush=True)
    # Sort by most-free descending — try greediest targets first.
    free_info.sort(key=lambda x: -x[1])
    for g, free_b in free_info:
        need_b = HOLD_GB * (1024**3)
        if free_b < need_b * 1.05:    # leave headroom for nvidia-smi overhead etc.
            continue
        try:
            t = torch.empty(N_ELEMS, dtype=torch.float16, device=f"cuda:{g}")
            held_gpu = g
            held_tensor = t
            print(f"[grab] SUCCESS grabbed GPU#{g} ({HOLD_GB}GB)", flush=True)
            break
        except RuntimeError as e:
            print(f"[grab] failed on GPU#{g}: {str(e)[:120]}", flush=True)
    if held_gpu is not None:
        break
    print(f"[grab] round {rnd+1}/{MAX_ROUNDS} no win yet; sleeping {SLEEP_BETWEEN_ROUNDS_SEC}s",
          flush=True)
    time.sleep(SLEEP_BETWEEN_ROUNDS_SEC)

if held_gpu is None:
    print("[grab] FAILED after all rounds.", flush=True)
    sys.exit(99)

print(f"[grab] holding GPU#{held_gpu}; kill PID {os.getpid()} or run "
      f"`kill $(cat {os.environ.get('PID_FILE','rounds/gpu_grabber.pid')})` to release.",
      flush=True)

pid_file = os.environ.get('PID_FILE', 'rounds/gpu_grabber.pid')
with open(pid_file, 'w') as f:
    f.write(str(os.getpid()))
fpath = os.environ.get('HELD_GPU_FILE', 'rounds/gpu_held.txt')
with open(fpath, 'w') as f:
    f.write(str(held_gpu))

while True:
    time.sleep(3600)
