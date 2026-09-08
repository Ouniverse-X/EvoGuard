"""Drive a colocate GRPO run to completion across as many process restarts as it takes.

WHY THIS EXISTS. `grpo_use_vllm_colocate: true` leaks ~19-21 GiB of the trainer card
per round (untraced; see memory/colocate_lifecycle.md), so a single process can only
train TWO rounds before `trainer.train()` dies on a genuine OOM. A 9-round run is
therefore 4 or 5 sequential processes, and each restart needs three config fields
changed in lockstep plus the trainer card emptied first. Doing that by hand is how
you end up resuming with `lora_adapter: null`, which silently collects rollouts
against the BASE model while GRPO warm-starts from trained weights.

Everything this script decides is READ FROM DISK OR FROM THE SERVER, never assumed:

  * the next `start_round` comes from `<exp>/latest_adapter_dir.txt`, i.e. the same
    marker the trainer warm-starts from. A round whose training died before saving
    leaves `grpo_native/rN/` without `adapter_weights` and leaves the marker at
    r(N-1), so that round is REDONE rather than skipped -- which is correct, and is
    exactly the case a hand-edit gets wrong.
  * the composite `lora_adapter` name comes from `GET <defender>/v1/models`, matched
    by the `root` field against that adapter dir. So the name is never guessed, and
    if the adapter is not registered on the endpoint we abort instead of running a
    leg against the wrong weights.

Usage (dry run first -- it prints the plan and touches nothing):
    python scripts/run_legged_experiment.py --config configs/agentdojo_llama31_grpo.yaml \
        --launch scripts/launch_agentdojo_llama31.sh --target-rounds 9 --dry-run

Then the real thing, itself under nohup because it outlives every leg:
    nohup python scripts/run_legged_experiment.py --config ... --launch ... \
        --target-rounds 9 > rounds/legged_supervisor.log 2>&1 &

It does NOT stop the serving endpoints at the end -- they are needed for the
held-out test replay -- but it does hand the trainer card back to the GPU keeper.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def log(msg: str) -> None:
    print(f"[legged {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- state


def read_config(path: str):
    """Only for READING. Writes go through `patch_config`, which keeps comments."""
    from evoguard.config import ExperimentConfig

    return ExperimentConfig.from_file(path)


def latest_trained_round(exp_dir: str) -> tuple[int, str]:
    """-> (round index whose adapter is saved, its adapter dir).

    `latest_adapter_dir.txt` is written only after weights land, so a round that
    crashed mid-training is invisible here and will be redone. r0 is the SFT path.
    """
    marker = os.path.join(exp_dir, "latest_adapter_dir.txt")
    with open(marker) as fh:
        adapter_dir = fh.read().strip()
    m = re.search(r"grpo_native/r(\d+)/", adapter_dir + "/")
    if m:
        return int(m.group(1)), adapter_dir
    if "sft_native" in adapter_dir:
        return 0, adapter_dir
    raise SystemExit(f"[legged] cannot read a round index out of {adapter_dir!r}")


def registered_adapter_name(base_url: str, adapter_dir: str) -> str:
    """Resolve the composite LoRA name the defender endpoint knows this dir by.

    Served `root` fields are repo-relative while the marker is absolute, so match on
    a suffix. Aborting here is the point: an unregistered adapter means the leg would
    run against whatever `lora_adapter` happens to say.
    """
    url = base_url.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.load(resp)
    want = adapter_dir.replace(REPO + "/", "").rstrip("/")
    hits = [e["id"] for e in payload.get("data", [])
            if str(e.get("root", "")).rstrip("/").endswith(want)]
    if not hits:
        raise SystemExit(f"[legged] no adapter on {url} has root ending {want!r}; "
                         f"registered: {[e['id'] for e in payload.get('data', [])]}")
    # Longest id = most composed name, which is the one the driver would extend.
    return max(hits, key=len)


# --------------------------------------------------------------------------- config


def _block_span(text: str, key: str) -> tuple[int, int]:
    """Character span of a top-level YAML block, so `lora_adapter` under `defense`
    is never confused with the one under `attacker` (which ships as null)."""
    m = re.search(rf"^{key}:\s*$", text, re.M)
    if m is None:
        raise SystemExit(f"[legged] no top-level `{key}:` block in the config")
    nxt = re.search(r"^[A-Za-z_][A-Za-z0-9_]*:", text[m.end():], re.M)
    return m.end(), m.end() + (nxt.start() if nxt else len(text) - m.end())


def _sub_once(text: str, pattern: str, repl: str, what: str,
              span: tuple[int, int] | None = None) -> str:
    lo, hi = span or (0, len(text))
    body, n = re.subn(pattern, repl, text[lo:hi], flags=re.M)
    if n != 1:
        raise SystemExit(f"[legged] expected exactly 1 match for {what}, got {n}")
    return text[:lo] + body + text[hi:]


def patch_config(path: str, start_round: int, max_rounds: int, adapter: str) -> None:
    """Rewrite the three resume fields in place, leaving comments byte-identical.

    Anchored at line start after whitespace, so commented-out mentions of the same
    keys (the config has several) cannot match.
    """
    with open(path) as fh:
        text = fh.read()
    pipe = _block_span(text, "pipeline")
    text = _sub_once(text, r"^(\s*start_round:\s*)\d+", rf"\g<1>{start_round}",
                     "pipeline.start_round", pipe)
    pipe = _block_span(text, "pipeline")
    text = _sub_once(text, r"^(\s*max_rounds:\s*)\d+", rf"\g<1>{max_rounds}",
                     "pipeline.max_rounds", pipe)
    dfn = _block_span(text, "defense")
    text = _sub_once(text, r"^(\s*lora_adapter:\s*)\S+", rf"\g<1>{adapter}",
                     "defense.llm.lora_adapter", dfn)
    with open(path, "w") as fh:
        fh.write(text)


# ------------------------------------------------------------------------ trainer gpu

_FREE_PROBE = "import torch;f,_=torch.cuda.mem_get_info(0);print(f/2**30)"


def free_gib(gpu: int) -> float:
    """Whole-card free VRAM, measured in a CHILD process.

    Two reasons it cannot be done inline: `nvidia-smi` on this box reports
    host-namespace pids and lies about occupancy, and initialising CUDA in the
    supervisor would leave a context on the very card the next leg must find empty.
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    out = subprocess.run([sys.executable, "-c", _FREE_PROBE], env=env,
                         capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise SystemExit(f"[legged] free-memory probe on gpu{gpu} failed: {out.stderr[-400:]}")
    return float(out.stdout.strip().splitlines()[-1])


def empty_trainer_card(gpu: int, need_gib: float, dry: bool) -> None:
    """`stop`, not `pause`: a paused keeper worker keeps a ~716 MiB CUDA context, and
    that shows up by pid in the OOM traceback when the last engine is squeezed."""
    keeper = os.path.join(REPO, "scripts", "gpu_keeper.sh")
    if dry:
        log(f"DRY: would run `gpu_keeper.sh stop -g {gpu}` then wait for >= {need_gib} GiB")
        return
    subprocess.run(["bash", keeper, "stop", "-g", str(gpu)], cwd=REPO, check=False)
    for _ in range(30):
        free = free_gib(gpu)
        if free >= need_gib:
            log(f"gpu{gpu} free = {free:.2f} GiB (>= {need_gib})")
            return
        log(f"gpu{gpu} free = {free:.2f} GiB, waiting for {need_gib}")
        time.sleep(10)
    raise SystemExit(f"[legged] gpu{gpu} never reached {need_gib} GiB free; refusing to "
                     f"launch a leg that would OOM. Something else is on the card.")


# ----------------------------------------------------------------------------- legs


def launch_leg(launch_script: str) -> tuple[int, str]:
    """-> (pid, log path), scraped from run_grpo_experiment.sh's own report.

    The launcher nohups python and returns, so waiting on the wrapper would return
    immediately; the pid it prints is the only handle on the actual run.
    """
    out = subprocess.run(["bash", launch_script], cwd=REPO, capture_output=True,
                         text=True, timeout=1800)
    sys.stdout.write(out.stdout)
    if out.returncode != 0:
        raise SystemExit(f"[legged] launcher failed rc={out.returncode}: {out.stderr[-800:]}")
    pid = re.search(r"^\s*pid\s*:\s*(\d+)", out.stdout, re.M)
    logp = re.search(r"^\s*log\s*:\s*(\S+)", out.stdout, re.M)
    if not pid or not logp:
        raise SystemExit("[legged] could not scrape pid/log out of the launcher output")
    return int(pid.group(1)), logp.group(1)


def wait_for_exit(pid: int, poll: float = 60.0) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(poll)


def hand_back_to_keeper(gpu: int, dry: bool) -> None:
    keeper = os.path.join(REPO, "scripts", "gpu_keeper.sh")
    if dry:
        log(f"DRY: would run `gpu_keeper.sh start -g {gpu}` then a sweeping `start`")
        return
    # Explicit -g bypasses EVOGUARD_KEEPER_EXCLUDE, which is the whole point here.
    subprocess.run(["bash", keeper, "start", "-g", str(gpu)], cwd=REPO, check=False)
    time.sleep(20)
    subprocess.run(["bash", keeper, "start"], cwd=REPO, check=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--launch", required=True, help="wrapper that exports the env contract")
    ap.add_argument("--target-rounds", type=int, required=True, help="same units as max_rounds")
    ap.add_argument("--rounds-per-leg", type=int, default=2)
    ap.add_argument("--trainer-gpu", type=int, default=7)
    ap.add_argument("--need-free-gib", type=float, default=70.0)
    ap.add_argument("--max-legs", type=int, default=8, help="runaway guard")
    ap.add_argument("--wait-pid", type=int, default=0,
                    help="wait for this already-running leg before planning the next")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = read_config(args.config)
    exp_dir = os.path.join(REPO, "rounds", cfg.name)
    base_url = cfg.defense.llm.base_url
    log(f"experiment={cfg.name} exp_dir={exp_dir} defender={base_url}")

    if args.wait_pid:
        log(f"waiting for the in-flight leg pid={args.wait_pid} before planning")
        if not args.dry_run:
            wait_for_exit(args.wait_pid)
            time.sleep(60)          # let the process release the card
        log(f"leg pid={args.wait_pid} has exited")

    for leg in range(1, args.max_legs + 1):
        done, adapter_dir = latest_trained_round(exp_dir)
        start = done + 1
        if start >= args.target_rounds:
            log(f"r{done} is the last trained round and target is {args.target_rounds}: "
                f"nothing left to do")
            break
        stop = min(start + args.rounds_per_leg, args.target_rounds)
        name = registered_adapter_name(base_url, adapter_dir)
        log(f"leg {leg}: r{start}..r{stop - 1}  (start_round={start} max_rounds={stop})")
        log(f"leg {leg}: warm-start adapter = {name}")
        if args.dry_run:
            log("DRY: stopping after planning the first leg")
            return 0

        patch_config(args.config, start, stop, name)
        empty_trainer_card(args.trainer_gpu, args.need_free_gib, args.dry_run)
        pid, logp = launch_leg(args.launch)
        log(f"leg {leg}: pid={pid} log={logp}")
        wait_for_exit(pid)
        log(f"leg {leg}: pid={pid} exited")
        time.sleep(60)

        after, _ = latest_trained_round(exp_dir)
        if after <= done:
            raise SystemExit(f"[legged] leg {leg} trained NO round (still r{after}); "
                             f"stopping rather than looping. Read {logp}.")
        log(f"leg {leg}: advanced r{done} -> r{after}")
    else:
        log(f"hit --max-legs {args.max_legs}; stopping")

    hand_back_to_keeper(args.trainer_gpu, args.dry_run)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())





