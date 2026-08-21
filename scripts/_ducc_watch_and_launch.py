#!/usr/bin/env python3
"""Robust two-phase orchestrator for v7_integrated experiment launcher.

Replaces earlier fragile bash-watcher that broke when ``exec`` replaced the
wrapper-shell PID mid-run. This version uses dynamic ``pgrep`` polling so it
survives subprocess-PID churn transparently, plus writes verbose status lines
into both stdout AND a persistent sidecar file under rounds/<exp>/logs/.

Lifecycle:
  Phase A wait : poll every 15s for any running evoguard.training.probes
                 process matching our config path argument;
                 abort after max_wait_minutes (=120) of zero such processes,
                 OR if artifact JSON never gets refreshed within grace window.
  Validation   : once no probe process remains, parse latest targets.json to
                 confirm schema_version='1' + non-empty recommended_target_modules
                 list + n_pairs_used>0 before greenlighting Phase B.
  Launch       : spawn ``python -m evoguard.run --config <v7.yaml>`` detached
                 from this parent's process group via os.setsid() so it survives
                 even if THIS orchestrator dies abruptly afterwards.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/ssd1/yx/yangxiao26/EvoGuard")
ARTIFACT_PATH = REPO_ROOT / "rounds/evoguard_agentdojo_full_local/probe_results/targets.json"
CONFIG_PATH_V7 = REPO_ROOT / "configs/agentdojo_full_v7_integrated.yaml"
EXP_NAME = "evoguard_agentdojo_full_v7_integrated"
LOG_DIR = REPO_ROOT / f"rounds/{EXP_NAME}/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SIDECAR_LOG = LOG_DIR / f"orchestrator_{datetime.now():%Y%m%d_%H%M%S}.log"

PROBE_PROCMATCH_RE = re.compile(r"training\.probes\s+\S*agentdojo_full_local\.yaml")
MAX_WAIT_MINUTES = 180            # hard ceiling on how long we'll keep watching for probe completion.


def log(msg: str) -> None:
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    try:
        with open(SIDECAR_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        # Sidecar write failure must NOT crash us -- just emit to stderr.
        print(f"[ORCH-sidecar-write-fail] {exc}", file=sys.stderr, flush=True)


def find_running_probe_pids() -> list[int]:
    """Return PIDs whose cmdline contains 'evoguard.training.probes' AND references
    configs/agentdojo_full_local.yaml -- avoids false positives if some unrelated
    probe invocation exists concurrently."""
    pids_found: list[int] = []
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        return pids_found
    for entry in proc_dir.iterdir():
        name = entry.name
        if not name.isdigit():
            continue
        cmd_path = entry / "cmdline"
        try:
            raw = cmd_path.read_bytes()
        except (OSError, PermissionError):
            continue
        if not raw:
            continue
        cmd_text = raw.decode("utf-8", errors="replace").replace("\x00", " ")
        if PROBE_PROCMATCH_RE.search(cmd_text):
            pids_found.append(int(name))
    return sorted(set(pids_found))


def validate_artifact_fresh(min_mtime_ts: float | None = None) -> tuple[bool, dict]:
    """Return ``(is_valid, parsed_dict_or_errinfo)``."""
    try:
        st = ARTIFACT_PATH.stat()
    except FileNotFoundError:
        return False, {"error": "file_not_found"}
    except OSError as exc:
        return False, {"error": f"os_error:{exc}"}
    if min_mtime_ts is not None and st.st_mtime < min_mtime_ts:
        return False, {"error": "stale_artifact", "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
                       "expected_after": datetime.fromtimestamp(min_mtime_ts).isoformat()}
    try:
        d = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:                                                # noqa: BLE001
        return False, {"error": f"json_load_fail:{exc}"}
    sv = str(d.get("schema_version", ""))
    mods = d.get("recommended_target_modules") or []
    sel = d.get("selected_blocks_sorted_desc") or []
    npairs_raw = int(d.get("n_pairs_used") or -1)
    info_dict = {
        "schema_version": sv,
        "n_pairs_used": npairs_raw,
        "top_k_value": int(d.get("top_k_value") or -1),
        "selected_blocks": sel[:12],
        "n_recommended_target_modules": len(mods),
        "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(),
    }
    valid = (sv == "1") and bool(mods) and npairs_raw >= 0
    return valid, {"valid": valid, **info_dict}


def _qianfan_credentials() -> tuple[str, str]:
    """QianFan appid + bearer token from the environment or the secrets file.

    Deliberately has no inline fallback literals: this file is committed, and a
    token pushed to a remote is a published token. Mirrors the contract in
    scripts/run_real.sh -- ``$EVOGUARD_SECRETS_FILE`` (default
    ``~/.evoguard_qianfan.env``) holds ``export EVOGUARD_QIANFAN_{APPID,TOKEN}=``.
    """
    appid = os.environ.get("EVOGUARD_QIANFAN_APPID", "")
    token = os.environ.get("EVOGUARD_QIANFAN_TOKEN", "")
    if not (appid and token):
        secrets_path = Path(
            os.environ.get("EVOGUARD_SECRETS_FILE")
            or (Path.home() / ".evoguard_qianfan.env")
        )
        if secrets_path.is_file():
            for line in secrets_path.read_text(encoding="utf-8").splitlines():
                m = re.match(
                    r"\s*(?:export\s+)?(EVOGUARD_QIANFAN_(?:APPID|TOKEN))\s*=\s*(.+?)\s*$",
                    line,
                )
                if not m:
                    continue
                value = m.group(2).strip().strip("'\"")
                if m.group(1).endswith("APPID") and not appid:
                    appid = value
                elif m.group(1).endswith("TOKEN") and not token:
                    token = value
    if not (appid and token):
        raise SystemExit(
            "[orchestrator] FATAL: QianFan credentials absent. Export "
            "EVOGUARD_QIANFAN_APPID/EVOGUARD_QIANFAN_TOKEN, or create "
            "~/.evoguard_qianfan.env exporting both."
        )
    return appid, token


def main() -> int:
    log("[orchestrator] started.")
    start_ts = time.time()
    min_acceptable_artifact_mtime = start_ts     # require refresh AFTER we started watching
                                                  # (i.e., tonight's rerun actually wrote something)
    last_seen_probe_count = -1
    poll_interval_s = 15
    waited_total_min = 0.0

    log(f"[orchestrator] waiting up to MAX={MAX_WAIT_MINUTES}m for any 'training.probes "
        f"...agentdojo_full_local.yaml...' process to disappear...")
    log(f"[orchestrator] requiring artifact at {ARTIFACT_PATH} be rewritten AFTER "
        f"{datetime.fromtimestamp(start_ts):%T}.")

    # ---- Phase-A wait loop -------------------------------------------------- #
    saw_at_least_one_alive_cycle = False
    while True:
        active_pids = find_running_probe_pids()
        now_elapsed_min = (time.time() - start_ts) / 60.0
        waited_total_min = now_elapsed_min

        if len(active_pids) != last_seen_probe_count:
            log(f"[orchestrator] probe-process count changed -> {len(active_pids)} "
                f"(pids={active_pids}); elapsed={now_elapsed_min:.1f}m")
            last_seen_probe_count = len(active_pids)

        if active_pids:
            saw_at_least_one_alive_cycle = True
            if now_elapsed_min >= MAX_WAIT_MINUTES:
                log(f"[orchestrator] ABORTING after {now_elapsed_min:.1f}m still seeing alive "
                    f"probes; manual intervention required.")
                return 11
            time.sleep(poll_interval_s)
            continue

        # No live probe found in this cycle.
        if not saw_at_least_one_alive_cycle:
            # Probe may have finished BEFORE we ever started scanning (race).
            # Still need artifact freshness check below to confirm whether tonite's rerun wrote anything.
            log("[orchestrator] WARNING: no live probe detected during any cycle yet; will validate "
                "existing artifact anyway since the original launched instance may have already exited cleanly.")
            break

        # We DID see at least one cycle where probe was alive; now all gone => real exit event.
        log(f"[orchestrator] probe(s) gone after total elapsed ~{now_elapsed_min:.1f}m; proceeding to artifact check.")
        break

    # Small grace period for filesystem flushes right at exit boundary.
    time.sleep(5)

    # ---- Artifact validation ----------------------------------------------- #
    ok, payload = validate_artifact_fresh(min_mtime_ts=min_acceptable_artifact_mtime)
    log(f"[orchestrator] artifact validation result: ok={ok}; details={payload}")
    if not ok:
        err_kind = payload.get("error", "<unknown>") if isinstance(payload, dict) else "?"
        log(f"[orchestrator] FATAL: artifact invalid or stale ({err_kind}). Will NOT auto-launch "
            f"main experiment because using default target modules would waste GPU cycles without "
            f"tonight's Δ-weighted selection benefit. Inspect manually then relaunch yourself.")
        return 22

    # ---- Main experiment launch -------------------------------------------- #
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_file = LOG_DIR / f"run_{ts_str}.log"
    qianfan_appid, qianfan_token = _qianfan_credentials()
    env = {
        **os.environ,
        "EVOGUARD_QIANFAN_APPID": qianfan_appid,
        "EVOGUARD_QIANFAN_TOKEN": qianfan_token,
        "EVOGUARD_VLLM_GPU":      os.environ.get("EVOGUARD_VLLM_GPU", "6"),
    }
    argv = [
        "/ssd1/conda_envs/evoguard/bin/python",
        "-m", "evoguard.run",
        "--config", str(CONFIG_PATH_V7),
    ]
    log(f"[orchestrator] launching MAIN EXPERIMENT:")
    log(f"               config : {CONFIG_PATH_V7}")
    log(f"               exp dir: round_{EXP_NAME}/")
    log(f"               logfile: {main_log_file}")
    log(f"               argv   : {' '.join(argv)}")

    # Use preexec_fn=os.setsid so child becomes its own session leader independent
    # of this orchestrator's lifetime. Even if we die later the experiment keeps running.
    try:
        with open(main_log_file, "w", encoding="utf-8") as fh_out:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=fh_out,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=env,
                start_new_session=True,           # equivalent to setsid()
            )
    except Exception as exc:                                              # noqa: BLE001
        log(f"[orchestrator] CRASH launching main experiment: {exc}")
        return 33

    Path("/tmp/ducc_phaseB_main.pid").write_text(str(proc.pid))
    log(f"[orchestrator] MAIN EXPERIMENT spawned successfully as PID={proc.pid}.")
    log(f"[orchestrator] tail command for live progress: tail -f '{main_log_file}'")
    log(f"[orchestrator] my own job here is done; exiting clean.")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
