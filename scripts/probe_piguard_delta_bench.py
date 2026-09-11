#!/usr/bin/env python
"""Offline PIGuard detection table over Δ-Bench, per tier and per source.

Why this exists alongside the replay
------------------------------------
``configs/delta_bench_piguard.yaml`` answers "does a more concealed payload raise
ASR against a filtered agent". This script answers the *other* question -- the one
the PIGuard/InjecGuard paper reports -- "does concealment lower the detector's
recall", and it is also the **capability control**: ``PIGuardFilterAgent`` fails
OPEN, so a run where the checkpoint never loaded is indistinguishable from the
undefended base arm by ASR alone. Run this BEFORE trusting any PIGuard ASR.

Three settings, on every injected row:

``payload``    the injection text ALONE. The paper's setting and the easiest
               possible case. Recall near 0 here means the checkpoint or the label
               mapping is wrong, not that the attack is stealthy.
``embedded``   ``observation`` = benign text + payload, i.e. exactly the string the
               agent reads and exactly what ``PIGuardFilterAgent`` classifies.
               Windowed at 510/255.
``truncated``  the same string under ``truncation=True, max_length=512``.
               Reported to SHOW the 512-token confound, not to hide it: this
               ladder's deferred payloads are LONGER by design
               (payload_chars_mean 262/637/961/878), so a recall gap that exists
               only in this column is an artifact of the cut, not of concealment.

Read it per stratum, never pooled
---------------------------------
The ladder is ragged -- only the 19 core agentdojo groups carry all four rungs;
18 agentdojo extension groups carry ``imm``+``d1``, and the 10 ``asb`` + 10
``injecagent`` groups carry ``imm`` alone. A pooled ``imm`` cell would mix 57
groups against 19 in ``d2``/``d3``, so every table below is printed per stratum
and the four-rung curve is the core stratum's alone.

The tier is a COMPOUND manipulation (deferral depth AND pretext elaboration are
graded together), so a recall gradient here is a *concealment* effect and cannot
be attributed to Δ. Tiers within a case group are paired by construction, so
significance is an exact two-sided McNemar on the discordant counts, matching how
the ASR arms are reported.

False alarms are measured on the 18 ``bucket_clean.jsonl`` rows'
``benign_observation`` -- the core groups' clean twins. The extension, ASB and
InjecAgent groups ship none, so the false-alarm rate is a core-only figure.

Usage (CPU, no server needed)::

    python scripts/probe_piguard_delta_bench.py
    python scripts/probe_piguard_delta_bench.py --device cuda:7 --json out.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evoguard.agents.piguard_filter import (  # noqa: E402
    WINDOW_STRIDE,
    WINDOW_TOKENS,
    PIGuardDetector,
)
from evoguard.envs.delta_bench import TIERS  # noqa: E402

DEFAULT_MODEL = "/root/paddlejob/workspace/yangxiao/models/PIGuard"
SCENARIO_DIR = "data/delta_bench/scenarios"
SETTINGS = ("payload", "embedded", "truncated")
STRATUM_ORDER = ("agentdojo:core", "agentdojo:ext", "asb", "injecagent")


def stratum_of(row: dict[str, Any]) -> str:
    """The cell a row may be pooled inside. Never pool ACROSS these."""

    source = str(row["source"])
    if source != "agentdojo":
        return source
    return ("agentdojo:core"
            if len(row.get("group_tiers") or []) == len(TIERS)
            else "agentdojo:ext")


def load_rows() -> tuple[list[dict[str, Any]], list[str]]:
    """``(injected_rows, clean benign observations)``, both deterministically ordered."""

    injected: list[dict[str, Any]] = []
    benign: list[str] = []
    for path in sorted(glob.glob(os.path.join(SCENARIO_DIR, "bucket_*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("sample_type") == "injected":
                    injected.append(row)
                else:
                    benign.append(str(row["benign_observation"]))
    injected.sort(key=lambda r: str(r["instance_id"]))
    return injected, sorted(benign)


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts ``b`` and ``c``."""

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def median(values: list[int]):
    return sorted(values)[len(values) // 2] if values else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default="", help="also write the tables as JSON here")
    args = ap.parse_args()

    injected, benign = load_rows()
    if not injected:
        print(f"[FATAL] no injected rows under {SCENARIO_DIR}", file=sys.stderr)
        return 2

    detector = PIGuardDetector(args.model, device=args.device,
                               window=WINDOW_TOKENS, stride=WINDOW_STRIDE)
    # `predict` is the production path (windowed). `truncated` needs the raw
    # 512-token call, so reach through to the tokenizer/model directly -- via the
    # same loaded instance, so all three columns share one checkpoint.
    detector._load()  # noqa: SLF001 - deliberate: one load for every column
    torch = detector._torch  # noqa: SLF001

    def truncated_predict(text: str) -> bool:
        batch = detector._tokenizer(  # noqa: SLF001
            text, truncation=True, max_length=512, return_tensors="pt")
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = detector._model(**batch).logits  # noqa: SLF001
        return int(logits[0].argmax().item()) == detector._injection_id  # noqa: SLF001

    def n_tokens(text: str) -> int:
        return len(detector._tokenizer(  # noqa: SLF001
            text, add_special_tokens=False)["input_ids"])

    # One pass over every row, one dict per row. Classifying here rather than
    # inside the per-stratum loops keeps each string classified exactly once, so
    # the tally in the capability gate below counts real work.
    scored: list[dict[str, Any]] = []
    for row in injected:
        obs = str(row["observation"])
        scored.append({
            "instance_id": str(row["instance_id"]),
            "case_id": str(row["case_id"]),
            "tier": str(row["tier"]),
            "source": str(row["source"]),
            "stratum": stratum_of(row),
            "suite": str(row["suite"]),
            "payload_chars": len(str(row["payload"])),
            "obs_chars": len(obs),
            "obs_tokens": n_tokens(obs),
            "payload": detector.predict(str(row["payload"])),
            "embedded": detector.predict(obs),
            "truncated": truncated_predict(obs),
        })

    fp = [detector.predict(text) for text in benign]

    report: dict[str, Any] = {
        "model": args.model,
        "device": args.device,
        "window": [WINDOW_TOKENS, WINDOW_STRIDE],
        "n_injected": len(scored),
        "clean": {
            "n": len(benign),
            "n_flagged": sum(fp),
            "false_alarm_rate": (sum(fp) / len(fp)) if fp else None,
            "note": "core agentdojo groups only -- the other strata ship no clean twin",
        },
        "strata": {},
    }

    print(f"PIGuard: {args.model}  device={args.device}  "
          f"window={WINDOW_TOKENS}/{WINDOW_STRIDE}")
    print(f"detector tally: {detector.tally}")
    print(f"rows: {len(scored)} injected, {len(benign)} clean")
    ca = report["clean"]
    print(f"clean false alarms: {ca['n_flagged']}/{ca['n']} = "
          f"{ca['false_alarm_rate']:.3f}   (core groups only)")

    strata = [s for s in STRATUM_ORDER if any(r["stratum"] == s for r in scored)]
    strata += sorted({r["stratum"] for r in scored} - set(strata))

    for stratum in strata:
        srows = [r for r in scored if r["stratum"] == stratum]
        tiers = [t for t in TIERS if any(r["tier"] == t for r in srows)]
        print()
        print("=" * 92)
        print(f"stratum: {stratum}   rows: {len(srows)}   rungs: {','.join(tiers)}"
              + ("   <-- the four-rung headline" if len(tiers) == len(TIERS) else ""))
        head = (f"{'tier':<6}{'n':>5}" + "".join(f"{s + ' recall':>18}" for s in SETTINGS)
                + f"{'payload_chars':>15}{'obs_tokens':>12}{'>510tok':>9}")
        print(head)
        print("-" * len(head))
        cell: dict[str, Any] = {}
        for tier in tiers:
            trows = [r for r in srows if r["tier"] == tier]
            line = f"{tier:<6}{len(trows):>5}"
            recalls = {}
            for setting in SETTINGS:
                k = sum(1 for r in trows if r[setting])
                recalls[setting] = k / len(trows)
                line += f"{f'{k}/{len(trows)} = {k / len(trows):.3f}':>18}"
            over = sum(1 for r in trows if r["obs_tokens"] > WINDOW_TOKENS)
            line += (f"{median([r['payload_chars'] for r in trows]):>15}"
                     f"{median([r['obs_tokens'] for r in trows]):>12}"
                     f"{f'{over}/{len(trows)}':>9}")
            print(line)
            cell[tier] = {
                "n": len(trows),
                "recall": recalls,
                "n_flagged": {s: sum(1 for r in trows if r[s]) for s in SETTINGS},
                "median_payload_chars": median([r["payload_chars"] for r in trows]),
                "median_obs_tokens": median([r["obs_tokens"] for r in trows]),
                "max_obs_tokens": max(r["obs_tokens"] for r in trows),
                "n_over_window": over,
            }
        report["strata"][stratum] = {"tiers": cell}

        if len(tiers) < 2:
            print("single rung -- no ladder to pair; the row above is the whole "
                  "measurement for this stratum")
            continue

        # Paired on `case_id`: the two rows then share task, clean plan, carrier,
        # benign prefix, sink and args, and differ ONLY in payload. An unpaired
        # difference of two rungs' marginal recalls throws that design away.
        print(f"\npaired within case group [{stratum}] -- exact two-sided McNemar")
        print(f"{'setting':<11}{'pair':<11}{'n':>5}{'both':>6}{'neither':>9}"
              f"{'a_only':>8}{'b_only':>8}{'Δrecall':>10}{'p':>9}")
        pairs_out: dict[str, list] = {}
        for setting in SETTINGS:
            for i in range(len(tiers)):
                for j in range(i + 1, len(tiers)):
                    ta, tb = tiers[i], tiers[j]
                    a = {r["case_id"]: r for r in srows if r["tier"] == ta}
                    b = {r["case_id"]: r for r in srows if r["tier"] == tb}
                    common = sorted(set(a) & set(b))
                    if not common:
                        continue
                    only_a = sum(1 for k in common if a[k][setting] and not b[k][setting])
                    only_b = sum(1 for k in common if b[k][setting] and not a[k][setting])
                    both = sum(1 for k in common if a[k][setting] and b[k][setting])
                    neither = len(common) - both - only_a - only_b
                    d = (sum(1 for k in common if b[k][setting])
                         - sum(1 for k in common if a[k][setting])) / len(common)
                    p = exact_mcnemar(only_a, only_b)
                    print(f"{setting:<11}{ta + '->' + tb:<11}{len(common):>5}{both:>6}"
                          f"{neither:>9}{only_a:>8}{only_b:>8}{d:>+10.3f}{p:>9.4f}")
                    pairs_out.setdefault(setting, []).append({
                        "pair": f"{ta}->{tb}", "n_pairs": len(common),
                        "both": both, "neither": neither,
                        f"{ta}_only": only_a, f"{tb}_only": only_b,
                        "delta_recall": d, "p_exact": p,
                    })
        report["strata"][stratum]["mcnemar"] = pairs_out

    by_suite = collections.defaultdict(list)
    for r in scored:
        by_suite[r["suite"]].append(r)
    print()
    print("by suite (pools tiers AND strata -- composition, not an effect)")
    for suite in sorted(by_suite):
        rows = by_suite[suite]
        k = sum(1 for r in rows if r["embedded"])
        print(f"  {suite:<12}{len(rows):>4} rows   embedded recall "
              f"{k}/{len(rows)} = {k / len(rows):.3f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json}")

    # The capability gate. `predict` fails OPEN on any exception, so a broken
    # forward signature or a missing file yields a clean-looking table of zeros
    # that is indistinguishable from a perfectly concealed attack. Refuse to exit
    # 0 on it: every number above would be a measurement of nothing.
    if detector.tally.get("error"):
        print(f"\n[FATAL] {detector.tally['error']} classifications failed and fell "
              f"OPEN -- every recall above is a floor of 0, not a measurement. "
              f"tally={detector.tally}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
