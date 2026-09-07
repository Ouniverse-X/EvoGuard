#!/usr/bin/env python
"""Offline PIGuard detection table over the camouflage probe's two arms.

Why this exists alongside the replay
------------------------------------
The replay arm (``configs/agentdojo_{stock,latent}_piguard.yaml``) answers
"does camouflage raise ASR against a filtered agent". This script answers the
*other* question -- the one the PIGuard/InjecGuard paper reports -- "does
camouflage lower the detector's recall", and it is also the **capability
control**: an ASR that barely moves is uninterpretable until we know the
checkpoint actually fires on something. Run it BEFORE trusting any PIGuard ASR.

Three settings per arm, all on the same 48 injected rows:

``payload``    the injection text ALONE. This is the paper's setting and the
               easiest possible case. Recall here near 0 means the checkpoint or
               the label mapping is wrong, not that the attack is stealthy.
``embedded``   ``observation`` = ``benign_observation + payload``, i.e. exactly
               the string the agent reads and exactly what
               ``PIGuardFilterAgent`` classifies. Windowed.
``truncated``  the same string under ``truncation=True, max_length=512``, the
               way ``bench_base/scripts/eval_piguard.py`` calls it. Reported to
               SHOW the 512-token confound rather than to hide it: the latent
               observations are longer, so any recall gap that exists only in
               this column is an artifact of the cut, not stealth.

False alarms are measured on the 46 ``clean`` rows' ``benign_observation``.
Those strings are **byte-identical between the two arms** by construction, so
there is one false-alarm number, not two -- if the script ever reports two
different ones the pairing has been broken (see
``tests/test_agentdojo_stock.py::Pairing``).

The latent-vs-stock comparison is paired (same task, same carrier, same benign
prefix, only ``payload`` differs), so significance is an exact two-sided McNemar
on the discordant counts, matching how the ASR arms are reported.

Usage (CPU, ~1 min, no server needed)::

    python scripts/probe_piguard_detection.py
    python scripts/probe_piguard_detection.py --device cuda:7 --json out.json
"""

from __future__ import annotations

import argparse
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

DEFAULT_MODEL = "/root/paddlejob/workspace/yangxiao/models/PIGuard"
ARMS = {"stock": "data/agentdojo_stock", "latent": "data/agentdojo_latent"}


def load_rows(root: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Return ``(injected_rows, benign_observations)`` sorted by ``instance_id``."""

    injected: list[dict[str, Any]] = []
    benign: list[str] = []
    for path in sorted(glob.glob(os.path.join(root, "scenarios", "*.jsonl"))):
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default="", help="also write the table as JSON here")
    args = ap.parse_args()

    detector = PIGuardDetector(args.model, device=args.device,
                               window=WINDOW_TOKENS, stride=WINDOW_STRIDE)
    # `predict` is the production path (windowed). `truncated` needs the raw
    # 512-token call, so reach through to the tokenizer/model directly -- via the
    # same loaded instance, so both columns share one checkpoint.
    detector._load()  # noqa: SLF001 - deliberate: one load for both columns
    torch = detector._torch  # noqa: SLF001

    def truncated_predict(text: str) -> bool:
        batch = detector._tokenizer(  # noqa: SLF001
            text, truncation=True, max_length=512, return_tensors="pt")
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = detector._model(**batch).logits  # noqa: SLF001
        return int(logits[0].argmax().item()) == detector._injection_id  # noqa: SLF001

    report: dict[str, Any] = {"model": args.model, "device": args.device, "arms": {}}
    flags: dict[str, dict[str, list[bool]]] = {}
    benign_by_arm: dict[str, list[str]] = {}

    for arm, root in ARMS.items():
        injected, benign = load_rows(root)
        benign_by_arm[arm] = benign
        cols = {
            "payload": [detector.predict(str(r["payload"])) for r in injected],
            "embedded": [detector.predict(str(r["observation"])) for r in injected],
            "truncated": [truncated_predict(str(r["observation"])) for r in injected],
        }
        flags[arm] = cols
        # Token lengths decide whether `truncated` is even a different call from
        # `embedded`. Without them a recall gap in that column is unattributable:
        # it could be the 512-token cut or it could be real stealth.
        tok = sorted(len(detector._tokenizer(str(r["observation"]),  # noqa: SLF001
                                            add_special_tokens=False)["input_ids"])
                     for r in injected)
        report["arms"][arm] = {
            "n_injected": len(injected),
            "instance_ids": [str(r["instance_id"]) for r in injected],
            "recall": {k: sum(v) / len(v) if v else None for k, v in cols.items()},
            "n_flagged": {k: sum(v) for k, v in cols.items()},
            "median_obs_chars": sorted(len(str(r["observation"])) for r in injected)[
                len(injected) // 2] if injected else None,
            "median_obs_tokens": tok[len(tok) // 2] if tok else None,
            "max_obs_tokens": tok[-1] if tok else None,
            "n_over_510_tokens": sum(1 for n in tok if n > WINDOW_TOKENS),
        }

    # One false-alarm number, and a hard check that it is legitimately one.
    if benign_by_arm["stock"] != benign_by_arm["latent"]:
        print("[FATAL] the two arms' benign observations differ -- the pairing is "
              "broken, so nothing below is a controlled comparison.", file=sys.stderr)
        return 2
    benign = benign_by_arm["stock"]
    fp = [detector.predict(text) for text in benign]
    report["clean"] = {
        "n": len(benign),
        "n_flagged": sum(fp),
        "false_alarm_rate": sum(fp) / len(fp) if fp else None,
    }

    ids = report["arms"]["stock"]["instance_ids"]
    if len(ids) != len(report["arms"]["latent"]["instance_ids"]):
        print("[FATAL] arms have different row counts; not paired.", file=sys.stderr)
        return 2

    print(f"PIGuard: {args.model}  device={args.device}  "
          f"window={WINDOW_TOKENS}/{WINDOW_STRIDE}")
    print(f"detector tally: {detector.tally}")
    print()
    print(f"{'setting':<12} {'stock recall':>14} {'latent recall':>15} "
          f"{'b(s>l)':>7} {'c(l>s)':>7} {'p(exact)':>9}")
    for setting in ("payload", "embedded", "truncated"):
        s = flags["stock"][setting]
        l = flags["latent"][setting]
        b = sum(1 for x, y in zip(s, l) if x and not y)
        c = sum(1 for x, y in zip(s, l) if y and not x)
        p = exact_mcnemar(b, c)
        report["arms"]["stock"].setdefault("mcnemar", {})[setting] = {
            "b_stock_only": b, "c_latent_only": c, "p_exact": p}
        print(f"{setting:<12} {sum(s)}/{len(s)} = {sum(s)/len(s):>6.3f}   "
              f"{sum(l)}/{len(l)} = {sum(l)/len(l):>6.3f}   "
              f"{b:>7} {c:>7} {p:>9.4f}")
    print()
    ca = report["clean"]
    print(f"clean false alarms: {ca['n_flagged']}/{ca['n']} = "
          f"{ca['false_alarm_rate']:.3f}  (identical strings on both arms)")
    for arm in ARMS:
        a = report["arms"][arm]
        print(f"{arm:>6} attacked observations: median {a['median_obs_chars']} chars / "
              f"{a['median_obs_tokens']} tokens, max {a['max_obs_tokens']} tokens, "
              f"{a['n_over_510_tokens']}/{a['n_injected']} exceed the {WINDOW_TOKENS}-token window")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json}")

    # The capability gate. `predict` fails OPEN on any exception, so a broken
    # forward signature or a missing file yields a clean-looking table of zeros
    # that is indistinguishable from a stealthy attack. Refuse to exit 0 on it:
    # every number above would be a measurement of nothing.
    if detector.tally.get("error"):
        print(f"\n[FATAL] {detector.tally['error']} classifications failed and fell "
              f"OPEN -- every recall above is a floor of 0, not a measurement. "
              f"tally={detector.tally}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
