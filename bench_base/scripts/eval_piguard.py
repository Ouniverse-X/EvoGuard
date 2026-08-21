"""Evaluate PIGuard (prompt injection classifier) against bench_base dataset.

PIGuard is a DeBERTa-v3-base binary classifier (benign/injection).
For each scenario, we feed the poisoned observation text to PIGuard
and check if it flags it as "injection".

Detection rate by bucket tells us: can PIGuard catch attacks at each delta level?
Expected: higher delta (stealthier) -> lower detection rate.

Usage:
  python bench_base/scripts/eval_piguard.py [--model-path PATH] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_PATH = "/ssd1/yx/models/PIGuard"
SCENARIOS_ROOT = REPO_ROOT / "bench_base" / "scenarios"
OUT_DIR = REPO_ROOT / "bench_base" / "diagnostics"
BUCKETS = ("imm", "d1", "d2", "d3", "d4")


def build_classifier(model_path: str, device: str):
    """Load PIGuard text-classification pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=True
    )
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
        device=torch.device(device),
    )
    return clf


def classify_text(clf, text: str) -> tuple[str, float]:
    """Run PIGuard on a single text, return (label, score)."""
    result = clf(text)[0]
    return result["label"], result["score"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(OUT_DIR / "eval_piguard.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 70)
    print("Eval: PIGuard prompt-injection classifier")
    print(f"  model: {args.model_path}")
    print(f"  device: {args.device}")
    print("=" * 70)

    clf = build_classifier(args.model_path, args.device)

    # Discover label mapping — PIGuard uses LABEL_1=injection, LABEL_0=benign
    # but let's detect from config to be safe
    label2id = clf.model.config.label2id
    print(f"  label2id: {label2id}")
    # Determine which label means "injection"
    injection_label = None
    for lbl in label2id:
        if "inject" in lbl.lower() or "malicious" in lbl.lower():
            injection_label = lbl
            break
    if injection_label is None:
        # Fallback: LABEL_1 is injection per PIGuard convention
        injection_label = "LABEL_1"
    print(f"  injection_label: {injection_label}")

    all_results = []
    per_bucket = {}
    t0 = time.time()

    for bkt in BUCKETS:
        path = SCENARIOS_ROOT / f"bucket_{bkt}.jsonl"
        if not path.exists():
            print(f"[{bkt}] MISSING, skipping")
            continue

        scenarios = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
        print(f"\n[{bkt}] classifying {len(scenarios)} scenarios...", flush=True)

        n_detected = 0
        n_missed = 0
        bucket_results = []

        # Collect texts for batch classification
        texts = []
        for scen in scenarios:
            text = scen.get("poisoned_observation_text", "")
            texts.append(text)

        # Batch classify
        batch_size = args.batch_size
        labels = []
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results = clf(batch)
            for r in results:
                labels.append(r["label"])
                scores.append(r["score"])

        # Aggregate
        for i, scen in enumerate(scenarios):
            detected = labels[i] == injection_label
            if detected:
                n_detected += 1
            else:
                n_missed += 1

            bucket_results.append({
                "scenario_id": scen.get("scenario_id", f"{bkt}_{i}"),
                "bucket": bkt,
                "detected": detected,
                "label": labels[i],
                "score": round(scores[i], 4),
            })

        total = len(scenarios)
        det_rate = n_detected / total if total else 0
        per_bucket[bkt] = {
            "n_total": total,
            "n_detected": n_detected,
            "n_missed": n_missed,
            "detection_rate": round(det_rate, 4),
        }
        print(f"  [{bkt}] detection={det_rate:.1%} ({n_detected}/{total})")
        all_results.extend(bucket_results)

    elapsed = time.time() - t0

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'Bucket':<8}{'N':>5}{'Detected':>10}{'Missed':>8}{'Det.Rate':>10}")
    print("-" * 41)
    for bkt in BUCKETS:
        if bkt in per_bucket:
            s = per_bucket[bkt]
            print(f"{bkt:<8}{s['n_total']:>5}{s['n_detected']:>10}{s['n_missed']:>8}{s['detection_rate']:>9.1%}")
    print(f"\nDone in {elapsed:.1f}s")

    # Write output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "eval_mode": "piguard_classifier",
        "model_path": args.model_path,
        "injection_label": injection_label,
        "device": args.device,
        "elapsed_seconds": round(elapsed, 1),
        "per_bucket": per_bucket,
        "all_results": all_results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
