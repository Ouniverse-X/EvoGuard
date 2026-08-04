"""Compute per-token Shannon entropy metrics from collected harness responses.

Reads ``raw_responses.jsonl`` produced by ``preliminary.harness.HarnessRunner.run``
and emits ``entropy_per_scenario.csv`` with columns fixed per spec §5:

    scenario_id,bucket,delta_value_orig,domain,method,truncated,
    n_tokens_generated,H_token_mean_full_response,H_token_std_full_response,
    H_first_k_mean_at_K_eq_64,H_normalized_token_mean,p_argmax_mean,
    whitespace_excluded_H_token_mean,prompt_hash

For each token position t we compute Shannon entropy over returned top-K probability
mass:  H_t^raw = -sum_k p_{t,k} ln p_{t,k}. Top-K probabilities are renormalized to
sum to unity defensively against minor float32 rounding drift introduced by vLLM's
JSON serialization path.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from typing import Any

if __package__ in (None, ""):                                                   # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from preliminary.config import ExperimentConfig, EntropyConfig

logger = logging.getLogger("preliminary.entropy")


# --------------------------------------------------------------------------- #
# Per-token helpers
# --------------------------------------------------------------------------- #
def _renormalize_topk(topk_entries: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """Return list[(token_str, prob)] after dropping None-prob entries & sum-renormalizing."""
    pairs: list[tuple[str, float]] = []
    for e in topk_entries or []:
        if not isinstance(e, dict):
            continue
        p_raw = e.get("prob")
        if p_raw is None:
            lp = e.get("logprob")
            if lp is not None:
                try:
                    p_raw = float(math.exp(float(lp)))
                except Exception:                                                  # noqa: BLE001 - skip unparseable entries silently
                    continue
            else:
                continue
        try:
            p_f = float(p_raw)
        except Exception:                                                          # noqa: BLE001
            continue
        if p_f <= 0:
            continue
        s_ = str(e.get("token_str") if "token_str" in e else "")
        pairs.append((s_, p_f))
    s_sum = sum(p for _, p in pairs)
    if s_sum <= 0:
        return []
    return [(s_, p / s_sum) for _, p in [(s_, px) for s_, px in pairs]]


def _shannon_entropy_nats(pairs_renormed: list[tuple[Any, float]]) -> tuple[float, int]:
    """Returns (H_in_nats, K_effective_count_of_nonzero_mass_alternatives_used).

    Uses natural-log nats throughout — consistent across all downstream computations;
    conversion to bits happens only at reporting time if requested by callers via division-by-ln(2).
    """
    h_val = 0.0
    k_eff = 0
    for _, p in pairs_renormed:
        k_eff += 1
        h_val -= p * math.log(p) if p > 0 else 0.0
    return (h_val, max(k_eff, 1))


def _is_whitespace_only_token(text_repr: str) -> bool:
    return text_repr.strip() == ""


# --------------------------------------------------------------------------- #
# Per-scenario metric computation
# --------------------------------------------------------------------------- #
def compute_metrics_for_record(record: dict[str, Any],
                                cfg_K_floor_for_norm: int = 20,
                                first_k_window_size: int = 64) -> dict[str, Any]:
    """Produce a single row dict ready to be appended into a CSV writer."""

    sid = record.get("scenario_id", "?")
    bucket_label = record.get("bucket", "?")
    delta_orig = record.get("delta_value_orig")
    domain = record.get("domain", "?")
    method_tag = record.get("method", "?")
    truncated_flag = bool(record.get("truncated", False))
    prompt_hash_digest = record.get("prompt_hash", "")

    finish_reason_raw = record.get("finish_reason")

    # If error-row / empty generation marker present return NaN-filled placeholder preserving schema integrity.
    err_msg = record.get("error_message") or ""
    is_error_row = bool(err_msg.strip()) or finish_reason_raw == "error"

    tokens_meta_list = record.get("tokens") or []

    if is_error_row or len(tokens_meta_list) == 0:
        nan_or_zero = float("nan")
        return {
            "scenario_id": sid,
            "bucket": bucket_label,
            "delta_value_orig": _fmt_int_safe(delta_orig),
            "domain": domain,
            "method": method_tag,
            "truncated": truncated_flag,
            "n_tokens_generated": 0,
            "H_token_mean_full_response": nan_or_zero,
            "H_token_std_full_response": nan_or_zero,
            "H_first_k_mean_at_K_eq_64": nan_or_zero,
            "H_normalized_token_mean": nan_or_zero,
            "p_argmax_mean": nan_or_zero,
            "whitespace_excluded_H_token_mean": nan_or_zero,
            "prompt_hash": prompt_hash_digest,
            "_finish_reason": finish_reason_raw or "",
            "_error_message_excerpt": err_msg[:120] if err_msg else "",
        }

    per_token_h_nats: list[float] = []
    per_token_argmax_prob: list[float] = []
    per_token_k_eff: list[int] = []
    whitespace_excluded_per_token_h_nats: list[float] = []

    for tok_entry in tokens_meta_list:
        topk_arr = tok_entry.get("logprobs_topk") or []
        argmax_prob_field = tok_entry.get("argmax_prob")
        token_text_repr = tok_entry.get("text", "") or ""
        norm_pairs = _renormalize_topk(topk_arr)

        if not norm_pairs:
            continue                                                              # cannot compute H without any prob mass info

        h_t, k_eff_t = _shannon_entropy_nats(norm_pairs)
        per_token_h_nats.append(h_t)
        per_token_k_eff.append(k_eff_t)

        # Determine argmax prob either from explicit field OR from largest entry within normalized topk.
        sorted_probs_desc = [p for _, p in sorted(norm_pairs, key=lambda kv: kv[1], reverse=True)]
        am_p_estimated = (
            float(argmax_prob_field)
            if isinstance(argmax_prob_field, (int, float)) and argmax_prob_field > 0
            else (sorted_probs_desc[0] if sorted_probs_desc else float("nan"))
        )
        per_token_argmax_prob.append(am_p_estimated)

        if not _is_whitespace_only_token(token_text_repr):
            whitespace_excluded_per_token_h_nats.append(h_t)

    n_gen_tokens = len(per_token_h_nats)
    if n_gen_tokens == 0:
        nan_v = float("nan")
        return {
            "scenario_id": sid,
            "bucket": bucket_label,
            "delta_value_orig": _fmt_int_safe(delta_orig),
            "domain": domain,
            "method": method_tag,
            "truncated": truncated_flag,
            "n_tokens_generated": 0,
            "H_token_mean_full_response": nan_v,
            "H_token_std_full_response": nan_v,
            "H_first_k_mean_at_K_eq_64": nan_v,
            "H_normalized_token_mean": nan_v,
            "p_argmax_mean": nan_v,
            "whitespace_excluded_H_token_mean": nan_v,
            "prompt_hash": prompt_hash_digest,
            "_finish_reason": finish_reason_raw or "",
            "_error_message_excerpt": "",
        }

    mean_full_resp = _mean(per_token_h_nats)
    std_full_resp = _stddev(per_token_h_nats, ddof=1)

    window_slice = per_token_h_nats[:first_k_window_size]
    mean_first_k_window = (_mean(window_slice)) if window_slice else float("nan")

    # Normalization uses effective-K-per-position averaged approach giving each position equal weight.
    norm_factors = [
        math.log(max(k_eft, 2))
        for k_eft in per_token_k_eff
    ]
    weighted_normalised_terms = [
        h_t / nf if nf > 0 else 0.0
        for h_t, nf in zip(per_token_h_nats, norm_factors)
    ]
    mean_normalised_token = (
        _mean(weighted_normalised_terms) if weighted_normalised_terms else float("nan")
    )

    p_argmax_avg = _mean(per_token_argmax_prob) if per_token_argmax_prob else float("nan")

    whitespace_excluded_mean = (
        _mean(whitespace_excluded_per_token_h_nats)
        if whitespace_excluded_per_token_h_nats else float("nan")
    )

    return {
        "scenario_id": sid,
        "bucket": bucket_label,
        "delta_value_orig": _fmt_int_safe(delta_orig),
        "domain": domain,
        "method": method_tag,
        "truncated": truncated_flag,
        "n_tokens_generated": n_gen_tokens,
        "H_token_mean_full_response": round(mean_full_resp, 6),
        "H_token_std_full_response": round(std_full_resp, 6),
        "H_first_k_mean_at_K_eq_64": round(mean_first_k_window, 6),
        "H_normalized_token_mean": round(mean_normalised_token, 6),
        "p_argmax_mean": round(p_argmax_avg, 6),
        "whitespace_excluded_H_token_mean": round(whitespace_excluded_mean, 6),
        "prompt_hash": prompt_hash_digest,
        "_finish_reason": finish_reason_raw or "",
        "_error_message_excerpt": "",
    }


def _fmt_int_safe(v: Any) -> str | int:
    """Render delta-value column as integer when possible else as string repr."""
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v.is_integer():
            return int(v)
        return f"{v:.4f}"
    return str(v)


def _mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _stddev(xs: list[float], *, ddof: int = 1) -> float:
    if len(xs) <= ddof:
        return float("nan")
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - ddof)
    return math.sqrt(var)


# --------------------------------------------------------------------------- #
# Driver reading JSONL writing CSV
# --------------------------------------------------------------------------- #
CSV_COLUMN_ORDER_PRIMARY: tuple[str, ...] = (
    "scenario_id",
    "bucket",
    "delta_value_orig",
    "domain",
    "method",
    "truncated",
    "n_tokens_generated",
    "H_token_mean_full_response",
    "H_token_std_full_response",
    "H_first_k_mean_at_K_eq_64",
    "H_normalized_token_mean",
    "p_argmax_mean",
    "whitespace_excluded_H_token_mean",
    "prompt_hash",
    "_finish_reason",
    "_error_message_excerpt",
)


def run(raw_responses_path: str, output_csv_path: str, *,
        cfg_K_floor_for_norm: int = 20,
        first_k_window_size: int = 64) -> dict[str, Any]:
    rows_written_count = 0
    skipped_no_logprobs = 0
    skipped_error_rows = 0
    seen_buckets_counter: dict[str, int] = {}

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    csv_tmp_path = output_csv_path + ".tmp"
    fout_csv = open(csv_tmp_path, "w", encoding="utf-8", newline="")
    wcsv = csv.DictWriter(fout_csv, fieldnames=CSV_COLUMN_ORDER_PRIMARY, extrasaction="ignore")
    wcsv.writeheader()

    with open(raw_responses_path, "r", encoding="utf-8") as fin_jsonl:
        for line_idx, line in enumerate(fin_jsonl):
            line = line.strip()
            if not line:
                continue
            try:
                rec_obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skip malformed %s:%d (%s)", raw_responses_path, line_idx + 1, exc)
                continue

            row_dict = compute_metrics_for_record(
                rec_obj,
                cfg_K_floor_for_norm=cfg_K_floor_for_norm,
                first_k_window_size=first_k_window_size,
            )
            if rec_obj.get("tokens"):
                pass
            elif rec_obj.get("error_message"):
                skipped_error_rows += 1
            else:
                skipped_no_logprobs += 1

            bk_str = str(row_dict.get("bucket") or "?")
            seen_buckets_counter[bk_str] = seen_buckets_counter.get(bk_str, 0) + 1
            wcsv.writerow({k: row_dict.get(k, "") for k in CSV_COLUMN_ORDER_PRIMARY})
            rows_written_count += 1

    fout_csv.close()
    os.replace(csv_tmp_path, output_csv_path)

    logger.info("wrote entropy_per_scenario.csv=%s ; rows=%d ; skips(no_lp,err)=(%d,%d)",
                output_csv_path, rows_written_count, skipped_no_logprobs, skipped_error_rows)

    print("\n=== ENTROPY COMPUTATION SUMMARY ===")
    print(f"output_csv={output_csv_path}")
    print(f"rows_written={rows_written_count} ; skipped(no_logprobs_returned={skipped_no_logprobs}, "
          f"explicit_error_rows={skipped_error_rows})")
    sbk = sorted(seen_buckets_counter.items(), key=lambda kv: kv[0])
    print("per-bucket rows:", ", ".join(f"{b}:{c}" for b, c in sbk))

    return {
        "output_csv_absolute": os.path.abspath(output_csv_path),
        "rows_written": rows_written_count,
        "skipped_breakdown": {"no_logprobs_returned": skipped_no_logprobs,
                              "explicit_error_rows": skipped_error_rows},
        "per_bucket_counts": seen_buckets_counter,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m preliminary.entropy")
    ap.add_argument("--raw-responses-jsonl-path", required=True,
                    help="Path to rounds/_preliminary/<ts>/raw_responses.jsonl produced by preliminary.harness.")
    ap.add_argument("--out-csv-path", required=True,
                    help="Path where entropy_per_scenario.csv should be written.")
    ap.add_argument("--k-floor-for-norm-flooring", type=int, default=20)
    ap.add_argument("--first-k-window-size", type=int, default=64)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s")
    result_summary = run(args.raw_responses_jsonl_path, args.out_csv_path,
                         cfg_K_floor_for_norm=args.k_floor_for_norm_flooring,
                         first_k_window_size=args.first_k_window_size)
    del result_summary  # stdout already printed inside `run`
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
