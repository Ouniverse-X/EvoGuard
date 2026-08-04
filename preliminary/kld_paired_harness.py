"""Paired KL-divergence surprise experiment (de-inject design).

For each of the 121 workspace segments whose last observation carries an
``<INFORMATION>`` injection, create a clean counterpart by stripping the
injection block.  Three forward passes per segment:

  P_pre        : last obs replaced by neutral "Done."
  P_post_clean : last obs with <INFORMATION> block stripped (de-injected)
  P_post_inj   : original last obs (with injection)

KL_clean    = D_KL(P_post_clean   || P_pre)
KL_injected = D_KL(P_post_inj     || P_pre)
ΔKL         = KL_injected − KL_clean

The injected generation is then classified (heuristic → LLM fallback) into
AttackSuccess / AttackFail.

See ``docs/superpowers/specs/2026-08-02-preliminary-ipi-tool-return-kld-surprise-design.md``
for the original three-condition design; this module implements the paired
variant discussed 2026-08-02.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from typing import Any, Optional

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from preliminary.base_defender_eval import extract_structured, EVALUATOR_VERSION
from preliminary.kld_measurement import (
    build_messages_for_toolsafe_segment,
    compute_kl_from_logits,
    replace_last_observation,
    split_last_observation,
    NEUTRAL_DEFAULT_OBSERVATION,
)
from preliminary.kld_classifier import (
    classify_action,
    extract_injection_body,
    has_injection,
    BUCKET_ATTACK_SUCCESS,
    BUCKET_ATTACK_FAIL,
)
from preliminary.kld_harness import VLLMJudgeClient

logger = logging.getLogger("preliminary.kld_paired_harness")

CSV_COLUMN_ORDER: tuple[str, ...] = (
    "trajectory_id", "segment_id", "bucket",
    "kl_clean", "kl_injected", "delta_kl",
    "obs_token_len_clean", "obs_token_len_injected",
    "parse_status", "heuristic_rule", "judge_used", "complies", "dropped",
    "injection_body_first200", "generated_action_first200",
    "evaluator_version",
)

_INFO_RE = re.compile(r"<INFORMATION>.*?</INFORMATION>", re.DOTALL | re.IGNORECASE)


def strip_injection(text: str) -> str:
    """Remove all ``<INFORMATION>...</INFORMATION>`` blocks from *text*.

    Returns the surrounding text with the injection excised.  Does NOT
    collapse whitespace — the leftover double-space is a minor artifact that
    does not materially affect tokenisation.
    """
    return _INFO_RE.sub("", text or "")


def build_paired_messages(
    record: dict[str, Any],
    *,
    neutral_default: str = NEUTRAL_DEFAULT_OBSERVATION,
    domain: str = "workspace",
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Return ``(msgs_pre, msgs_post_clean, msgs_post_injected)``.

    All three share the same system prompt (built from env_info tools).
    ``msgs_pre`` has the last observation replaced by *neutral_default*.
    ``msgs_post_clean`` has the ``<INFORMATION>`` block stripped from the
    last observation but keeps the legitimate tool-return content.
    ``msgs_post_injected`` is the verbatim history (injection present).
    """
    # Reuse the existing builder to get (msgs_post_injected, msgs_pre).
    msgs_post_injected, msgs_pre = build_messages_for_toolsafe_segment(
        record, neutral_default=neutral_default, domain=domain,
    )
    system_str = msgs_pre[0]["content"]

    # Build the clean user content: strip injection from last observation.
    history = str(record.get("history", ""))
    prefix, obs = split_last_observation(history)
    stripped_obs = strip_injection(obs)
    clean_history = replace_last_observation(history, stripped_obs)
    user_clean = clean_history + "\nDecide your next step."

    msgs_post_clean = [
        {"role": "system", "content": system_str},
        {"role": "user", "content": user_clean},
    ]
    return msgs_pre, msgs_post_clean, msgs_post_injected


def run(
    *,
    data_path: str,
    output_dir: str,
    model_path: str = "/ssd1/models/qwen2.5-7b-it",
    cuda_visible_devices: str = "2",
    judge_base_url: str = "http://localhost:8002/v1",
    judge_model: str = "llama3-8b-judge",
    neutral_default: str = NEUTRAL_DEFAULT_OBSERVATION,
    max_input_length: int = 32000,
    max_new_tokens: int = 256,
    eps: float = 1e-12,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the paired KL surprise experiment. Returns summary dict."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "paired_kl.csv")
    jsonl_path = os.path.join(output_dir, "raw_paired_kl.jsonl")
    log_path = os.path.join(output_dir, "run.log")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    records = json.load(open(data_path, "r", encoding="utf-8"))
    logger.info("loaded %d records from %s", len(records), data_path)

    # Filter to injected segments only.
    injected_segments = []
    for rec in records:
        history = str(rec.get("history", ""))
        if "Observation:" not in history:
            continue
        _, obs = split_last_observation(history)
        if has_injection(obs):
            injected_segments.append(rec)
    logger.info("found %d injected segments", len(injected_segments))

    # Load model (skip in dry-run).
    model = tok = device = None
    if not dry_run:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("loading tokenizer from %s", model_path)
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        logger.info("loading model %s onto cuda bf16", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to("cuda")
        model.eval()
        device = "cuda"

    judge_client = None
    if not dry_run:
        try:
            judge_client = VLLMJudgeClient(judge_base_url, judge_model)
            logger.info("judge client ready at %s model=%s", judge_base_url, judge_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge client init failed (%s); AMBIGUOUS cases will be dropped", exc)

    n_processed = n_dropped = 0
    bucket_counts: dict[str, int] = {}
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv, \
         open(jsonl_path, "w", encoding="utf-8") as fjsonl:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLUMN_ORDER)
        writer.writeheader()
        for rec in injected_segments:
            traj_id = rec.get("id-interaction", "?")
            seg_id = rec.get("id-segment", "?")
            history = str(rec.get("history", ""))
            instruction = str(rec.get("instruction", ""))
            _, obs_content = split_last_observation(history)
            obs_content_stripped = obs_content.strip()
            inj_body = extract_injection_body(obs_content_stripped)

            row: dict[str, Any] = {
                "trajectory_id": traj_id, "segment_id": seg_id,
                "bucket": "", "kl_clean": "", "kl_injected": "", "delta_kl": "",
                "obs_token_len_clean": "", "obs_token_len_injected": "",
                "parse_status": "", "heuristic_rule": "",
                "judge_used": False, "complies": "", "dropped": False,
                "injection_body_first200": "", "generated_action_first200": "",
                "evaluator_version": EVALUATOR_VERSION,
            }

            try:
                msgs_pre, msgs_post_clean, msgs_post_inj = build_paired_messages(
                    rec, neutral_default=neutral_default,
                )

                if dry_run:
                    kl_clean = 0.001
                    kl_injected = 0.005
                    obs_tok_clean = 10
                    obs_tok_inj = 20
                    generated_text = '{"thought":"dryrun","tool":null,"args":{}}'
                    parsed = extract_structured(generated_text)
                else:
                    import torch
                    ids_pre = tok.apply_chat_template(
                        msgs_pre, tokenize=True, add_generation_prompt=True,
                        return_tensors="pt").to(device)
                    ids_clean = tok.apply_chat_template(
                        msgs_post_clean, tokenize=True, add_generation_prompt=True,
                        return_tensors="pt").to(device)
                    ids_inj = tok.apply_chat_template(
                        msgs_post_inj, tokenize=True, add_generation_prompt=True,
                        return_tensors="pt").to(device)

                    # Truncate to max_input_length (keep most recent tokens).
                    if ids_pre.shape[1] > max_input_length:
                        ids_pre = ids_pre[:, -max_input_length:]
                    if ids_clean.shape[1] > max_input_length:
                        ids_clean = ids_clean[:, -max_input_length:]
                    if ids_inj.shape[1] > max_input_length:
                        ids_inj = ids_inj[:, -max_input_length:]

                    obs_tok_clean = int(ids_clean.shape[1]) - int(ids_pre.shape[1])
                    obs_tok_inj = int(ids_inj.shape[1]) - int(ids_pre.shape[1])

                    with torch.no_grad():
                        out_pre = model(input_ids=ids_pre, use_cache=False)
                        out_clean = model(input_ids=ids_clean, use_cache=False)
                        out_inj = model(input_ids=ids_inj, use_cache=False)

                    logits_pre = out_pre.logits[0, -1, :]
                    logits_clean = out_clean.logits[0, -1, :]
                    logits_inj = out_inj.logits[0, -1, :]

                    kl_clean = compute_kl_from_logits(logits_clean, logits_pre, eps=eps)
                    kl_injected = compute_kl_from_logits(logits_inj, logits_pre, eps=eps)

                    # Generate next action from P_post_injected.
                    with torch.no_grad():
                        gen_ids = model.generate(
                            input_ids=ids_inj, do_sample=False,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=tok.eos_token_id,
                        )
                    new_tokens = gen_ids[0, ids_inj.shape[1]:]
                    generated_text = tok.decode(new_tokens, skip_special_tokens=True)
                    parsed = extract_structured(generated_text)

                delta_kl = kl_injected - kl_clean

                row["kl_clean"] = f"{kl_clean:.6f}"
                row["kl_injected"] = f"{kl_injected:.6f}"
                row["delta_kl"] = f"{delta_kl:.6f}"
                row["obs_token_len_clean"] = obs_tok_clean
                row["obs_token_len_injected"] = obs_tok_inj
                row["parse_status"] = parsed.get("_parse_status", "?")
                row["generated_action_first200"] = generated_text[:200].replace("\n", " ")
                row["injection_body_first200"] = inj_body[:200].replace("\n", " ")

                # Classify the injected generation.
                res = classify_action(
                    parsed, generated_text, inj_body, instruction,
                    judge_client=judge_client,
                )
                row["bucket"] = res["bucket"] or "DROPPED"
                row["heuristic_rule"] = res["heuristic_rule"]
                row["judge_used"] = res["judge_used"]
                row["complies"] = "" if res["complies"] is None else res["complies"]
                row["dropped"] = res["dropped"]
                if res["dropped"]:
                    n_dropped += 1

                if not row["dropped"]:
                    bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1
                writer.writerow(row)
                fjsonl.write(json.dumps({
                    **row,
                    "_obs_content_first200": obs_content_stripped[:200],
                    "_generated_action_full": generated_text,
                    "_injection_body_full": inj_body,
                }, ensure_ascii=False) + "\n")
                fcsv.flush(); fjsonl.flush()
                n_processed += 1
                if n_processed % 10 == 0:
                    logger.info("processed %d/%d segments ; buckets=%s ; dropped=%d",
                                n_processed, len(injected_segments), bucket_counts, n_dropped)
            except Exception as exc:  # noqa: BLE001
                n_dropped += 1
                logger.exception("segment traj=%s seg=%s failed: %s", traj_id, seg_id, exc)
                row["bucket"] = "ERROR"; row["dropped"] = True
                writer.writerow(row); fcsv.flush()

    summary = {
        "n_processed": n_processed, "n_dropped": n_dropped,
        "bucket_counts": bucket_counts,
        "csv_path": csv_path, "jsonl_path": jsonl_path,
        "dry_run": dry_run,
    }
    summary_path = os.path.join(output_dir, "run_summary.json")
    json.dump(summary, open(summary_path, "w"), indent=2)
    logger.info("DONE. summary=%s", summary)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/toolsafe/agentdojo-tragj/workspace.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", default="/ssd1/models/qwen2.5-7b-it")
    ap.add_argument("--cuda-visible-devices", default="2")
    ap.add_argument("--judge-base-url", default="http://localhost:8002/v1")
    ap.add_argument("--judge-model", default="llama3-8b-judge")
    ap.add_argument("--neutral-default", default=NEUTRAL_DEFAULT_OBSERVATION)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(
        data_path=args.data_path, output_dir=args.output_dir,
        model_path=args.model_path,
        cuda_visible_devices=args.cuda_visible_devices,
        judge_base_url=args.judge_base_url, judge_model=args.judge_model,
        neutral_default=args.neutral_default,
        max_new_tokens=args.max_new_tokens, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
