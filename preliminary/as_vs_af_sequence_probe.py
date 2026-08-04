"""Sequence-level AS-vs-AF probe — feeds *each rollout's own recorded thought*
as continuation past the shared scaffolded prefix, then extracts per-position
softmax signals along the entire decoded span.

Why this fixes the design flaw of ``as_vs_af_mechanism_probe.py``:
-------------------------------------------------------------------
The old probe ran ONE forward pass over ``[chat_prefix + scaffold]`` only,
producing the predicted next-token distribution AT POSITION 0 inside the
thinking frame. Because success-rollouts and fail-rollouts receive LITERALLY
IDENTICAL prefixes (only ``injected_obs`` is fed into BOTH buckets while
outcome divergence happens later during temperature-sampled generation),
the greedy argmax / softmax at pos-0 collapses bit-for-bit equal between
buckets -- making any direct AS↔AF comparison trivially degenerate by
construction.

The fix: append EACH ROLLOUT'S OWN RECORDED-thought-text tokens after the
scaffold (' {"thought": "') before doing the forward pass. Now every
rollout has a UNIQUE conditioning context starting from position-T_scaffold
onward, so the conditional distributions p(y_t | prefix, y_<t) DO depend
on the specific sampled trajectory the agent produced -- giving us real
signal-carriers for distinguishing attack-success outcomes from attack-fail
outcomes beyond just immediate post-exposure surprise.

Outputs land in <output_dir>/seq_probe/
    feats.jsonl         : one record per rollout with per-position scalars
                          (entropy/top1/refuse_mass/nll_actual_yt arrays).
    topk_dists.npz      : stacked arrays shape=(N, T_max, TOP_K) probs(fp16)
                          + indices(int32); rows shorter than T_max are
                          right-padded with zeros and have ``valid_len``
                          field set in feats.jsonl/index.csv.
    hstates_lastpos.npy : memmap array shape=(N, 29, 3584) capturing hidden
                          states ONLY at final position-of-recorded-segment;
                          useful auxiliary signal not central to plan A/B.
    index.csv           : mapping global_idx -> scenario_id/bucket/task/etc +
                          valid_len_per_row.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np

from preliminary.alertness_v3_harness import _build_context_messages


logger = logging.getLogger("preliminary.as_vs_af_sequence_probe")

EVALUATOR_VERSION = "as_vs_af_seq_probe_v1"
TOP_K = 256               # truncate vocab snapshot to top-K per position
DEFAULT_MAX_THOUGHT_TOKENS = 128   # uniform cap on number of decoded positions kept
                                 # Tier-1 raise: was hardcoded 64 in v0 which clipped
                                 # late-band [48..128) signal-carrier region entirely.
MAX_INPUT_LEN = 32000     # left-truncate if longer

REFUSE_SEED_WORDS_ENGLISH = [
    "Hold", "However", "Wait", "Stop", "Suspicious", "Careful", "Cautious",
    "Actually", "Note", "Notice", "Hang", "Pause", "Reconsider",
    "Strange", "Odd", "Unusual", "Hmm", "But", "Although", "Despite",
]


def _build_refusal_marker_ids(tok) -> dict[int, str]:
    out: dict[int, str] = {}
    for w in REFUSE_SEED_WORDS_ENGLISH:
        for variant_prefix in ("", " ", '"', '" ', "'", "' "):
            text = variant_prefix + w
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                out[ids[0]] = text
            elif len(ids) == 2 and variant_prefix:
                out[ids[1]] = text
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", default="/ssd1/models/qwen2.5-7b-it")
    ap.add_argument("--cuda-visible-devices", default="0")
    ap.add_argument("--limit-scenarios", type=int, default=None)
    ap.add_argument(
        "--max-thought-tokens",
        type=int,
        default=DEFAULT_MAX_THOUGHT_TOKENS,
        help=(
            "Uniform cap on number of decoded thought-token positions kept per "
            "rollout (Tier-1 raise: 128 by default; v0 hard-coded 64 which "
            "clipped the late-band [48..128) signal-carrier region entirely)."
        ),
    )
    ap.add_argument("--pre-reserve-gb", type=float, default=0.0,
                    help="If >0, allocate a placeholder tensor of this size on the "
                         "target GPU BEFORE loading model weights. Useful to mark "
                         "territory in highly contended environments so competing "
                         "jobs skip this card while we spin up.")
    args = ap.parse_args()

    MAX_THOUGHT_TOKENS = int(args.max_thought_tokens)

    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s][%(name)s] %(message)s")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Pre-reserve territory if requested — must happen BEFORE model load.
    _reserve_tensor = None
    if args.pre_reserve_gb > 0:
        n_elems = int(args.pre_reserve_gb * (1024**3) / 2)
        try:
            _reserve_tensor = torch.empty(n_elems, dtype=torch.float16, device="cuda:0")
            logger.info("pre-reserved %.2f GiB on cuda:0 (physical=%s)",
                        args.pre_reserve_gb, args.cuda_visible_devices)
        except RuntimeError as e:
            logger.warning("pre-reserve failed (%s); continuing without", str(e)[:120])

    scenarios_path = Path(args.scenarios_json).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading %d scenarios from %s", "?", scenarios_path)
    scenarios: list[dict[str, Any]] = json.loads(scenarios_path.read_text())
    if args.limit_scenarios:
        scenarios = scenarios[:args.limit_scenarios]

    # Build execution plan identical to v3 harness grouping convention.
    plan: list[tuple[Any, ...]] = []   # tuples: (global_idx, scen_i, kind_label, rinfo_dict_or_None_for_clean_in_attacked_list_positioning, obs_override_str)
    for si, s in enumerate(scenarios):
        cr = s.get("clean_rollout") or {}
        ar_list = s.get("attacked_rollouts") or []
        plan.append((len(plan), si, "clean", cr, s.get("clean_obs", "")))
        for ai, ar in enumerate(ar_list):
            bucket = ar.get("bucket", "")
            plan.append((len(plan), si, bucket, ar, s.get("injected_obs", "")))

    N_ROLLOUTS = len(plan)
    assert N_ROLLOUTS > 0
    logger.info("total rollouts queued: %d (= %.0f avg/scenario)",
                N_ROLLOUTS, N_ROLLOUTS/max(len(scenarios), 1))

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    ref_markers_map = _build_refusal_marker_ids(tok)
    ref_marker_id_set = set(ref_markers_map.keys())
    logger.info("collected %d unique refusal-marker token ids",
                len(ref_markers_map))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    device = next(model.parameters()).device

    SCAFFOLD_STR = ' {"thought": "'
    scaffold_ids_cpu = tok.encode(SCAFFOLD_STR, add_special_tokens=False)

    # Allocate persistent storage.
    feats_jsonl_p = out_dir / "feats.jsonl"
    topk_npz_p = out_dir / "topk_dists.npz"
    hstates_mm_p = out_dir / "hstates_lastpos.npy"
    index_csv_p = out_dir / "index.csv"

    LAYERS_PLUS_EMBED = 29
    HIDDEN_DIM = 3584
    hstates_mm = np.memmap(str(hstates_mm_p), dtype=np.float16,
                           mode="w+", shape=(N_ROLLOUTS, LAYERS_PLUS_EMBED, HIDDEN_DIM))
    topk_probs_arr = np.zeros((N_ROLLOUTS, MAX_THOUGHT_TOKENS, TOP_K), dtype=np.float16)
    topk_indices_arr = np.zeros((N_ROLLOUTS, MAX_THOUGHT_TOKENS, TOP_K), dtype=np.int32)

    idx_rows: list[dict[str, Any]] = []

    eps = 1e-12
    t_start = time.time()
    n_done = 0
    errors = []

    with open(feats_jsonl_p, "w", encoding="utf-8") as fj:
        for global_idx, scen_i, kind_label, rinfo, obs_override in plan:
            try:
                scn = scenarios[scen_i]
                msgs = _build_context_messages(
                    scn["system_prompt"],
                    scn["serialized_actions"],
                    scn["target_turn"],
                    obs_override,
                )
                prefix_ids_base = tok.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt").to(device)
                scaffold_tensor = torch.tensor([scaffold_ids_cpu], dtype=torch.long, device=device)
                prefix_full = torch.cat([prefix_ids_base, scaffold_tensor], dim=1)

                rec_thought_text = (rinfo.get("thought") or "").strip()
                if not rec_thought_text:
                    raise RuntimeError(f"empty recorded thought for gidx={global_idx} "
                                       f"task={scn.get('task_id')} kind={kind_label}")
                thought_ids = tok.encode(rec_thought_text, add_special_tokens=False)[:MAX_THOUGHT_TOKENS]
                if not thought_ids:
                    raise RuntimeError(f"encoded empty thought-token list gidx={global_idx}")

                thought_tensor = torch.tensor([thought_ids], dtype=torch.long, device=device)
                full_input = torch.cat([prefix_full, thought_tensor], dim=1)

                overflow = full_input.shape[1] - MAX_INPUT_LEN
                if overflow > 0:
                    # Left-truncate preserving recent history incl injected span critical region.
                    full_input = full_input[:, overflow:]
                prefix_offset_after_truncation = full_input.shape[1] - len(thought_ids)
                valid_len = min(len(thought_ids), MAX_THOUGHT_TOKENS)

                with torch.no_grad():
                    out = model(input_ids=full_input,
                                output_hidden_states=False, use_cache=False)
                # We want predictions FOR positions corresponding to each observed
                # thought token. Standard causal LM gives logits[t] := predict-next-after-pos-t.
                # So prediction for thought-id at relative-offset k corresponds to logit-row
                # at absolute position (prefix_offset_after_truncation - 1 + k). Equivalently,
                # slice logits[prefix_offset-1 .. prefix_offset-1+valid_len].
                start_logit_idx = prefix_offset_after_truncation - 1
                end_logit_idx = start_logit_idx + valid_len
                logits_segment_bf16 = out.logits[0, start_logit_idx:end_logit_idx].float()  # [T_valid,V]
                del out

                probs_seg = torch.softmax(logits_segment_bf16, dim=-1)        # [T_valid,V]
                top_probs_t, top_indices_t = torch.topk(probs_seg, k=min(TOP_K, probs_seg.shape[-1]),
                                                       dim=-1)
                top_probs_np = top_probs_t.cpu().numpy().astype(np.float16)
                top_indices_np = top_indices_t.cpu().numpy().astype(np.int32)

                entropies = -(probs_seg * torch.log(probs_seg + eps)).sum(dim=-1).cpu().numpy().astype(np.float32)
                top1_vals = top_probs_np[..., 0].astype(np.float32)
                refuse_masses = sum(float(probs_seg[i, mid].item()) for i in range(valid_len)
                                    for mid in [])  # placeholder slow path replaced below
                # vectorized refusal-mass computation
                if ref_marker_id_set:
                    mark_idx_vec = sorted(ref_marker_id_set)
                    refuse_masses = (
                        probs_seg.index_select(-1, torch.tensor(mark_idx_vec, device=device))
                        .sum(dim=-1).cpu().numpy().astype(np.float32))
                else:
                    refuse_masses = np.zeros(valid_len, dtype=np.float32)

                # NLL of recorded token id at each offset == self-information of actual_yt under model.
                diag_logits_at_target = probs_seg.gather(
                    -1, torch.tensor(thought_ids[:valid_len], device=device).unsqueeze(-1)).squeeze(-1)
                nll_actual_yt = (-torch.log(diag_logits_at_target + eps)).cpu().numpy().astype(np.float32)

                feat_record = {
                    "_gidx": global_idx,
                    "scenario_index": scen_i,
                    "task_id": scn["task_id"],
                    "domain": scn.get("domain"),
                    "target_tool": scn.get("target_tool"),
                    "mal_tool": scn.get("mal_tool"),
                    "kind": kind_label,
                    # Tier-2 deconfounder: carry through sampling-T stratum tag
                    # if present on the source rollout record.
                    "collection_temperature":
                        rinfo.get("collection_temperature") if isinstance(rinfo, dict) else None,
                    "valid_len": int(valid_len),
                    "seq_entropy": entropies.tolist(),
                    "seq_top1": top1_vals.tolist(),
                    "seq_refuse_mass": refuse_masses.tolist(),
                    "seq_nll": nll_actual_yt.tolist(),
                    "evaluator_version": EVALUATOR_VERSION,
                }
                fj.write(json.dumps(feat_record, ensure_ascii=False) + "\n")
                fj.flush()

                topk_probs_arr[global_idx, :, :] = 0
                topk_indices_arr[global_idx, :, :] = 0
                topk_probs_arr[global_idx, :valid_len] = top_probs_np
                topk_indices_arr[global_idx, :valid_len] = top_indices_np

                idx_rows.append({
                    "global_idx": global_idx,
                    "scenario_index": scen_i,
                    "task_id": scn["task_id"],
                    "domain": scn.get("domain"),
                    "rollout_kind": kind_label,
                    "collection_temperature":
                        rinfo.get("collection_temperature") if isinstance(rinfo, dict) else None,
                    "valid_len": int(valid_len),
                    "has_tool_call": bool(rinfo.get("tool_call_name")),
                    "thought_chars": len(rinfo.get("thought", "")),
                })

                del probs_seg, top_probs_t, top_indices_t, logits_segment_bf16
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()

            except Exception as exc:
                emsg = f"gidx={global_idx} scen={scen_i} kind={kind_label}: {exc!r}"
                logger.exception(emsg[:200])
                errors.append({"gidx": global_idx, "err": repr(exc)[:300]})
                idx_rows.append({
                    "global_idx": global_idx,
                    "scenario_index": scen_i,
                    "task_id": "",
                    "domain": scn.get("domain") if isinstance(scn, dict) else None,
                    "rollout_kind": kind_label,
                    "collection_temperature":
                        rinfo.get("collection_temperature") if isinstance(rinfo, dict) else None,
                    "valid_len": 0,
                    "has_tool_call": False,
                    "thought_chars": 0,
                    "error": repr(exc)[:200],
                })

            n_done += 1
            if n_done % 25 == 0 or n_done == N_ROLLOUTS:
                elapsed = time.time() - t_start
                rate = n_done / max(elapsed, 0.001)
                eta_sec = (N_ROLLOUTS - n_done) / max(rate, 0.001)
                logger.info("[%d/%d] elapsed=%.1fs rate=%.2f/s eta≈%.0fs errors=%d",
                            n_done, N_ROLLOUTS, elapsed, rate, eta_sec, len(errors))
                hstates_mm.flush()

    hstates_mm.flush()
    np.savez_compressed(topk_npz_p,
                        topk_probs=topk_probs_arr.astype(np.float16),
                        topk_indices=topk_indices_arr.astype(np.int32))
    # Union-of-keys ensures rows carrying an ``error`` field are still writable
    all_fieldnames_ordered: list[str] = []
    seen=set()
    for r in idx_rows:
        for k_ in r.keys():
            if k_ not in seen:
                seen.add(k_)
                all_fieldnames_ordered.append(k_)
    with open(index_csv_p, "w", newline="", encoding="utf-8") as fc:
        writer = csv.DictWriter(fc, fieldnames=all_fieldnames_ordered)
        writer.writeheader()
        writer.writerows(idx_rows)

    summary = {
        "completed_records_written": n_done,
        "errors_count": len(errors),
        "first_few_errors_preview": errors[:10],
        "output_files": {
            "features_jsonl": str(feats_jsonl_p),
            "topk_dist_npz": str(topk_npz_p),
            "index_csv": str(index_csv_p),
        },
        "config": {
            "top_k": TOP_K,
            "max_thought_tokens": MAX_THOUGHT_TOKENS,
            "max_input_len": MAX_INPUT_LEN,
        },
        # Tier-1 multi-domain + Tier-2 temperature-sweep coverage summary.
        "domains_breakdown":
            {d: sum(1 for r in idx_rows if r.get("domain") == d)
             for d in sorted({r.get("domain") for r in idx_rows
                              if r.get("domain")})},
        "temperature_strata_seen":
            sorted([t for t in {r.get("collection_temperature") for r in idx_rows
                                if r.get("collection_temperature") is not None}]),
        "elapsed_seconds_total": round(time.time() - t_start, 2),
    }
    (out_dir.parent / "as_vs_af_seq_extraction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("DONE.")


if __name__ == "__main__":
    main()
