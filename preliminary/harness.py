"""Entropy-measurement harness.

Reads scenarios from ``bench/corpus_*.jsonl``, assembles chat messages via
``preliminary.context_builder.build_messages_for_scenario``, calls vLLM's
OpenAI-compatible ``/v1/chat/completions`` endpoint with ``logprobs=True``
and ``top_logprobs=20`` capturing per-token distributions of the agent's
first response after exposure to the poisoned tool observation, and streams
results incrementally to ``raw_responses.jsonl`` for resume-safety.

Dry-run mode (cfg.dry_run=True) bypasses the network entirely returning a
synthetic mock-logprob payload so downstream entropy/stats/plot stages can
be exercised offline without GPU/network consumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional

if __package__ in (None, ""):                                                   # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from preliminary.config import ExperimentConfig
    from preliminary.context_builder import build_messages_for_scenario
else:
    from .config import ExperimentConfig
    from .context_builder import build_messages_for_scenario

logger = logging.getLogger("preliminary.harness")


# --------------------------------------------------------------------------- #
# Scenario loading helpers
# --------------------------------------------------------------------------- #
def iter_scenarios(bench_dir: str) -> Iterator[dict[str, Any]]:
    """Yield every scenario across all corpus_*.jsonl files in deterministic order."""
    if not os.path.isdir(bench_dir):
        return
    files = sorted(f for f in os.listdir(bench_dir) if f.startswith("corpus_") and f.endswith(".jsonl"))
    for fname in files:
        with open(os.path.join(bench_dir, fname), "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except json.JSONDecodeError as exc:
                    logger.warning("skip malformed scenario %s (%s)", fname, exc)


def load_existing_done_ids(out_path: str) -> set[str]:
    """Return set of scenario_ids already present in raw_responses.jsonl (for resume)."""
    done: set[str] = set()
    if not os.path.isfile(out_path):
        return done
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    d = json.loads(ln)
                    sid = d.get("scenario_id")
                    if sid:
                        done.add(str(sid))
                except Exception:                                                  # noqa: BLE001 - tolerate partial writes from prior crash mid-line
                    continue
    except Exception as exc:                                                      # pragma: no cover - defensive IO error path
        logger.warning("failed reading existing raw_responses %s : %s", out_path, exc)
        return done


# --------------------------------------------------------------------------- #
# Network client wrapper with retry/backoff + dry-run mock fallback.
# --------------------------------------------------------------------------- #
class _VLLMClient:
    """Thin wrapper around OpenAI SDK client configured against local vLLM endpoint."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg_model_id = cfg.model.model_id
        self.base_url = cfg.model.base_url
        api_key_val = os.environ.get(cfg.model.api_key_env_var) or "EMPTY"
        self._client = None  # lazy-constructed on first call to keep dry-run network-free at startup time
        self._api_key_value = api_key_val
        self._gen_kwargs_template = {
            "max_tokens": cfg.generation.max_new_tokens,
            "temperature": cfg.generation.temperature,
            "top_p": cfg.generation.top_p,
            "logprobs": True,
            "top_logprobs": int(min(20, max(1, cfg.generation.top_logprobs))),
            "stop": list(cfg.generation.extra_stop_sequences),
            "seed": cfg.generation.seed_passthrough,
        }

    def _ensure_client(self):                                                    # pragma: no cover - exercised only when not dry-run
        if self._client is None:
            try:
                from openai import OpenAI                                        # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError(
                    "`openai` Python SDK is required to run real measurements but is "
                    "not installed. Install it via `pip install openai` or use --dry-run."
                ) from exc
            self._client = OpenAI(base_url=self.base_url, api_key=self._api_key_value)
        return self._client

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Call /v1/chat/completions once. Returns dict-shaped response record."""
        cli = self._ensure_client()
        last_exc: Optional[Exception] = None
        backoff_s = 1.0
        for attempt_idx in range(3):
            t0 = time.monotonic()
            try:
                resp = cli.chat.completions.create(
                    model=self.cfg_model_id,
                    messages=messages,
                    **self._gen_kwargs_template,
                )
                latency_ms = int((time.monotonic() - t0) * 1000)
                resp_dict = resp.model_dump()
                resp_dict["_latency_ms"] = latency_ms
                return _normalize_chat_completion_response(resp_dict)
            except Exception as exc:                                              # noqa: BLE001 - retry any transient server-side failure
                last_exc = exc
                logger.warning("vLLM request failed attempt=%d err=%s ; sleeping %.2fs",
                               attempt_idx + 1, type(exc).__name__, backoff_s)
                time.sleep(backoff_s)
                backoff_s *= 2.0
                continue
        # All retries exhausted; surface structured-failure payload instead of raising.
        raise RuntimeError(f"vLLM unreachable after retries; last_err={last_exc}")


def _normalize_chat_completion_response(resp_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten OpenAI-style response into our canonical token-list shape used downstream."""

    choices = resp_dict.get("choices") or []
    choice0 = choices[0] if isinstance(choices, list) else {}
    msg_obj = choice0.get("message") or {}
    text_out = msg_obj.get("content") or ""
    finish_reason_raw = choice0.get("finish_reason")
    logp_payload = choice0.get("logprobs")

    tokens_meta_list: list[dict[str, Any]] = []
    truncated_flag = False

    content_logprob_entries = None
    if isinstance(logp_payload, dict):
        # Chat-completions logprobs structure is {content:[{token,bytes,logprob,top_logprobs:[...]}, ...], ...}
        clist = logp_payload.get("content")
        if isinstance(clist, list):
            content_logprob_entries = clist

    n_gen_tokens_reported_by_server = len(content_logprob_entries or [])

    if content_logprob_entries:
        for entry in content_logprob_entries:
            tok_text_repr = entry.get("token") if entry.get("token") else ""
            tok_bytes = entry.get("bytes") or []
            decoded_str = "".join(chr(b) if isinstance(b, int) else "" for b in tok_bytes) \
                          if tok_bytes else (tok_text_repr or "")
            lp_main = float(entry.get("logprob")) if entry.get("logprob") is not None else None
            topk_arr = entry.get("top_logprobs") or []
            topk_normed: list[dict[str, Any]] = []
            for tk in topk_arr[:20]:                                              # cap K explicitly even though we requested top_K=20
                if not isinstance(tk, dict):
                    continue
                ttok_repr = tk.get("token") or ""
                tb_bts = tk.get("bytes") or []
                ttok_decoded = "".join(chr(b) if isinstance(b, int) else "" for b in tb_bts) \
                                if tb_bts else (ttok_repr or "")
                tl_lp = tk.get("logprob")
                tl_prob = math.exp(float(tl_lp)) if tl_lp is not None else None
                topk_normed.append({
                    "token_str": ttok_decoded,
                    "token_id": tk.get("token_id"),
                    "logprob": float(tl_lp) if tl_lp is not None else None,
                    "prob": float(tl_prob) if tl_prob is not None else None,
                })
            prob_main_argmax_match = (
                next((x["prob"] for x in topk_normed if x["token_str"] == decoded_str), None)
                if topk_normed else None
            )
            argmax_prob_estimate = (
                prob_main_argmax_match
                if prob_main_argmax_match is not None
                else (math.exp(lp_main) if lp_main is not None else None)
            )
            tokens_meta_list.append({
                "text": decoded_str,
                "token_id": entry.get("token_id"),
                "argmax_prob": argmax_prob_estimate,
                "logprobs_topk": topk_normed,
            })

    if finish_reason_raw == "length":
        truncated_flag = True
    elif finish_reason_raw == "stop" or finish_reason_raw is None:
        pass   # natural stop or unknown — treat as non-truncated by default
    elif finish_reason_raw == "model_length" or finish_reason_raw == "max_tokens":
        truncated_flag = True

    return {
        "generated_text": text_out,
        "truncated": bool(truncated_flag),
        "n_generated_tokens_server_count": n_gen_tokens_reported_by_server,
        "tokens": tokens_meta_list,
        "_raw_finish_reason": finish_reason_raw,
    }


# --------------------------------------------------------------------------- #
# Dry-run synthetic-response generator producing plausible-but-uniform distribution.
# --------------------------------------------------------------------------- #
_DRYRUN_VOCAB_CHARS = [c for c in
                       "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:'\"!?-_()/"]


def _synthetic_response_for_dryrun(scenario: dict[str, Any],
                                   rng_seed_offset_for_bucket: int | None = None) -> dict[str, Any]:
    """Produce a fixed-shape fake completion exercising all downstream code paths without GPU/network.

    Token-level probability mass follows a Dirichlet-like construction that deliberately DOES NOT vary
    meaningfully across buckets → null result expected during smoke validation confirming plumbing works;
    a separate optional bucket-aware variant can be enabled later to sanity-test trend detection logic too.
    """

    rng = random.Random(scenario.get("_dry_run_rng_basis_int", 12345))
    target_n_tokens = 32
    generated_chars: list[str] = []
    token_records: list[dict[str, Any]] = []
    chosen_masses_per_token: list[float] = []
    while len(token_records) < target_n_tokens:
        cidx = rng.randrange(len(_DRYRUN_VOCAB_CHARS))
        ch = _DRYRUN_VOCAB_CHARS[cidx]
        generated_chars.append(ch)

        # Build uniform-ish distribution over first ~10 vocab candidates plus heavier weight on actual pick.
        k_alt = min(8, len(_DRYRUN_VOCAB_CHARS))
        alt_indices_pool = [_DRYRUN_VOCAB_CHARS[i] for i in range(k_alt)]
        weights_unnorm = [(rng.random() * 0.5) + 0.05 for _ in range(k_alt)]  # noise around small base
        main_pick_weight = rng.uniform(0.6, 0.95)
        total_w_after_pick_boost = sum(weights_unnorm) + main_pick_weight
        probs_topk: list[tuple[str, float]] = [
            (_DRYRUN_VOCAB_CHARS[i], w / total_w_after_pick_boost)
            for i, w in enumerate(weights_unnorm)
        ]
        probs_topk.sort(key=lambda kv: kv[1], reverse=True)
        probs_topk.insert(0, (ch, main_pick_weight / total_w_after_pick_boost))

        renorm_sum_check = sum(p for _, p in probs_topk)
        if abs(renorm_sum_check - 1.0) > 1e-4:
            scale_fix = 1.0 / renorm_sum_check
            probs_topk = [(s_, p_ * scale_fix) for s_, p_ in probs_topk]

        topk_structured = [{
            "token_str": s_,
            "token_id": i_,
            "logprob": (float(math.log(max(p_, 1e-12))) if p_ > 0 else None),
            "prob": p_,
        } for i_, (s_, p_) in enumerate(probs_topk)]

        chosen_masses_per_token.append(probs_topk[0][1])
        token_records.append({
            "text": ch,
            "token_id": ord(ch),
            "argmax_prob": probs_topk[0][1],
            "logprobs_topk": topk_structured,
        })
        if len(generated_chars) >= target_n_tokens:
            break

    gen_text = "".join(generated_chars)
    avg_h_uniform_baseline = sum(-p_*math.log(p_) for _, p_ in probs_topk if p_>0)/len([p_ for _,p_ in probs_topk])
    del avg_h_uniform_baseline  # placeholder removed intentionally; left here only as documentation hint about expected H magnitude under this generator

    return {
        "generated_text": gen_text,
        "truncated": False,
        "n_generated_tokens_server_count": len(token_records),
        "tokens": token_records,
        "_raw_finish_reason": "stop",
    }


# --------------------------------------------------------------------------- #
# Main orchestrator class
# --------------------------------------------------------------------------- #
class HarnessRunner:
    def __init__(self, cfg: ExperimentConfig, repo_root_override: str | None = None):
        self.cfg = cfg
        self.repo_root = repo_root_override or cfg.resolve_repo_root()
        self.bench_root = os.path.join(self.repo_root, cfg.mining.bench_output_dir_relative_to_repo)

    @property
    def output_dir(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
        od_rel = self.cfg.output_dir_template.format(run_timestamp=ts)
        return os.path.join(self.repo_root, od_rel.replace("/", os.sep)) if not os.path.isabs(od_rel) else od_rel

    def ensure_output_dir(self) -> str:
        od = self.output_dir
        os.makedirs(od, exist_ok=True)
        return od

    def run(self) -> tuple[str, dict[str, Any]]:
        """Execute harness over bench dataset writing raw_responses.jsonl under output dir.

        Returns ``(output_dir_abs, summary_stats_dict)`` where summary captures per-bucket counts etc.
        """

        od = self.ensure_output_dir()
        out_responses_jsonl = os.path.join(od, "raw_responses.jsonl")
        done_ids = load_existing_done_ids(out_responses_jsonl)
        if done_ids:
            logger.info("resume mode detected: skipping %d already-completed scenario_ids", len(done_ids))

        # Instantiate either real VLLMClient OR use dry-run inline generation path.
        client_real_or_none: Optional[_VLLMClient] = None
        if not self.cfg.dry_run:
            client_real_or_none = _VLLMClient(self.cfg)

        counter_total_seen = 0
        counter_newly_written = 0
        counter_skipped_already_done = 0
        counter_skipped_network_error = 0
        counter_skipped_empty_generation = 0
        bucket_counter_completed: dict[str, int] = {}

        fout = open(out_responses_jsonl, "a", encoding="utf-8")
        try:
            ts_started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for scen in iter_scenarios(self.bench_root):
                sid = scen.get("scenario_id")
                if not sid:
                    continue
                counter_total_seen += 1
                if sid in done_ids:
                    counter_skipped_already_done += 1
                    continue

                messages = build_messages_for_scenario(scen, self.repo_root)
                prompt_hash_digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()

                collected_at_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
                latency_ms_observed: int | None = None

                try:
                    if client_real_or_none is not None:
                        normalized_resp = client_real_or_none.generate(messages)
                        latency_ms_observed = int(normalized_resp.pop("_latency_ms", 0) or 0)
                        endpoint_used_label = self.cfg.model.base_url
                    else:
                        scen_with_dryrngbasis = dict(scen)
                        scen_with_dryrngbasis["_dry_run_rng_basis_int"] = hash(sid) & 0xFFFFFFFF
                        normalized_resp = _synthetic_response_for_dryrun(scen_with_dryrngbasis)
                        endpoint_used_label = "<DRY_RUN_MOCK>"
                except Exception as exc:                                          # noqa: BLE001 - never crash pipeline on single-scenario failures
                    logger.warning("[%s] harness collection FAILED: %s -- recording skip-row", sid, exc)
                    endpoint_used_label = ("ERROR_AFTER_ATTEMPT"
                                            if client_real_or_none else "<DRY_RUN_MOCK>")
                    fail_record = {
                        "scenario_id": sid,
                        "bucket": scen.get("bucket"),
                        "domain": scen.get("domain"),
                        "prompt_hash": prompt_hash_digest,
                        "collected_at_utc": collected_at_iso,
                        "endpoint_used": endpoint_used_label,
                        "generated_text": "",
                        "truncated": False,
                        "finish_reason": "error",
                        "error_message": str(exc)[:500],
                        "tokens": [],
                        "_meta": {"ts_collected_utc": collected_at_iso},
                    }
                    fout.write(json.dumps(fail_record, ensure_ascii=False) + "\n"); fout.flush()
                    counter_skipped_network_error += 1
                    continue

                tokens_meta = normalized_resp.get("tokens") or []
                if not tokens_meta:
                    counter_skipped_empty_generation += 1
                    empty_record = {
                        "scenario_id": sid,
                        "bucket": scen.get("bucket"),
                        "domain": scen.get("domain"),
                        "method": scen.get("method"),
                        "delta_value_orig": scen.get("delta_value_orig"),
                        "goal_instruction_excerpt": str(scen.get("goal_instruction",""))[:200],
                        "task_id": scen.get("task_id"),
                        "injection_target_turn_index": scen.get("injection_target_turn_index"),
                        "original_signals_for_reference": scen.get("original_signals_for_reference"),
                        "source": scen.get("source"),
                        "prompt_hash": prompt_hash_digest,
                        "collected_at_utc": collected_at_iso,
                        "endpoint_used": endpoint_used_label,
                        "generated_text": "",
                        "truncated": False,
                        "finish_reason": normalized_resp.get("_raw_finish_reason"),
                        "error_message": "",
                        "tokens": [],
                        "_meta": {
                            "ts_collected_utc": collected_at_iso,
                            "latency_ms": latency_ms_observed,
                            "endpoint_used": endpoint_used_label,
                        },
                    }
                    fout.write(json.dumps(empty_record, ensure_ascii=False) + "\n"); fout.flush()
                    continue

                rec_to_persist = {
                    "scenario_id": sid,
                    "bucket": scen.get("bucket"),
                    "domain": scen.get("domain"),
                    "method": scen.get("method"),
                    "delta_value_orig": scen.get("delta_value_orig"),
                    "goal_instruction_excerpt": str(scen.get("goal_instruction",""))[:200],
                    "task_id": scen.get("task_id"),
                    "injection_target_turn_index": scen.get("injection_target_turn_index"),
                    "original_signals_for_reference": scen.get("original_signals_for_reference"),
                    "source": scen.get("source"),
                    "prompt_hash": prompt_hash_digest,
                    "collected_at_utc": collected_at_iso,
                    "endpoint_used": endpoint_used_label,
                    "generated_text": normalized_resp["generated_text"],
                    "truncated": normalized_resp["truncated"],
                    "finish_reason": normalized_resp.get("_raw_finish_reason"),
                    "error_message": "",
                    "tokens": tokens_meta,
                    "_meta": {
                        "ts_collected_utc": collected_at_iso,
                        "latency_ms": latency_ms_observed,
                        "endpoint_used": endpoint_used_label,
                    },
                }
                fout.write(json.dumps(rec_to_persist, ensure_ascii=False) + "\n")
                fout.flush()
                counter_newly_written += 1
                bk = str(rec_to_persist.get("bucket") or "?")
                bucket_counter_completed[bk] = bucket_counter_completed.get(bk, 0) + 1

                if counter_newly_written % 25 == 0:
                    logger.info("[progress] newly written=%d skipped(resume/err/empty)=(%d/%d/%d)",
                                counter_newly_written, counter_skipped_already_done,
                                counter_skipped_network_error, counter_skipped_empty_generation)
        finally:
            fout.close()

        ts_finished_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_runmeta_yaml(self.cfg, od, ts_started_iso, ts_finished_iso, {
            "total_seen_in_corpus": counter_total_seen,
            "newly_written_this_session": counter_newly_written,
            "skipped_already_done_resume_safe": counter_skipped_already_done,
            "skipped_network_errors": counter_skipped_network_error,
            "skipped_empty_generations_no_logprobs_returned": counter_skipped_empty_generation,
            "completed_bucket_counts": bucket_counter_completed,
        }, out_responses_jsonl_file=out_responses_jsonl)

        summary = {
            "output_dir_absolute": od,
            "total_seen_in_corpus": counter_total_seen,
            "newly_written_this_session": counter_newly_written,
            "skipped_breakdown": {
                "already_done_resume_safe": counter_skipped_already_done,
                "network_error_retries_exhausted": counter_skipped_network_error,
                "empty_generation_no_logprobs_returned": counter_skipped_empty_generation,
            },
            "completed_bucket_counts": bucket_counter_completed,
            "started_at_utc_iso8601": ts_started_iso,
            "finished_at_utc_iso8601": ts_finished_iso,
        }

        print("\n=== HARNESS SUMMARY ===")
        print(f"output_dir={od}")
        print(f"newly_written={counter_newly_written} ; seen_total={counter_total_seen}")
        print(f"skips: already_done={counter_skipped_already_done}, neterr={counter_skipped_network_error}, "
              f"empty_gen={counter_skipped_empty_generation}")
        if bucket_counter_completed:
            sbk = sorted(bucket_counter_completed.items(), key=lambda kv: kv[0])
            print("per-bucket completions:", ", ".join(f"{b}:{c}" for b, c in sbk))
        return od, summary


def write_runmeta_yaml(cfg: ExperimentConfig, output_dir: str, started_iso: str, finished_iso: str,
                       extra_summary_fields: dict[str, Any], *, out_responses_jsonl_file: str) -> None:
    """Persist reproducibility metadata YAML alongside outputs."""
    mp = os.path.join(output_dir, "run_meta.yaml")
    lines: list[str] = []

    def emit(k: str, v: Any, indent: int = 0) -> None:
        prefix = "  " * indent
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            for kk, vv in sorted(v.items()):
                emit(kk, vv, indent + 1)
        elif isinstance(v, (list, tuple)):
            lines.append(f"{prefix}{k}:")
            for item in v:
                lines.append(f"{prefix}- {item!r}")
        else:
            sv = repr(v) if isinstance(v, str) else repr(v)
            lines.append(f"{prefix}{k}: {sv}")

    git_sha_now = "unknown"

    import subprocess as sp
    try:
        git_sha_now = sp.check_output(["git", "-C", cfg.resolve_repo_root(), "rev-parse", "HEAD"],
                                      stderr=sp.DEVNULL).decode().strip()
    except Exception:                                                            # pragma: no cover
        pass

    meta_blob = {
        "experiment_name": cfg.experiment_name,
        "git_head_commit_sha": git_sha_now,
        "host_hostname": os.environ.get("HOSTNAME", ""),
        "gpu_visible_devices_env_var_at_launch_time": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "vllm_endpoint_baseurl_configured": cfg.model.base_url,
        "vllm_endpoint_modelid_expected": cfg.model.model_id,
        "require_no_lora_loaded_setting": cfg.model.require_no_lora_loaded,
        "generation_params_snapshot": {
            "temperature": cfg.generation.temperature,
            "top_p": cfg.generation.top_p,
            "top_logprobs_requested": cfg.generation.top_logprobs,
            "extra_stop_sequences": list(cfg.generation.extra_stop_sequences),
            "seed_passthrough_resolved": cfg.generation.seed_passthrough,
            "max_new_tokens_cap": cfg.generation.max_new_tokens,
        },
        "mining_parameters_snapshot": {
            "cap_per_bucket": cfg.mining.cap_per_bucket,
            "max_domain_fraction_in_bucket": cfg.mining.max_domain_fraction_in_bucket,
            "sampling_seed": cfg.mining.sampling_seed,
            "sources_mined": list(cfg.mining.source_experiments),
        },
        "stats_plan_snapshot": {
            "alpha_overall_declared_upfront": cfg.stats.alpha_overall,
            "bonferroni_correction_count_planned": cfg.stats.bonferroni_correction_count,
            "bootstrap_iters_planned": cfg.stats.bootstrap_iters,
            "ci_level_targeted": cfg.stats.ci_level,
            "planned_pairwise_comparisons": [["imm","d1"],["d1","d2"],["d2","d3"],["d3","d4"]],
        },
        "harness_runtime_summary_extra": extra_summary_fields,
        "raw_responses_filepath_relative_to_outputdir": os.path.relpath(out_responses_jsonl_file, output_dir),
        "dry_run_mode_active_during_collection_phase": cfg.dry_run,
        "started_at_utc_iso8601": started_iso,
        "finished_at_utc_iso8601": finished_iso,
    }
    emit("__root__", meta_blob)
    yaml_body_lines = ["__root__:"]
    flat_dump_yaml(meta_blob, depth=0, sink=yaml_body_lines)
    with open(mp, "w", encoding="utf-8") as fy:
        fy.write("# Auto-generated by preliminary.harness.HarnessRunner.run()\n")
        fy.write("\n".join(yaml_body_lines))


def flat_dump_yaml(obj: Any, depth: int, sink: list[str]) -> None:
    """Tiny recursive emitter avoiding dependency on PyYAML round-trip semantics for nested dicts/lists/tuples/scalars."""
    pad = "  " * depth
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                sink.append(f"{pad}{k}:")
                flat_dump_yaml(v, depth + 1, sink)
            elif isinstance(v, (list, tuple)):
                sink.append(f"{pad}{k}:")
                for el in v:
                    if isinstance(el, dict):
                        sink.append(f"{pad}- ")
                        flat_dump_yaml(el, depth + 2, sink)
                    else:
                        sink.append(f"{pad}- {el!r}")
            else:
                sink.append(f"{pad}{k}: {_yaml_scalar(v)}")
    else:
        sink.append(f"{pad}{_yaml_scalar(obj)}")


def _yaml_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return repr(v) if "\n" not in str(v) else "|-\n" + str(v)


# --------------------------------------------------------------------------- #
# CLI entrypoint invoked when running standalone (`python -m preliminary.harness ...`)
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m preliminary.harness")
    ap.add_argument("--config", default="configs/preliminary_entropy.yaml")
    ap.add_argument("--loglevel", default=None)
    args = ap.parse_args(argv)

    if __package__ in (None, ""):                                                # pragma: no cover - module-script invocation shim
        from preliminary.config import load_config
    else:
        from .config import load_config
    cfg = load_config(args.config)
    lvl = args.loglevel or cfg.logging_level.upper()
    logging.basicConfig(level=lvl, format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s")

    runner = HarnessRunner(cfg)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
