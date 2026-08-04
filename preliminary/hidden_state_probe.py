"""隐状态熵探针：绕开 RLHF 抛光过的、被截到 top-K 的表层 logprob。

背景与动机
----------
Phase-A 用 vLLM 返回的 ``top_logprobs=20`` 算出的 token 平均熵，
跟 Δ 分桶（攻击潜伏等级）之间 Spearman 几乎为零。一个高度可疑的混淆因素是：
Qwen2.5-Instruct 经过 SFT/RLHF 之后，其末层输出投影已被训练成「低熵的确定式续写」，
无论内心有多犹豫，最终落在 vocab 上的概率都会被压得很尖，
于是仅凭表层 token 分布根本读不出模型的真实警觉程度。

所以本探针换个角度：跳过对外接口这道「伪装」，
用 ``transformers.AutoModelForCausalLM`` 直接把底座 Qwen2.5-7B-Instruct 原始权重加载进来，
亲自跑一次前向传播，去观测进入输出投影之前的内在不确定性。

核心思路
--------
对 bench 里每一条场景只跑 **一次** 前向传播，重点盯住的位置就是
「刚刚读完那条被投毒的工具结果之后」紧挨着的下一格 logits 与隐藏态：

1. 由 :func:`preliminary.context_builder.build_messages_for_scenario`
   构造好标准的对话上下文 messages；
2. tokenizer.apply_chat_template 加上 ``add_generation_prompt=True`` 转 id 序列——
   相当于告诉模型「现在该你说话了」，但其实并不让它继续生成；
3. 若输入超过 ``max_input_length``(默认32000)，则从左边砍掉保留最近一段；
4. 在 ``torch.no_grad`` 下调用
   ``model(input_ids, output_hidden_states=True, use_cache=False)`` 完成单次推理；
5. 取出最后一个序列位的 logits 以及若干层的 hidden-state 向量计算下面这些度量。

测出来哪些量
------------
============================== ============================================
度量字段                       大白话含义
============================== ============================================
``H_logits_full_vocab``        整张约 15 万词表的 softmax 分布上的香农熵；这是真正的全谱不确定度，不受 K=20 截断限制。
``H_top20_restricted``         只取最大的 20 个候选再重新归一化后算的熵；用于和老 harness 数字对齐做一致性检查。
``p_argmax_full_vocab``        全词表中最大那个候选所占的概率质量；越小表示越纠结。
``l2_norm_final_layer``        最末一层 transformer block 输出向量的模长；可理解为当前时刻神经活动总强度。
``l2_norm_mid_layer``          中间深度同一位置隐藏向量的模长。
``eff_rank_final_layer``       「有效秩」。把每一维平方当作一份能量求占比分布再算香农熵；能量越摊得均匀有效秩越高。
``eff_rank_mid_layer``         同上但取自中层激活。
``n_input_tokens_at_probe_position``
                               进入探针位置之前的输入侧 token 数目。
============================== ============================================

数据落盘约定
------------
所有指标一律使用上述真实名称作为 CSV 字段，不再像早期那样借用旧 harness
的列名做兼容映射——既然不复用旧的统计/画图代码，就无需承担名字映射带来的认知负担。
元信息部分（scenario_id / bucket / delta_value_orig / domain / method /
truncated / prompt_hash / _finish_reason / _error_message_excerpt）维持原样不变，
以便已有的按 scenario_id 断点续跑机制继续生效。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
from typing import Any

if __package__ in (None, ""):                                                   # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from preliminary.context_builder import build_messages_for_scenario
    from preliminary.harness import iter_scenarios
else:
    from .context_builder import build_messages_for_scenario
    from .harness import iter_scenarios

logger = logging.getLogger("preliminary.hidden_state_probe")

DEFAULT_MODEL_PATH = "/ssd1/models/qwen2.5-7b-it"
QWEN25_VOCAB_SIZE_LOG_APPROX = math.log(152064)   # ln(|vocab|) for normalized-H denominator


# 字段全部使用本探针的原生命名——不再借壳复用旧 harness 列名，
# 因为后续的统计与画图逻辑都在本文件内部实现，无需向外部 stats/plot 模块妥协。
PROBE_CSV_COLUMN_ORDER: tuple[str, ...] = (
    # 元数据
    "scenario_id", "bucket", "delta_value_orig", "domain", "method",
    "truncated",
    # 探针核心度量（原名直出）
    "n_input_tokens_at_probe_position",
    "H_logits_full_vocab_nats",
    "H_top20_restricted_from_full_softmax",
    "p_argmax_full_vocab",
    "H_normalized_by_log_vocab_size",
    "l2_norm_final_layer",
    "l2_norm_mid_layer",
    "eff_rank_final_layer",
    "eff_rank_mid_layer",
    # 元数据补充 / 调试线索
    "prompt_hash",
    "_finish_reason",
    "_error_message_excerpt",
    "__provenance_method_label",
)

# 分桶汇总时关心的「数值型核心度量」。其它列要么是元信息要么是 NaN 占位。
NUMERIC_METRIC_COLUMNS: tuple[str, ...] = (
    "H_logits_full_vocab_nats",
    "H_top20_restricted_from_full_softmax",
    "p_argmax_full_vocab",
    "H_normalized_by_log_vocab_size",
    "l2_norm_final_layer",
    "l2_norm_mid_layer",
    "eff_rank_final_layer",
    "eff_rank_mid_layer",
)


def _sha256_messages(messages: list[dict[str, str]]) -> str:
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# Model loading helpers
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(model_path: str, device_str: str):
    """Lazy-import torch+transformers (heavy deps kept behind gate).

    Loads model in bfloat16 to fit ~14 GiB VRAM budget comfortably within one GPU."""
    import torch                                                              # noqa: F401 lazy guard
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading tokenizer from %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    logger.info("loading %s onto device=%s dtype=bfloat16", model_path, device_str)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device_str)
    model.eval()
    return model, tok


# --------------------------------------------------------------------------- #
# Per-scenario forward pass
# --------------------------------------------------------------------------- #
def probe_one(scenario: dict[str, Any],
              repo_root: str,
              tokenizer,
              model,
              device_obj,
              *,
              max_input_length: int = 32000,
              numpy_ref=None) -> dict[str, Any]:
    """Run single forward pass returning metrics row matching PROBE_CSV_COLUMN_ORDER keys."""

    import torch                                                               # local heavy dep kept behind function gate
    np = numpy_ref or __import__("numpy")                                      # late binding so caller controls env

    sid = str(scenario.get("scenario_id") or "?")
    bk = str(scenario.get("bucket") or "?")
    delta_orig = scenario.get("delta_value_orig")
    dom = str(scenario.get("domain") or "?")
    method_tag = str(scenario.get("method") or "?")
    prompt_hash_local = ""

    try:
        msgs = build_messages_for_scenario(scenario, repo_root=repo_root)
        prompt_hash_local = _sha256_messages(msgs)[:16]

        input_ids_tensor = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device_obj)

        if input_ids_tensor.shape[1] > max_input_length:
            input_ids_tensor = input_ids_tensor[:, -max_input_length:]

        n_input_toks = int(input_ids_tensor.shape[1])

        with torch.no_grad():
            out = model(input_ids=input_ids_tensor, output_hidden_states=True, use_cache=False)

        last_pos_logits_fp32 = out.logits[0, -1, :].to(torch.float32).cpu().numpy().astype("float64")
        probs_full = np.exp(last_pos_logits_fp32 - last_pos_logits_fp32.max())      # subtractive-max for numerical stability pre-exp
        Z_total = float(probs_full.sum())
        if Z_total <= 0:
            raise RuntimeError("softmax normalization failed (Z<=0)")
        probs_full /= Z_total                                                    # now sums to ~1.0 within fp64 precision

        eps = 1e-12
        sp = probs_full.clip(min=eps)
        H_full_vocab_nats = float(-(sp * np.log(sp)).sum())

        k20_idx_arr = np.argpartition(probs_full, -20)[-20:]
        topk_p_vals = probs_full[k20_idx_arr].astype(float)
        Z_k20_val = float(topk_p_vals.sum())
        if Z_k20_val <= 0:
            H_top_20_restricted = float("nan")
        else:
            pk20_normalised = topk_p_vals / Z_k20_val
            spk20 = pk20_normalised.clip(min=eps)
            H_top_20_restricted = float(-(spk20 * np.log(spk20)).sum())

        p_argmax_fv = float(probs_full.max())

        all_layers_tuple = out.hidden_states                                      # tuple len=num_layers+1 idx0 embeddings passthrough
        hs_final_vec = all_layers_tuple[-1][0, -1, :].to(torch.float32).cpu().numpy()
        n_layers_total = len(all_layers_tuple) - 1                                # exclude embedding-passthrough entry [0]
        mid_idx_incl_embed_formulation = max(1, n_layers_total // 2)
        hs_mid_vec = all_layers_tuple[mid_idx_incl_embed_formulation][0, -1, :].to(torch.float32).cpu().numpy()

        def _local_eff_rank(vec_np):
            sq_loc = vec_np * vec_np
            s_loc = float(sq_loc.sum())
            if s_loc <= 0:
                return 0.0
            pr_loc = (sq_loc / s_loc).clip(min=eps)
            return float(-(pr_loc * np.log(pr_loc)).sum())

        eff_rank_final = _local_eff_rank(hs_final_vec)
        eff_rank_mid = _local_eff_rank(hs_mid_vec)
        l2_final = float(math.sqrt(float((hs_final_vec * hs_final_vec).sum())))
        l2_mid = float(math.sqrt(float((hs_mid_vec * hs_mid_vec).sum())))

        finish_reason_label = "stop"
        err_excerpt_out = ""

    except Exception as exc:                                                     # noqa: BLE001 defensive catch preserving long-run progress
        logger.exception("forward pass failed for scenario_id=%s (%s)", sid, exc)
        return {
            "scenario_id": sid, "bucket": bk, "delta_value_orig": delta_orig,
            "domain": dom, "method": method_tag, "truncated": True,
            "n_input_tokens_at_probe_position": 0,
            "H_logits_full_vocab_nats": float("nan"),
            "H_top20_restricted_from_full_softmax": float("nan"),
            "p_argmax_full_vocab": float("nan"),
            "H_normalized_by_log_vocab_size": float("nan"),
            "l2_norm_final_layer": "",
            "l2_norm_mid_layer": "",
            "eff_rank_final_layer": "",
            "eff_rank_mid_layer": "",
            "prompt_hash": "",
            "_finish_reason": "error",
            "_error_message_excerpt": f"{exc!s}"[:120],
            "__provenance_method_label": "hidden_state_probe_error_row",
        }

    return {
        "scenario_id": sid,
        "bucket": bk,
        "delta_value_orig": delta_orig,
        "domain": dom,
        "method": method_tag,
        "truncated": False,
        "n_input_tokens_at_probe_position": n_input_toks,

        # 探针原生度量（原名直出，无任何借壳映射）
        "H_logits_full_vocab_nats": round(H_full_vocab_nats, 6),
        "H_top20_restricted_from_full_softmax": round(H_top_20_restricted, 6),
        "p_argmax_full_vocab": round(p_argmax_fv, 6),
        "H_normalized_by_log_vocab_size":
            round(H_full_vocab_nats / QWEN25_VOCAB_SIZE_LOG_APPROX, 6),

        # 隐藏态侧度量
        "l2_norm_final_layer": round(l2_final, 6),
        "l2_norm_mid_layer": round(l2_mid, 6),
        "eff_rank_final_layer": round(eff_rank_final, 4),
        "eff_rank_mid_layer": round(eff_rank_mid, 4),

        # 元信息
        "prompt_hash": prompt_hash_local,
        "_finish_reason": finish_reason_label,
        "_error_message_excerpt": err_excerpt_out,

        "__provenance_method_label": "hidden_state_forward_pass_pre_output_layer_full_vocab_softmax",
    }


# --------------------------------------------------------------------------- #
# 自带的分桶汇总与画图：不再依赖 preliminary.stats / preliminary.plot
# --------------------------------------------------------------------------- #
def _coerce_float(v: Any) -> float | None:
    """把 CSV 单元格里的值尽量转成 float，转不动就返回 None。"""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if isinstance(f, float) and (f != f):                       # NaN 判定
        return None
    return f


def _summarize_rows_by_bucket(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 bucket 聚合每个数值度量的 count / mean / std / min / max。

    返回结构 ``{bucket_name: {metric_name: {n,mean,std,min,max}}}``，
    其中还塞一个 ``"_n_total"`` 记录该桶总行数（含 error 行）便于核对。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        bk = str(r.get("bucket") or "?")
        grouped.setdefault(bk, []).append(r)

    summary: dict[str, dict[str, Any]] = {}
    for bk, lst in sorted(grouped.items()):
        per_metric: dict[str, Any] = {"_n_total": len(lst)}
        for col in NUMERIC_METRIC_COLUMNS:
            vals = [_coerce_float(r.get(col)) for r in lst]
            vals = [v for v in vals if v is not None]
            if not vals:
                per_metric[col] = {"n": 0, "mean": None, "std": None,
                                    "min": None, "max": None}
                continue
            n_val = len(vals)
            mean_v = sum(vals) / n_val
            var_v = sum((v - mean_v) ** 2 for v in vals) / max(1, n_val - 1)
            std_v = math.sqrt(var_v)
            per_metric[col] = {
                "n": n_val, "mean": round(mean_v, 6),
                "std": round(std_v, 6),
                "min": round(min(vals), 6), "max": round(max(vals), 6),
            }
        summary[bk] = per_metric
    return summary


def _format_bucket_summary_table(per_bucket: dict[str, dict[str, Any]]) -> str:
    """把分桶汇总渲染成一段人类可读的纯文本表格，方便直接打印到终端。"""
    if not per_bucket:
        return "(no completed rows to summarize)"

    headers = ["bucket", "n"] + [c.replace("_nats", "").replace("H_", "")
                                  .replace("_from_full_softmax", "_topk")
                                  .replace("_by_log_vocab_size", "_norm")
                                  for c in NUMERIC_METRIC_COLUMNS]
    lines = ["\t".join(headers)]
    for bk, metrics in per_bucket.items():
        row_cells = [bk, str(metrics.get("_n_total", 0))]
        for col in NUMERIC_METRIC_COLUMNS:
            m = metrics.get(col) or {}
            mu = m.get("mean")
            sd = m.get("std") or 0.0
            row_cells.append(f"{mu:.4f}±{sd:.4f}" if mu is not None else "-")
        lines.append("\t".join(row_cells))
    return "\n".join(lines)


def _maybe_render_plots(per_bucket: dict[str, dict[str, Any]],
                        png_path: str) -> tuple[bool, str]:
    """尝试用 matplotlib 把各度量按桶的均值条形图画出来。

    成功返回 ``(True, png_path)``；若环境里没装 matplotlib 或者保存失败，
    返回 ``(False, reason_str)``，调用方继续往下走即可，不让绘图拖垮主流程。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")                                    # 无显示器环境也能存盘
        import matplotlib.pyplot as plt
    except Exception as exc_imp:                                 # noqa: BLE001
        return False, f"matplotlib unavailable ({exc_imp})"

    bucket_names = list(per_bucket.keys())
    if not bucket_names:
        return False, "no data points to plot"
    metric_cols = list(NUMERIC_METRIC_COLUMNS)
    n_metrics = len(metric_cols)

    # 一张图里给每个度量开一个子图；横轴是桶、纵轴是均值并叠加 std 误差棒。
    fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 2.2 * n_metrics),
                             squeeze=False)
    for idx_ax, col in enumerate(metric_cols):
        ax = axes[idx_ax][0]
        means: list[float] = []
        stds: list[float] = []
        valid_buckets: list[str] = []
        for bk in bucket_names:
            m = per_bucket[bk].get(col) or {}
            mu = m.get("mean"); sd = m.get("std") or 0.0
            if mu is None:
                continue
            means.append(float(mu)); stds.append(float(sd)); valid_buckets.append(bk)
        if not valid_buckets:
            ax.set_title(f"{col}\n(no data)")
            continue
        x_pos = list(range(len(valid_buckets)))
        ax.bar(x_pos, means, yerr=stds, capsize=4, color="#5b9bd5",
               edgecolor="#1f3864")
        ax.set_xticks(x_pos); ax.set_xticklabels(valid_buckets, rotation=30)
        ax.set_ylabel(col); ax.set_title(col)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    try:
        plt.savefig(png_path, dpi=120)
        plt.close(fig)
        return True, png_path
    except Exception as exc_save:                                # noqa: BLE001 best-effort
        try:
            plt.close(fig)
        except Exception:                                        # noqa: BLE001 ignore cleanup error
            pass
        return False, f"savefig failed ({exc_save})"



def run(*, bench_dir: str,
        output_csv_path: str,
        raw_jsonl_path: str | None = None,
        repo_root: str,
        model_path: str = DEFAULT_MODEL_PATH,
        cuda_visible_devices_override: str | None = None,
        resume: bool = True) -> dict[str, Any]:

    """Iterate scenarios emitting per-scenario probe rows to CSV incrementally resumable by scenario_id."""

    # Apply CUDA_VISIBLE_DEVICES env override BEFORE importing/loading torch-bound objects.
    env_cvd_save = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices_override is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices_override)
        logger.warning("set CUDA_VISIBLE_DEVICES=%s for this process lifetime "
                       "(prior value=%s)", cuda_visible_devices_override, env_cvd_save)

    done_ids: set[str] = set()
    if raw_jsonl_path and resume and os.path.isfile(raw_jsonl_path):
        with open(raw_jsonl_path, "r", encoding="utf-8") as fin_done:
            for ln in fin_done:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec_dct = json.loads(ln)
                    sid_done = rec_dct.get("scenario_id")
                    sid_skip_marker = rec_dct.get("_skipped_error")
                    if isinstance(sid_done, str) and not sid_skip_marker:
                        done_ids.add(sid_done)
                except Exception:
                    continue
        logger.info("resume mode detected: skipping %d already-completed scenario_ids", len(done_ids))

    fout_raw_handle = open(raw_jsonl_path, "a+", encoding="utf-8") if raw_jsonl_path else None

    csv_tmp_path = output_csv_path + ".tmp"
    fields_list_extended = list(PROBE_CSV_COLUMN_ORDER)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    fout_csv = open(csv_tmp_path, "w", encoding="utf-8", newline="")
    wcsv = csv.DictWriter(fout_csv, fieldnames=fields_list_extended, extrasaction="ignore")
    wcsv.writeheader()

    import torch                                                                  # ensure available to derive device string
    dev_str_resolved = "cuda" if torch.cuda.is_available() else "cpu"
    if dev_str_resolved == "cpu":
        logger.error("torch.cuda.is_available()==False ; falling back to CPU will be extremely slow.")
    elif cuda_visible_devices_override is not None:
        try:
            gpu_name = torch.cuda.get_device_name(int(cuda_visible_devices_override))
            logger.info("GPU pinned CUDA_VISIBLE_DEVICES=%s -> '%s'", cuda_visible_devices_override, gpu_name)
        except Exception as exc_gn:                                                # noqa: BLE001 best-effort diagnostic
            logger.warning("could not resolve pinned-GPU display name (%s)", exc_gn)

    model_obj, tok_obj = load_model_and_tokenizer(model_path=model_path, device_str=dev_str_resolved)

    seen_buckets_counter: dict[str, int] = {}
    processed_count_this_run = 0
    skipped_already_done_resume_count = 0
    skipped_error_rows_count = 0
    collected_rows: list[dict[str, Any]] = []                   # 供末尾分桶汇总使用

    scen_iter = iter_scenarios(bench_dir)
    for scn in scen_iter:
        sid_iter = str(scn.get("scenario_id"))
        if sid_iter in done_ids:
            skipped_already_done_resume_count += 1
            continue

        row_dict_this_scen = probe_one(scn, repo_root=repo_root,
                                       tokenizer=tok_obj, model=model_obj,
                                       device_obj=dev_str_resolved)

        wcsv.writerow({fld_nm: row_dict_this_scen.get(fld_nm, "")
                       for fld_nm in fields_list_extended})
        collected_rows.append(row_dict_this_scen)

        fin_reason_seen = str(row_dict_this_scen.get("_finish_reason",""))
        if fin_reason_seen == "error":
            skipped_error_rows_count += 1
        else:
            processed_count_this_run += 1
            bk_seen_now = str(row_dict_this_scen.get("bucket") or "?")
            seen_buckets_counter[bk_seen_now] = seen_buckets_counter.get(bk_seen_now, 0) + 1

        if fout_raw_handle:
            fout_raw_handle.write(json.dumps(row_dict_this_scen, default=str, ensure_ascii=False) + "\n")
            fout_raw_handle.flush()

        if (processed_count_this_run + skipped_error_rows_count) % 10 == 0:
            logger.info("progress total_processed_or_errored=%d ; newly_probed_ok=%d ; errors_so_far=%d ; last_sid=%s",
                        processed_count_this_run + skipped_error_rows_count,
                        processed_count_this_run, skipped_error_rows_count, sid_iter)

    fout_csv.close()
    os.replace(csv_tmp_path, output_csv_path)
    if fout_raw_handle:
        fout_raw_handle.close()

    # 自带的分桶汇总：不再依赖 preliminary.stats。
    per_bucket_summary = _summarize_rows_by_bucket(collected_rows)

    output_csv_abs = os.path.abspath(output_csv_path)
    out_dir_for_artifacts = os.path.dirname(output_csv_abs) or "."
    summary_json_path = os.path.join(out_dir_for_artifacts,
                                     "hidden_state_probe_summary.json")

    # 把每条原始行的关键字段也一并落盘，方便事后用其它工具复算相关性等指标，
    # 而无需再去解析 CSV（CSV 里 NaN/空字符串混用容易踩坑）。
    flat_metric_records: list[dict[str, Any]] = []
    for r in collected_rows:
        rec: dict[str, Any] = {
            "scenario_id": r.get("scenario_id"),
            "bucket": r.get("bucket"),
            "delta_value_orig": r.get("delta_value_orig"),
            "domain": r.get("domain"),
            "method": r.get("method"),
            "_finish_reason": r.get("_finish_reason"),
        }
        for col in NUMERIC_METRIC_COLUMNS:
            v = _coerce_float(r.get(col))
            if col.endswith(("_final_layer", "_mid_layer")):
                pass                                                        # 数值列保持原样即可，不强制转 float
            rec[col] = (v if v is not None else None)
        flat_metric_records.append(rec)

    try:
        with open(summary_json_path, "w", encoding="utf-8") as fsum:
            json.dump({
                "per_bucket_aggregates": per_bucket_summary,
                "per_row_metrics": flat_metric_records,
                "column_order_emitted_to_csv": list(PROBE_CSV_COLUMN_ORDER),
                "numeric_metric_columns": list(NUMERIC_METRIC_COLUMNS),
                "counts": {
                    "newly_probed_ok": processed_count_this_run,
                    "resumed_skipped": skipped_already_done_resume_count,
                    "errors": skipped_error_rows_count,
                },
            }, fsum, ensure_ascii=False, indent=2, default=str)
    except Exception as exc_dump:                                          # noqa: BLE001 best-effort artifact write
        logger.warning("failed to dump %s (%s)", summary_json_path, exc_dump)

    png_target_path = os.path.join(out_dir_for_artifacts,
                                   "hidden_state_probe_per_bucket.png")
    plot_ok_flag, plot_msg = _maybe_render_plots(per_bucket_summary, png_target_path)

    print("\n=== HIDDEN-STATE PROBE SUMMARY ===")
    print(f"output_csv={output_csv_path}")
    print(f"raw_jsonl={raw_jsonl_path}")
    print(f"summary_json={summary_json_path}")
    print(f"plot_png={'%s' % png_target_path if plot_ok_flag else '(skipped: %s)' % plot_msg}")
    print(f"processed_this_run={processed_count_this_run} "
          f"; resumed_skipped={skipped_already_done_resume_count}"
          f"; errors={skipped_error_rows_count}")
    sbk_sorted_items = sorted(seen_buckets_counter.items(), key=lambda kv: kv[0])
    print("per-bucket newly-probed-ok:",
          ", ".join(f"{b}:{c}" for b,c in sbk_sorted_items))

    print("\n--- per-bucket metric aggregates (mean±std across rows) ---")
    print(_format_bucket_summary_table(per_bucket_summary))

    return {
        "output_csv_absolute": output_csv_abs,
        "summary_json_absolute": os.path.abspath(summary_json_path),
        "plot_png_rendered": bool(plot_ok_flag),
        "per_bucket_aggregates": per_bucket_summary,
        "processed_this_run": processed_count_this_run,
        "resumed_skipped": skipped_already_done_resume_count,
        "errors": skipped_error_rows_count,
    }


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m preliminary.hidden_state_probe")
    ap.add_argument("--bench-dir", required=False, default="bench")
    ap.add_argument("--output-csv-path", required=True)
    ap.add_argument("--raw-jsonl-path", default=None)
    ap.add_argument("--repo-root",
                    default=os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--cuda-visible-devices-override", type=str, default=None,
                    help="Pin specific GPU id BEFORE importing torch/loaders.")
    args_cli_parsed = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
    result_summary_dict = run(
        bench_dir=args_cli_parsed.bench_dir,
        output_csv_path=args_cli_parsed.output_csv_path,
        raw_jsonl_path=args_cli_parsed.raw_jsonl_path,
        repo_root=args_cli_parsed.repo_root,
        model_path=args_cli_parsed.model_path,
        cuda_visible_devices_override=args_cli_parsed.cuda_visible_devices_override,
    )
    del result_summary_dict
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
