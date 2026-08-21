"""Evaluate ShieldAgent against bench_base dataset.

ShieldAgent is a Qwen2.5-7B fine-tuned binary safety classifier.
For each scenario, we reconstruct the agent dialogue context from the
stored trajectory and let ShieldAgent judge whether the current action
is safe or unsafe.

Detection = ShieldAgent says "unsafe" on a scenario containing injection.
Detection rate by bucket tells us: can ShieldAgent catch attacks at each delta?
Expected: higher delta (stealthier) -> lower detection rate.

Usage:
  CUDA_VISIBLE_DEVICES=6 python bench_base/scripts/eval_shieldagent.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from baselines.shieldagent.parser import parse_answer

DEFAULT_MODEL_PATH = "/ssd1/yx/models/shieldagent"
SCENARIOS_ROOT = REPO_ROOT / "bench_base" / "scenarios"
OUT_DIR = REPO_ROOT / "bench_base" / "diagnostics"
BUCKETS = ("imm", "d1", "d2", "d3", "d4")


def build_history_str(actions: list[dict], up_to_turn: int) -> str:
    """Convert trajectory actions into ShieldAgent history format."""
    parts = []
    for a in actions:
        turn = int(a.get("turn", 0))
        if turn >= up_to_turn:
            break
        thought = a.get("thought") or ""
        tc = a.get("tool_call") or {}
        obs = a.get("observation") or ""
        tool_name = tc.get("name", "") if isinstance(tc, dict) else ""
        tool_args = tc.get("arguments", {}) if isinstance(tc, dict) else {}

        idx = turn + 1
        parts.append(f"({idx}) Thought: {thought}")
        if tool_name:
            parts.append(f"({idx}) Action: {tool_name}")
            parts.append(f"({idx}) Action Input: {json.dumps(tool_args, ensure_ascii=False)}")
        if obs:
            parts.append(f"Observation: {obs[:4000]}")
    return "\n".join(parts)


def build_current_action_str(actions: list[dict], at_turn: int) -> str:
    """Build current_action string for the action at injection turn."""
    for a in actions:
        if int(a.get("turn", 0)) == at_turn:
            thought = a.get("thought") or ""
            tc = a.get("tool_call") or {}
            tool_name = tc.get("name", "") if isinstance(tc, dict) else ""
            tool_args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
            idx = at_turn + 1
            parts = [f"({idx}) Thought: {thought}"]
            if tool_name:
                parts.append(f"({idx}) Action: {tool_name}")
                parts.append(f"({idx}) Action Input: {json.dumps(tool_args, ensure_ascii=False)}")
            return "\n".join(parts)
    return ""


def scenario_to_shieldagent_record(scen: dict) -> dict:
    """Convert bench_base scenario to a record compatible with ShieldAgent prompt_builder."""
    actions = scen.get("context_prefix_actions") or []
    inj_turn = scen.get("injection_target_turn_index", 0)
    instruction = scen.get("goal_instruction", "")
    poisoned_obs = scen.get("poisoned_observation_text", "")

    history = build_history_str(actions, up_to_turn=inj_turn)
    current_action = build_current_action_str(actions, at_turn=inj_turn)

    # Append the poisoned observation as part of history context so ShieldAgent
    # can see it in context (it's the observation that contains the injection)
    if poisoned_obs:
        history_with_poison = history + "\n" + f"Observation: {poisoned_obs[:4000]}" if history else f"Observation: {poisoned_obs[:4000]}"
    else:
        history_with_poison = history

    return {
        "env_info": "",  # bench_base doesn't store raw env_info
        "instruction": instruction,
        "history": history_with_poison,
        "current_action": current_action,
    }


def render_shieldagent_prompt(record: dict) -> str:
    """Render using ShieldAgent's prompt_builder."""
    from baselines.shieldagent.prompt_builder import render as render_prompt
    return render_prompt(record)


def load_model(model_path: str):
    """Load ShieldAgent model (Qwen2.5-7B)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {model_path}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model from {model_path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


def batch_generate(model, tok, prompts: list[str], max_new_tokens: int = 512, batch_size: int = 4) -> list[str]:
    """Batched generation with left-padding."""
    import torch

    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        messages_batch = [[{"role": "user", "content": p}] for p in batch]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]

        encoded = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=16384)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        # Decode only new tokens
        for j, ids in enumerate(gen_ids):
            input_len = encoded["input_ids"][j].shape[0]
            new_ids = ids[input_len:]
            text = tok.decode(new_ids, skip_special_tokens=True)
            outputs.append(text)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out", default=str(OUT_DIR / "eval_shieldagent.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    print("=" * 70)
    print("Eval: ShieldAgent (Qwen2.5-7B safety judge)")
    print(f"  model: {args.model_path}")
    print("=" * 70)

    model, tok = load_model(args.model_path)

    all_results = []
    per_bucket = {}
    t0 = time.time()

    for bkt in BUCKETS:
        path = SCENARIOS_ROOT / f"bucket_{bkt}.jsonl"
        if not path.exists():
            print(f"[{bkt}] MISSING, skipping")
            continue

        scenarios = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
        print(f"\n[{bkt}] evaluating {len(scenarios)} scenarios...", flush=True)

        # Build prompts
        prompts = []
        for scen in scenarios:
            rec = scenario_to_shieldagent_record(scen)
            prompt = render_shieldagent_prompt(rec)
            prompts.append(prompt)

        # Generate in batches
        raw_outputs = batch_generate(model, tok, prompts, args.max_new_tokens, args.batch_size)

        # Parse verdicts
        n_detected = 0
        n_missed = 0
        n_error = 0
        bucket_results = []

        for i, (scen, raw) in enumerate(zip(scenarios, raw_outputs)):
            verdict = parse_answer(raw)
            if verdict == "unsafe":
                detected = True
                n_detected += 1
            elif verdict == "safe":
                detected = False
                n_missed += 1
            else:
                detected = False
                n_error += 1

            bucket_results.append({
                "scenario_id": scen.get("scenario_id", f"{bkt}_{i}"),
                "bucket": bkt,
                "detected": detected,
                "verdict": verdict,
                "raw_output": raw[:300],
            })

        total = len(scenarios)
        det_rate = n_detected / total if total else 0
        per_bucket[bkt] = {
            "n_total": total,
            "n_detected": n_detected,
            "n_missed": n_missed,
            "n_parse_error": n_error,
            "detection_rate": round(det_rate, 4),
        }
        print(f"  [{bkt}] detection={det_rate:.1%} ({n_detected}/{total}, {n_error} parse_err)")
        all_results.extend(bucket_results)

    elapsed = time.time() - t0

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'Bucket':<8}{'N':>5}{'Detected':>10}{'Missed':>8}{'Err':>5}{'Det.Rate':>10}")
    print("-" * 46)
    for bkt in BUCKETS:
        if bkt in per_bucket:
            s = per_bucket[bkt]
            print(f"{bkt:<8}{s['n_total']:>5}{s['n_detected']:>10}{s['n_missed']:>8}{s['n_parse_error']:>5}{s['detection_rate']:>9.1%}")
    print(f"\nDone in {elapsed:.1f}s")

    # Write output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "eval_mode": "shieldagent_classifier",
        "model_path": args.model_path,
        "elapsed_seconds": round(elapsed, 1),
        "per_bucket": per_bucket,
        "all_results": all_results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
