"""verl / AEPO GRPO wrapper (``training/intro.md``).

Mirrors :mod:`evoguard.training.sft_runner` for the RL half of the defender
training recipe (``docs/plan.md``: SFT cold-start + GRPO on LoRA). AEPO ships
verl with a PPO/GRPO trainer driven by Hydra YAML configs; this module renders
an EvoGuard-flavored config plus dataset files and produces a launch command
mirroring ``AEPO/scripts/AEPO_Qwen25_7B_DeepResearch.sh``.

AEPO expects parquet inputs with at least ``prompt``/``data_source`` columns.
We emit prompts as JSONL alongside a small converter script (pandas+pyarrow,
already part of AEPO's own requirements) so users can produce parquet on demand;
in ``dry_run`` mode we only materialize artifacts without invoking anything.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from evoguard.config import TrainingConfig
from evoguard.utils.logging import get_logger


logger = get_logger("training.grpo")


_CONVERTER_TEMPLATE = '''\
"""Convert EvoGuard GRPO JSONL into an AEPO/verl parquet file."""
from __future__ import annotations
import argparse, json, os
import pandas as pd  # type: ignore  -- present via AEPO requirements


def _load(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-prompt", required=True)
    ap.add_argument("--out-train", required=True)
    args = ap.parse_args()

    rows = _load(args.in_prompt)
    df = pd.DataFrame(rows)
    df["data_source"] = "evoguard_grpo"
    # verl reads reward via reward_fn_key=data_source; we attach scalar reward to
    # extra_info so user-supplied reward functions can pick it up deterministically.
    df["extra_info"] = [{"reward": float(r)} for r in df.get("reward", [0] * len(df))]
    out_dir = os.path.dirname(args.out_train) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.out_train)


if __name__ == "__main__":
    main()
'''


@dataclass
class GRPORunSpec:
    """Result of preparing a GRPO run."""

    prompt_jsonl: str           # per-sample row file written here
    convert_script: str         # script turning jsonl -> AEPO parquet input
    hydra_yaml: str             # rendered ppo_trainer.yaml override file
    command_template: str       # full shell invocation string for live runs
    adapter_dir: str            # where actor checkpoints land after training


@dataclass
class GRPOPlan:
    spec: GRPORunSpec
    n_samples: int
    launched: bool


# --------------------------------------------------------------------------- #
# Dataset writers & renderers                                                 #
# --------------------------------------------------------------------------- #
def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_rl_prompts(samples: list, prompt_jsonl_path: str) -> int:
    """Write :class:`RLSample`-like objects as JSON lines."""

    _ensure_parent(prompt_jsonl_path)
    with open(prompt_jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            payload = {
                "prompt": [
                    {"role": "system", "content": getattr(s, "system", "")},
                    {"role": "user",   "content": getattr(s, "prompt", "")},
                ],
                "response": getattr(s, "response", ""),
                "reward": float(getattr(s, "reward", 0.0)),
                "meta": dict(getattr(s, "meta", {}) or {}),
                "data_source": "evoguard_grpo",
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
    return len(samples)


def render_aepo_hydra_override(
    base_model: str,
    train_parquet_abs: str,
    val_parquet_abs: str | None,
    output_adapter_dir: str,
    cfg: TrainingConfig,
) -> str:
    """Render an EvoGuard override yaml consumed alongside vendored ppo_trainer.yaml.

    Users launch with::

        cd <AEPO>/verl_aepo_entropy && \\
        python -m verl.trainer.main_ppo evoguard=<our_file>
    """

    val_files_arg = val_parquet_abs or train_parquet_abs.replace(".parquet", "_val.parquet")
    experiment_label = output_adapter_dir.rstrip("/").split("/")[-2]

    parts: list[str] = []
    parts.append("# EvoGuard-rendered AEPO/verl PPO trainer configuration.")
    parts.append(f"# base model     : {base_model}")
    parts.append(f"# adapter output : {output_adapter_dir}")
    parts.append("")
    parts.append("hydra:")
    parts.append('  searchpath:')
    parts.append('    - pkg://verl.trainer.config')
    parts.append('')
    parts.append('defaults:')
    parts.append('  - ppo_trainer')
    parts.append('  - _self_')
    parts.append('')
    parts.append('data:')
    parts.append(f'  tokenizer: "{base_model}"')
    parts.append(f'  train_files: "{train_parquet_abs}"')
    parts.append(f'  val_files: "{val_files_arg}"')
    parts.append('  prompt_key: "prompt"')
    parts.append('  reward_fn_key: "data_source"')
    parts.append('  return_raw_chat: True')
    parts.append('  max_prompt_length: 2048')
    parts.append('  max_response_length: 1024')
    effective_batch = cfg.per_device_batch_size * cfg.gradient_accumulation * 8 or 128
    parts.append(f'  train_batch_size: {effective_batch}')
    parts.append('')
    parts.append('actor_rollout_ref:')
    parts.append('  hybrid_engine: True')
    parts.append('  model:')
    parts.append(f'    path: "{base_model}"')
    parts.append('    enable_gradient_checkpointing: True')
    parts.append(f'    lora_rank: {cfg.lora_rank}')
    parts.append(f'    lora_alpha: {cfg.lora_alpha}')
    parts.append('    target_modules: all-linear')
    parts.append('  actor:')
    parts.append('    strategy: fsdp')
    parts.append(f'    ppo_mini_batch_size: {max(1,cfg.per_device_batch_size*16)}')
    parts.append(f'    learning_rate: {cfg.grpo_learning_rate}')
    parts.append('    enable_kl_in_reward: True')
    parts.append('    kl_penalty: kl')
    parts.append('    kl_coef: 0.001')
    parts.append('  rollout:')
    parts.append('    name: vllm')
    parts.append('    mode: sync_with_tool')
    parts.append('    temperature: 0.7')
    parts.append('    top_p: 0.95')
    parts.append('    rollout_n: 16')
    parts.append('    stop_tokens: ["<|im_end|>"]')
    parts.append('')
    parts.append('trainer:')
    parts.append('  total_epochs: 1')
    parts.append('  project_name: evoguard_defense_rl')
    parts.append(f'  experiment_name: evoguard_round_{experiment_label}')
    parts.append(f'  default_local_dir: "{output_adapter_dir}/trl_state"')
    parts.append('  default_hdfs_dir: null')

    return "\n".join(parts)


def prepare_and_run_grpo(
    samples: list,
    exp_rounds_root: str,
    *,
    training_cfg: TrainingConfig,
    round_id_for_label: str = "",
    served_adapter_name_hint: str = "evoguard_defense_grpo_v0",
) -> GRPOPlan:
    """Materialize prompts/converter/yaml/command; launches iff not dry_run."""

    grpo_dir = os.path.join(exp_rounds_root, "grpo", round_id_for_label or "latest")
    adapter_out = os.path.join(grpo_dir, "adapter")
    data_subdir = os.path.join(grpo_dir, "data")
    os.makedirs(data_subdir, exist_ok=True)

    prompt_jsonl = os.path.join(data_subdir, "prompts.jsonl")
    n_samples = write_rl_prompts(samples, prompt_jsonl)

    conv_script = os.path.join(data_subdir, "to_parquet.py")
    with open(conv_script, "w", encoding="utf-8") as f:
        f.write(_CONVERTER_TEMPLATE)

    train_parquet_abs = os.path.abspath(os.path.join(data_subdir, "train.parquet"))
    yaml_text = render_aepo_hydra_override(
        base_model=training_cfg.base_model,
        train_parquet_abs=train_parquet_abs,
        val_parquet_abs=None,
        output_adapter_dir=adapter_out,
        cfg=training_cfg,
    )
    yaml_path = os.path.join(grpo_dir, "ppo_trainer.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    verl_root_abs = os.path.abspath(training_cfg.verl_root)
    cmd_parts = [
        f"cd {verl_root_abs}",
        f"python {conv_script} --in-prompt {prompt_jsonl} --out-train {train_parquet_abs}",
        (
            f"python -m verl.trainer.main_ppo "
            f"evoguard={yaml_path}"
        ),
    ]
    full_cmd = " ; ".join(cmd_parts)

    spec = GRPORunSpec(
        prompt_jsonl=prompt_jsonl,
        convert_script=conv_script,
        hydra_yaml=yaml_path,
        command_template=full_cmd,
        adapter_dir=adapter_out,
    )

    launched = False
    if not training_cfg.dry_run and samples:
        logger.info("[GRPO] launching (%d samples): %s", len(samples), full_cmd)
        rc = os.system(full_cmd)
        if rc != 0:
            raise RuntimeError(f"GRPO training failed with exit code {rc}")
        launched = True
    elif training_cfg.enabled:
        logger.info("[GRPO] dry-run prepared under %s ; command=%s", grpo_dir, full_cmd)

    return GRPOPlan(spec=spec, n_samples=n_samples, launched=launched)


# --- Public API ----------------------------------------------------------- #
__all__ = ["GRPORunSpec", "prepare_and_run_grpo", "write_rl_prompts", "render_aepo_hydra_override"]
