"""LLaMA-Factory SFT wrapper (``training/intro.md``).

This module turns a round's :class:`~evoguard.process.SFTExample` list into a
ready-to-train LLaMA-Factory dataset on disk and renders the matching launch
command. It is *dry-run friendly*: in ``dry_run`` mode it only writes the
dataset + config + command and never invokes the trainer, so the pipeline can
be exercised end-to-end without GPU resources.

The dataset format follows the ShareGPT-style schema expected by
LLaMA-Factory's ``sharegpt`` template:

    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "..."},
        {"role": "assistant", "content": "{...JSON action...}"}
    ]}

A sibling ``dataset_info.json`` registers the file under the name used by the
rendered YAML so LLaMA-Factory picks it up automatically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from evoguard.config import TrainingConfig
from evoguard.utils.logging import get_logger

logger = get_logger("training.sft")


@dataclass
class SFTRunSpec:
    """Result of preparing an SFT run."""

    dataset_path: str            # jsonl written to disk
    dataset_info_path: str       # dataset_info.json next to it
    config_yaml: str             # rendered YAML for llamafactory-cli train
    command: str                 # full shell invocation (dry_run or live)
    adapter_dir: str             # where LoRA weights will land after training


def write_sft_dataset(
    examples: list,
    out_dir: str,
    *,
    dataset_name: str = "evoguard_sft",
) -> tuple[str, str]:
    """Write SFTExamples as a LLaMA-Factory-compatible jsonl + info pair.

    Returns ``(jsonl_path, info_path)``.
    """

    os.makedirs(out_dir, exist_ok=True)
    data_file = os.path.join(out_dir, f"{dataset_name}.jsonl")
    with open(data_file, "w", encoding="utf-8") as f:
        # Each example already exposes .to_llamafactory() producing {messages, meta}.
        for ex in examples:
            payload = ex.to_llamafactory() if hasattr(ex, "to_llamafactory") else ex.to_dict()
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
    info_file = os.path.join(os.path.dirname(out_dir), "dataset_info.json")

    existing: dict[str, dict] = {}
    if os.path.exists(info_file):
        try:
            with open(info_file, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                if isinstance(loaded, dict):
                    existing.update(loaded)
        except json.JSONDecodeError:
            logger.warning("Existing %s was corrupt; overwriting.", info_file)

    rel_datafile = os.path.relpath(
        data_file, start=os.path.dirname(info_file) or "."
    )
    existing[dataset_name] = {
        "file_name": rel_datafile,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return data_file, info_file


def render_lora_sft_config(
    base_model: str,
    dataset_name: str,
    output_dir: str,
    cfg: TrainingConfig,
    *,
    extra_fields: dict | None = None,
) -> str:
    """Render a minimal but complete LLaMA-Factory LoRA-SFT yaml.

    Mirrors the structure of vendored example configs under
    ``LLaMA-Factory/examples/train_lora/`` while exposing EvoGuard-relevant knobs.
    """

    lines: list[str] = []
    def _kv(k, v):
        if isinstance(v, bool):
            return f"{k}: {'true' if v else 'false'}"
        if isinstance(v, float):
            return f"{k}: {v}"
        if isinstance(v, int):
            return f"{k}: {v}"
        if isinstance(v, str):
            v_no_newlines = v.replace("\n", "\\n").replace('"', '\\"')
            return f'{k}: "{v_no_newlines}"'
        if isinstance(v, list):
            inner = ",".join(str(x).replace('"', "") for x in v)
            return f"{k}: [{inner}]"
        raise TypeError(f"unsupported value type for key={k!r}")

    fields_map: dict[str, object] = {}
    fields_map["### model"] = None  # section marker below; we render manually
    body_lines: list[tuple[str, object]] = []

    sections: list[tuple[str, list[tuple[str, object]]]] = [
        ("### model", [("model_name_or_path", base_model), ("trust_remote_code", True)]),
        (
            "### method",
            [
                ("stage", "sft"),
                ("do_train", True),
                ("finetuning_type", "lora"),
                ("lora_rank", cfg.lora_rank),
                ("lora_alpha", cfg.lora_alpha),
                ("lora_dropout", cfg.lora_dropout),
                ("lora_target", cfg.lora_target_modules),
            ],
        ),
        (
            "### dataset",
            [
                ("dataset", dataset_name),
                ("template", "qwen" if "Qwen" in base_model else "default"),
                ("cutoff_len", 4096),
                ("overwrite_cache", True),
                ("preprocessing_num_workers", 4),
            ],
        ),
        (
            "### output",
            [
                ("output_dir", output_dir),
                ("logging_steps", 10),
                ("save_steps", 500),
                ("plot_loss", True),
                ("overwrite_output_dir", True),
                ("report_to", "none"),
            ],
        ),
        (
            "### train",
            [
                ("per_device_train_batch_size", cfg.per_device_batch_size),
                ("gradient_accumulation_steps", cfg.gradient_accumulation),
                ("learning_rate", cfg.sft_learning_rate),
                ("num_train_epochs", cfg.sft_epochs),
                ("lr_scheduler_type", "cosine"),
                ("warmup_ratio", 0.1),
                ("bf16", True),
            ],
        ),
    ]
    del body_lines, fields_map

    for header, kvs in sections:
        lines.append(header)
        for k, v in kvs:
            lines.append(_kv(k, v))
        lines.append("")
    if extra_fields:
        for k, v in extra_fields.items():
            lines.append(f"{_kv(k, v)}")
    text = "\n".join(lines).rstrip() + "\n"
    return text


@dataclass
class SFTPlan:
    spec: SFTRunSpec
    n_examples: int
    launched: bool


def prepare_and_run_sft(
    examples: list,
    exp_rounds_root: str,
    *,
    training_cfg: TrainingConfig,
    round_id_for_label: str = "",
    served_adapter_name_hint: str = "evoguard_defense_v0",
) -> SFTPlan:
    """Materialize datasets/config/command. Launches iff not dry_run.

    Parameters mirror what GRPO also needs:
      - ``exp_rounds_root`` -- directory under which we place ``sft/<round>/``.
      - ``served_adapter_name_hint`` is recorded on the returned plan so that,
        when launched against a real vLLM server, the pipeline knows how to ask
        for this newly-trained adapter by name on subsequent rounds.
    """

    sft_dir = os.path.join(exp_rounds_root, "sft", round_id_for_label or "latest")
    os.makedirs(sft_dir, exist_ok=True)
    adapter_out = os.path.join(sft_dir, "adapter")

    dataset_name = f"evoguard_sft_{round_id_for_label}".strip("_").lower()
    data_jsonl, info_json = write_sft_dataset(examples, os.path.join(sft_dir, "data"), dataset_name=dataset_name)

    yaml_text = render_lora_sft_config(training_cfg.base_model, dataset_name, adapter_out, training_cfg)
    yaml_path = os.path.join(sft_dir, "config.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    lf_cli = "llamafactory-cli"
    cmd = f"{lf_cli} train {yaml_path}"

    spec = SFTRunSpec(
        dataset_path=data_jsonl,
        dataset_info_path=info_json,
        config_yaml=yaml_path,
        command=cmd,
        adapter_dir=adapter_out,
    )

    launched = False
    if not training_cfg.dry_run and len(examples) > 0:
        logger.info("[SFT] launching (%d examples): %s", len(examples), cmd)
        rc = os.system(cmd)
        if rc != 0:
            raise RuntimeError(f"SFT training failed with exit code {rc}")
        launched = True
    elif training_cfg.enabled:
        logger.info("[SFT] dry-run prepared at %s ; command=%s", sft_dir, cmd)
    return SFTPlan(spec=spec, n_examples=len(examples), launched=launched)


# Public surface ----------------------------------------------------------- #
__all__ = ["SFTRunSpec", "write_sft_dataset", "render_lora_sft_config", "prepare_and_run_sft", "SFTPlan"]
