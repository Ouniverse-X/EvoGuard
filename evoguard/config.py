"""Configuration objects for the EvoGuard co-evolution pipeline.

Configuration is expressed as nested dataclasses so it is both type-checked in
code and (de)serializable from YAML/JSON. ``ExperimentConfig.from_file`` loads a
YAML or JSON file and overlays it on the defaults, so partial configs are valid.

The default values encode the hyper-parameters from ``docs/plan.md``
(population ``N=50``, offspring ``M=45``, elites ``E=5``, tournament ``k=3``,
termination window ``K=5`` and success-rate threshold ``epsilon``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional, get_type_hints


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    """Connection + sampling settings for an LLM endpoint.

    ``backend`` selects the client implementation:

    * ``"openai"`` -- an OpenAI-compatible HTTP endpoint (this is what a local
      vLLM ``openai.api_server`` exposes). ``base_url`` / ``api_key`` / ``model``
      identify it.
    * ``"qianfan"``-- Baidu QianFan v2 gateway at
      https://qianfan.baidubce.com/v2/chat/completions. Authenticates via
      custom appid/Bearer headers; credentials read from env vars
      ``EVOGUARD_QIANFAN_APPID`` / ``EVOGUARD_QIANFAN_TOKEN``, or from a
      pipe-separated literal in :attr:`api_key` (format ``"<appid>|<bce-v3/...>"``).
      Model field selects e.g. ``glm-5`` or ``glm-5.2``.
    * ``"mock"``   -- a deterministic offline client used for smoke tests.
    """

    backend: str = "openai"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    base_url: Optional[str] = None  # e.g. "http://localhost:8000/v1" for vLLM
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    timeout: float = 120.0
    max_retries: int = 3
    # Optional LoRA adapter name to request from the vLLM server for this role.
    lora_adapter: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseConfig:
    """Defense agent settings."""

    llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(temperature=0.2, max_tokens=768)
    )
    max_turns: int = 12
    system_prompt: Optional[str] = None


@dataclass
class AttackerConfig:
    """Attacker LLM + genetic algorithm settings (``docs/plan.md``)."""

    llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(temperature=0.9, max_tokens=768)
    )
    population_size: int = 50            # N
    offspring_size: int = 45             # M
    elite_size: int = 5                  # E
    tournament_k: int = 3                # k
    crossover_rate: float = 0.9
    mutation_rate: float = 0.5
    # Diversity penalty: how strongly to discount individuals whose injection
    # position / method are highly similar to already-selected ones.
    diversity_penalty: float = 0.5
    diversity_position_window: int = 1   # turns within which positions count as "close"
    random_seed: int = 0


@dataclass
class EnvConfig:
    """Environment settings."""

    # Which dataset env to build: "agentdojo" | "agentharm".
    dataset: str = "agentdojo"
    # Data roots (relative to repo root unless absolute).
    data_root: str = "data"
    # AgentDojo suite selection (empty => all discovered suites).
    suites: list[str] = field(default_factory=list)
    # Cap the number of tasks (0 => no cap); useful for smoke runs.
    max_tasks: int = 0
    # LLM that simulates tool execution.
    tool_executor_llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(temperature=0.0, max_tokens=512)
    )
    # Whether the injection oracle should judge attack success with an LLM.
    judge_llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(temperature=0.0, max_tokens=256)
    )


@dataclass
class ProcessConfig:
    """Signal-computation settings (``docs/plan.md``)."""

    # 超过该阈值，认为两个工具调用不再对齐，拐点发生.
    divergence_threshold: float = 0.5
    # Whether to normalize delta by clean-trajectory length (else by max length).
    normalize_by: str = "clean_length"  # "clean_length" | "max_length"


@dataclass
class TrainingConfig:
    """Defender training settings (SFT cold-start + GRPO on LoRA)."""

    enabled: bool = True
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    method: str = "sft_then_grpo"  # "sft" | "grpo" | "sft_then_grpo"
    # LoRA hyper-parameters.
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # Framework roots (vendored under evoguard/training).
    llamafactory_root: str = "evoguard/training/LLaMA-Factory"
    verl_root: str = "evoguard/training/AEPO/verl_aepo_entropy"
    sft_epochs: float = 2.0
    sft_learning_rate: float = 1.0e-4
    grpo_learning_rate: float = 1.0e-6
    per_device_batch_size: int = 1
    gradient_accumulation: int = 8
    # If True, only render configs/datasets and print commands without launching.
    dry_run: bool = True


@dataclass
class PipelineConfig:
    """Top-level co-evolution loop settings."""

    max_rounds: int = 20
    # Termination: stop after K consecutive rounds with ASR < epsilon on val set.
    patience_rounds: int = 5     # K
    asr_threshold: float = 0.05  # epsilon
    # Also stop if a round produces zero successful (B) attacks.
    stop_on_zero_success: bool = True
    validation_fraction: float = 0.2

    # ------------------------------------------------------------------ #
    # Concurrency knobs                                                  #
    # ------------------------------------------------------------------ #
    # Both layers can fan out independently because individual LLM calls
    # block mostly on network IO rather than CPU/GPU compute locally --
    # the dominant cost being remote GLM thinking-model RTT (~25s/call).
    #
    # Effective peak outbound HTTP-in-flight ≈ task × attack product,
    # bounded externally by your paid-endpoint rate quota. For Baidu
    # Qianfan default tier (RPM=60 / TPM=250K), a combined ceiling around
    # ~16 simultaneous requests leaves comfortable margin below throttle
    # threshold while saturating available throughput.
    #
    # Set either field <=1 to disable that layer's parallelism entirely.
    task_concurrency: int = 4
    attack_concurrency: int = 4


@dataclass
class ExperimentConfig:
    """Root configuration for a full EvoGuard experiment."""

    name: str = "evoguard-exp"
    rounds_dir: str = "rounds"
    seed: int = 0
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # ---- (de)serialization ------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return _dataclass_from_dict(cls, d)

    @classmethod
    def from_file(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if path.endswith((".yaml", ".yml")):
            import yaml  # local import keeps yaml optional at import time

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text) if text.strip() else {}
        return cls.from_dict(data)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if path.endswith((".yaml", ".yml")):
                import yaml

                yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
            else:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Generic dataclass <-> dict helpers (overlay-friendly)
# --------------------------------------------------------------------------- #
def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def _dataclass_from_dict(cls: Any, data: dict[str, Any]) -> Any:
    """Build a (possibly nested) dataclass, overlaying ``data`` on defaults.

    Unknown keys are ignored; missing keys keep their default. Nested dataclass
    fields recurse so partial YAML configs work.

    Because this module uses ``from __future__ import annotations``, every
    field type is stored as a *string* annotation; we resolve them once via
    :func:`typing.get_type_hints` so :func:`is_dataclass` sees the actual class
    object instead of a string.
    """

    if not isinstance(data, dict):
        return data
    try:
        resolved_hints = get_type_hints(cls)
    except Exception:
        # Forward-refs that can't be evaluated fall back to raw string hints.
        resolved_hints = {f.name: f.type for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = resolved_hints.get(f.name, f.type)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _dataclass_from_dict(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
