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
    * ``"llamacpp"`` -- a local llama.cpp ``llama-server`` serving a GGUF model
      (e.g. GLM-5.2 on CPU). OpenAI-compatible wire format, but strips GLM
      ``<think>`` blocks and is limited by decode slots rather than a remote RPM
      quota. Used for the attacker role to remove the gateway rate ceiling.
    * ``"mock"``   -- a deterministic offline client used for smoke tests.
    """

    backend: str = "openai"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    base_url: Optional[str] = None  # e.g. "http://localhost:8000/v1" for vLLM
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    timeout: float = 300.0
    max_retries: int = 3
    # Optional LoRA adapter name to request from the vLLM server for this role.
    lora_adapter: Optional[str] = None
    # Whether to enable model-side chain-of-thought ("thinking mode") for backends
    # that support it (e.g. QianFan GLM-5 / glm-5.2). Thinking models consume hidden
    # CoT tokens BEFORE emitting visible content; disabling it dramatically reduces
    # finish_reason="length" truncations observed during real rollouts.
    # Backend clients honor this best-effort: unknown backends silently ignore False.
    enable_thinking: bool = True
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
    # Crowding-based selection (replaces pure tournament when > 1): each tournament
    # bracket may contain at most ``crowding_factor`` individuals sharing the same
    # ``method`` label, forcing exploration across method niches.
    crowding_factor: int = 2
    # Random-immigrant injection rate in [0, 1]: fraction of worst-fitness
    # individuals replaced with freshly-seeded random genomes per generation.
    # Activates when EITHER mean-best fitness drops by ``fitness_drop_threshold``
    # between consecutive generations OR ``immigrant_stagnation_gens`` elapsed
    # without any elite improvement -- prevents premature convergence like the
    # r5 collapse observed in evoguard_agentdojo_full run.
    immigrant_injection_rate: float = 0.2
    fitness_drop_threshold: float = 0.5
    # Stagnation-based alternative trigger for immigrant injection. Fires when
    # this many consecutive generations pass WITHOUT a strict increase in the
    # population's best fitness -- catches plateau-style stalls where best_fit
    # stays at zero forever so the relative-drop rule above can never fire.
    # Set to 0 to disable.
    immigrant_stagnation_gens: int = 4
    # Adaptive-mutation toggle + cap. When enabled, mutation_rate scales UP
    # multiplicatively within one generation whenever phenotypic variance of
    # current-population fitness collapses near zero; bounded by max.
    adaptive_mutation_enabled: bool = True
    mutation_rate_min: float = 0.05   # absolute lower bound post-scaling.
    mutation_rate_max: float = 0.85    # upper bound preventing pure-random walk.
    behavioral_archive_size: int = 20  # rolling window length for novelty bonus.
    novelty_bonus_weight: float = 0.03 # additive boost caps raw_fitness range shift.
    random_seed: int = 0

    # ------------------------------------------------------------------ #
    # Search-backend selector + MCTS-specific knobs                     #
    # (see docs/mcts_attacker_design.md). All mcts_* fields are ignored  #
    # when search_method == "ga"; they carry defaults that make the     #
    # "mcts_delta" backend work out-of-the-box without extra yaml.       #
    # ------------------------------------------------------------------ #
    # Selects which attacker backend to instantiate via build_attacker():
    #   "ga"         -> GeneticAttacker (legacy, default for backward compat)
    #   "mcts_delta" -> DeltaGuidedMCTSAttacker
    search_method: str = "ga"
    mcts_ucb_c: float = 1.414           # classic UCB exploration coefficient (= sqrt(2))
    mcts_lambda_delta: float = 0.6      # weight of the delta-potential term in selection score
    mcts_failure_credit_eps: float = 0.03  # fraction of late-caught-failure tau converted into partial credit
    mcts_tau_window_size: int = 8        # sliding window of recent C-class taus kept per node


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
    # Optional fallback judge for benign-task completion scoring when the
    # native AgentDojo verifier is unavailable (no matching user-task class or
    # environment replay fails). When omitted, env falls back to a no-op scorer.
    utility_judge_llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(temperature=0.0, max_tokens=512)
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
    # When ``method == "native_sft"`` we bypass both vendored frameworks entirely
    # and use a thin in-process wrapper around HuggingFace + PEFT + TRL's
    # :class:`SFTTrainer`. This sidesteps dependency conflicts between LF/verl's
    # pinned versions (numpy<2, peft<=0.15) and what is actually installed.
    # Incremental rounds load this round's previous-adapter as warm-start rather
    # than re-doing cold-start each time, mirroring plan.md intent without the
    # full GRPO online-RL machinery (deferred until GRPO integration stabilizes).
    use_native_trainer: bool = False
    # Comma-separated GPU indices passed through as CUDA_VISIBLE_DEVICES while
    # training; empty string means "use whatever the parent process sees".
    cuda_visible_devices: str = ""
    # Per-round cap on number of incremental SFT steps beyond which we stop even
    # if max_steps not yet reached -- prevents runaway long rounds on noisy data.
    # 0 disables the cap (use sft_epochs alone).
    native_max_steps_per_round: int = 0
    sft_epochs: float = 2.0
    sft_learning_rate: float = 1.0e-4
    grpo_learning_rate: float = 1.0e-6
    per_device_batch_size: int = 1
    gradient_accumulation: int = 8

    # ---- Native GRPO hyperparameters (spec §5) ----------------------- #
    # Used by evoguard/training/native_grpo_runner.py when method is one of
    # {"native_grpo","sft_then_native_grpo"}. All defaults chosen to match
    # standard RLHF recipe ranges; users override via YAML as needed.
    grpo_beta: float = 0.04                  # KL coefficient toward reference policy β.
    grpo_group_size_g: int = 8               # G completions sampled per prompt for group-relative advantage.
    grpo_clip_epsilon: float = 0.20          # PPO clip range ε.
    grpo_rollout_temperature: float = 0.90   # sampling temperature during inner-loop generation (>defense temp encourages exploration).
    grpo_max_prompts_per_round: int = 32     # cap prompts fed to trainer each round bounds runtime.
    # 方案乙 (spec §3 explicit Δ↔advantage coupling): multiplicative curriculum factor applied
    # DIRECTLY on top of group-relative advantages before PPO ratio computation:
    #     Ã⁽ᵍᵖ⁾ = (1 + λ·δ_p) · A⁽ᵍᵖ⁾
    # where δ_p is the originating record's normalized Δ carried per-prompt via PromptMeta.
    # Default λ=0.0 reproduces legacy equal-weight behaviour bit-for-bit so existing yamls keep working;
    # positive values amplify gradients on latent-attack prompts without touching reward scale itself.
    # Since S1 (2026-08-21) deleted the inline ``r_early`` term this is the ONLY
    # place Δ still influences training, so a run intending Δ-monotonicity MUST
    # set λ>0. Keeping Δ out of the scalar avoids the dilution that crippled
    # r_early: it peaked at +0.50 against a ±2.0 safety term, ~0.4% of the summed
    # variance.
    grpo_advantage_curriculum_lambda: float = 0.0
    # Fraction of ``grpo_max_prompts_per_round`` reserved for BENIGN prompts cut
    # out of clean (A) trajectories. They run the SAME formula as attacked rows
    # with r_safety pinned to a constant, so all their signal comes from the
    # judged r_progress term in grpo_reward.py.
    # 0.0 = attacked-only, i.e. legacy behaviour bit-for-bit. Without benign
    # rows every prompt in the batch pays for blocking, all G group siblings
    # converge on the same blocking behaviour, reward std collapses to zero and
    # nothing in the gradient distinguishes "blocked and still served the user"
    # from "blocked and gave up" -- the 一刀切 failure. 0.5 gives roughly one
    # benign prompt per attacked one.
    grpo_clean_prompt_ratio: float = 0.0

    # ---- SFT dataset quality gate (item D1, 2026-08-20) ------------------- #
    # Utility below which a source trajectory is too poor to imitate.
    # 0.0 = no filtering (legacy behaviour bit-for-bit). At 0.5, a clean (A)
    # rollout that failed its own task is dropped, and an attack-fail (C)
    # rollout that blocked the attack but then failed the task is truncated to
    # its payload-free prefix -- those post-injection steps are literally the
    # "attack blocked, benign task not delivered" behaviour counted by
    # ``utils.metrics.RoundMetrics.blocked_unfinished_rate``, and they were 37.5%
    # of C records / ~28% of all SFT rows on r0.
    sft_min_source_utility: float = 0.0

    # ---- SFT two-class recipe (2026-08-20) -------------------------------- #
    # True = keep only the two observed behaviours we want at cold start:
    #   class 1: clean (A) rollouts that completed the task;
    #   class 2: attack-fail (C) rollouts that completed the task, i.e. resisted
    #            the bait AND still delivered.
    # Successful attacks (B) are dropped, which removes every hand-written
    # corrective-refusal template from the dataset -- the r6 mode-collapse
    # vector, and the only source of refusal-shaped targets. Detection is then
    # delegated entirely to GRPO. Measured on r0: only 0.6% of eligible C
    # rollouts contain any detection wording, so class 2 teaches "ignore the
    # bait and finish the job" rather than "announce the bait".
    sft_two_class: bool = False
    # 0 = unlimited. Caps how many source rollouts one task may contribute, so a
    # heavily-probed task cannot dominate. On r0 the per-task C count ranged
    # 1..15; a cap of 4 flattens the row distribution without losing tasks.
    sft_max_records_per_task: int = 0

    # ---- Corrective-row hard cap (plan 乙, 2026-08-21) -------------------- #
    # Upper bound on the fraction of SFT rows whose TARGET is a refusal-shaped
    # ``corrective_refusal`` example. 0.0 = no cap (legacy). This exists so
    # ``sft_two_class`` can go back to False -- reinstating detection supervision
    # that two-class mode removed entirely -- WITHOUT reinstating the r6
    # mode-collapse, which came from that supervision being an unbounded share of
    # the corpus (45.7% of decoded steps parroted one corrective sentence).
    # B records are admitted in a deterministic order until the next one would
    # push the share over this bound, so rebuilding a round is byte-identical.
    sft_max_corrective_share: float = 0.0

    # If True, only run SFT cold-start on round_0; subsequent rounds reuse the
    # existing adapter and apply GRPO incrementally. Saves ~30-40 min/round of
    # redundant cold-start compute observed in evoguard_agentdojo_full run.
    sft_coldstart_only_round_zero: bool = False
    # Minimum number of NEW successful B-trajectories accumulated since the last
    # training step required to actually fire another GRPO update. Setting this
    # above zero avoids retraining when the latest round produced no novel signal.
    grpo_min_new_successes: int = 5
    # If True, only render configs/datasets and print commands without launching.
    dry_run: bool = True

    # ------------------------------------------------------------------ #
    # Pre-training LoRA layer probe (one-shot static version)            #
    # (see docs/delta_signal_essence.md §2.4 causal-chain relay stations)#
    #                                                                    #
    # When ``lora_probe_enabled`` is true, an offline pre-pass scores    #
    # every Transformer block's sensitivity to injection via paired      #
    # clean/attacked forward passes and writes a JSON artifact at        #
    # ``lora_probe_artifact_path`` whose ``recommended_target_modules``  #
    # field overrides the static default target_modules above when the   #
    # native trainer loads a fresh cold-start adapter in round r0.       #
    # Subsequent rounds inherit those layers automatically via PEFT      #
    # adapter_config.json reuse so no per-round re-probing is needed     #
    # for this static version.                                           #
    # ------------------------------------------------------------------ #
    lora_probe_enabled: bool = False
    # Scoring algorithm selector; MVP ships "attn_kl" only.
    # "grad_attr"/"act_patch" reserved as interface placeholders only
    # and are NOT implemented by probes/sensitivity.py yet -- selecting them
    # raises NotImplementedError at call time rather than silently falling back.
    lora_probe_method: str = "attn_kl"
    lora_probe_top_k_blocks: int = 8           # how many Transformer blocks enter target set.
    lora_probe_artifact_path: str = ""         # written by run_lora_probe.sh entry point;
                                               # read by native_runner.py to override defaults.
    lora_probe_max_pairs: int = 80             # stratified-per-domain sampling size cap.


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
    # Optional sub-directory inserted between ``<exp_dir>/evo_data`` and the
    # three bucket names (clean / attack_success_B / attack_failure_C) so that
    # parallel experiments sharing one repo root can namespace their exports
    # without colliding. Empty string preserves legacy flat layout.
    evo_data_subdir: str = ""
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
