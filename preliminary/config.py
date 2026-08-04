"""Configuration loader for the preliminary-entropy experiment.

Mirrors the dataclass-based YAML/JSON loader pattern used in
``evoguard/config.py`` but kept deliberately narrow: only the fields needed
by the entropy-measurement pipeline are modelled. Unknown keys raise so the
experimenter notices typos rather than silently getting defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace, fields as _dc_fields
from typing import Any

try:
    import yaml  # PyYAML
except ImportError as exc:  # pragma: no cover - hard dep at runtime
    raise SystemExit(
        "PyYAML is required to read configs/preliminary_entropy.yaml. "
        "Install with: pip install pyyaml"
    ) from exc


# --------------------------------------------------------------------------- #
# Section dataclasses — each carries a ``_coerce`` classmethod that normalises
# YAML-loaded plain dicts/lists into typed instances (e.g. list -> tuple[str,...]).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key_env_var: str = "OPENAI_API_KEY"
    model_id: str = "Qwen2.5-7B-Instruct"
    require_no_lora_loaded: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_logprobs: int = 20
    extra_stop_sequences: tuple[str, ...] = ("<|im_end|>", "</answer>")
    seed_passthrough: int | None = None  # resolved to experiment.seed if left None at load time

    @classmethod
    def _coerce(cls, raw: dict[str, Any]) -> "GenerationConfig":
        d = dict(raw)
        v = d.get("extra_stop_sequences")
        if isinstance(v, (list, tuple)):
            d["extra_stop_sequences"] = tuple(str(x) for x in v)
        return cls(**d)


@dataclass(frozen=True)
class MiningConfig:
    source_experiments: tuple[str, ...] = (
        "evoguard_agentdojo_full_v6_mcts",
        "evoguard_agentdojo_full_v7_integrated",
        "agentdojo_full_mcts_cont",
    )
    rounds_root_relative_to_repo: str = "rounds"
    require_outcome_success: bool = True
    require_numeric_delta: bool = True
    cap_per_bucket: int = 40
    max_domain_fraction_in_bucket: float = 0.50
    buckets: tuple[str, ...] = ("imm", "d1", "d2", "d3", "d4")
    sampling_seed: int | None = None  # resolved to experiment.seed if None
    bench_output_dir_relative_to_repo: str = "bench"
    domain_scope_restriction: str | None = None  # bench_v2: restrict to one domain (e.g. "workspace"); miner itself only balances via max_domain_fraction_in_bucket, hard filter applied at migrate stage

    @classmethod
    def _coerce(cls, raw: dict[str, Any]) -> "MiningConfig":
        d = dict(raw)
        for k in ("source_experiments", "buckets"):
            v = d.get(k)
            if isinstance(v, (list, tuple)):
                d[k] = tuple(str(x) for x in v)
        return cls(**d)


@dataclass(frozen=True)
class EntropyConfig:
    granularities_reported: tuple[str, ...] = (
        "token_mean_full_response",
        "first_k_mean_at_K_eq_64",
        "normalized_token_mean",
    )
    aux_columns_also_emitted: tuple[str, ...] = (
        "token_std_full_response",
        "argmax_mass_mean",
        "whitespace_filtered_variant",
    )
    effective_K_for_normalization_floor: int = 20
    first_k_window_size: int = 64

    @classmethod
    def _coerce(cls, raw: dict[str, Any]) -> "EntropyConfig":
        d = dict(raw)
        for k in ("granularities_reported", "aux_columns_also_emitted"):
            v = d.get(k)
            if isinstance(v, (list, tuple)):
                d[k] = tuple(str(x) for x in v)
        return cls(**d)


@dataclass(frozen=True)
class StatsConfig:
    bootstrap_iters: int = 10000
    ci_level: float = 0.95
    alpha_overall: float = 0.05
    planned_pairwise_comparisons: tuple[tuple[str, str], ...] = (
        ("imm", "d1"),
        ("d1", "d2"),
        ("d2", "d3"),
        ("d3", "d4"),
    )
    bonferroni_correction_count: int = 4
    random_state_for_resampling: int | None = None  # resolved to experiment.seed if None

    @classmethod
    def _coerce(cls, raw: dict[str, Any]) -> "StatsConfig":
        d = dict(raw)
        v = d.get("planned_pairwise_comparisons")
        if isinstance(v, (list, tuple)):
            coerced: list[tuple[str, str]] = []
            for pair in v:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(
                        f"planned_pairwise_comparisons entries must be [a,b] pairs; got {pair!r}"
                    )
                coerced.append((str(pair[0]), str(pair[1])))
            d["planned_pairwise_comparisons"] = tuple(coerced)
        return cls(**d)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config consumed by ``preliminary.cli``."""

    experiment_name: str = "preliminary_entropy_v1"
    seed: int = 42
    repo_root_override: str | None = None  # auto-detected when None; see resolve_repo_root()

    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)

    output_dir_template: str = "rounds/_preliminary/{run_timestamp}"
    dry_run: bool = False
    logging_level: str = "INFO"

    # ----- loaders -------------------------------------------------------- #
    @classmethod
    def from_file(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        seed = int(d.get("seed", cls.__dataclass_fields__["seed"].default))
        d = _resolve_seed_placeholders(d, seed)

        known_top_keys = set(_f.name for _f in _dc_fields(cls))
        unknown = [k for k in d.keys() if k not in known_top_keys]
        if unknown:
            raise ValueError(
                f"Unknown top-level config key(s): {unknown}. "
                f"Allowed: {sorted(known_top_keys)}"
            )

        model_dct = _section(d, "Model") or {}
        gen_dct = _section(d, "Generation") or {}
        mining_dct = _section(d, "Mining") or {}
        ent_dct = _section(d, "Entropy") or {}
        stats_dct = _section(d, "Stats") or {}

        # Resolve seed-derived defaults inline so we don't need a fragile post-init rebuild.
        if gen_dct.get("seed_passthrough", None) is None and "seed_passthrough" not in gen_dct:
            pass  # leave default sentinel None handled below via explicit fill
        elif gen_dct.get("seed_passthrough") is None:
            del gen_dct["seed_passthrough"]  # let default kick in then override explicitly below
        mining_seed_default_set = mining_dct.get("sampling_seed") is not None or "sampling_seed" in mining_dct
        stats_rsrs_explicitly_none = "random_state_for_resampling" in stats_dct and stats_dct["random_state_for_resampling"] is None

        model_cfg = ModelConfig(**model_dct) if model_dct else ModelConfig()
        gen_cfg_raw = GenerationConfig._coerce(gen_dct) if gen_dct else GenerationConfig()
        min_cfg_raw = MiningConfig._coerce(mining_dct) if mining_dct else MiningConfig()
        ent_cfg = EntropyConfig._coerce(ent_dct) if ent_dct else EntropyConfig()
        stt_cfg_raw = StatsConfig._coerce(stats_dct) if stats_dct else StatsConfig()

        # Apply seed-default substitutions.
        new_gen = replace(
            gen_cfg_raw,
            seed_passthrough=int(gen_cfg_raw.seed_passthrough)
            if gen_cfg_raw.seed_passthrough is not None else seed,
        )
        new_min = replace(
            min_cfg_raw,
            sampling_seed=int(min_cfg_raw.sampling_seed)
            if min_cfg_raw.sampling_seed is not None else seed,
        ) if min_cfg_raw.sampling_seed is not None or mining_seed_default_set else \
            replace(min_cfg_raw, sampling_seed=seed)
        new_stt = replace(
            stt_cfg_raw,
            random_state_for_resampling=int(stt_cfg_raw.random_state_for_resampling)
            if stt_cfg_raw.random_state_for_resampling is not None else seed,
        )

        return cls(
            experiment_name=str(
                d.get("experiment_name", cls.__dataclass_fields__["experiment_name"].default)
            ),
            seed=seed,
            repo_root_override=d.get("repo_root_override"),
            output_dir_template=str(
                d.get("output_dir_template", cls.__dataclass_fields__["output_dir_template"].default)
            ),
            dry_run=bool(d.get("dry_run", False)),
            logging_level=str(d.get("logging_level", "INFO")),
            model=model_cfg,
            generation=new_gen,
            mining=new_min,
            entropy=ent_cfg,
            stats=new_stt,
        )

    # ----- path helpers --------------------------------------------------- #
    def resolve_repo_root(self) -> str:
        if self.repo_root_override:
            return self.repo_root_override
        here = os.path.abspath(os.path.dirname(__file__))
        for parent in (here, *iter(_parents(here))):
            if all(os.path.isdir(os.path.join(parent, m)) for m in ("evoguard", "data")) \
               or os.path.isfile(os.path.join(parent, "CLAUDE.md")):
                return parent
        return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def _section(d: dict[str, Any], dc_basename_lower_first_letter: str) -> dict[str, Any]:
    """Fetch sub-dict under either PascalCase section name ('Mining') or lowercased form.

    Returns empty dict when section absent. Caller validates keys against target dataclass fields.
    """
    candidates = [
        dc_basename_lower_first_letter.lower(),   # 'mining'
        dc_basename_lower_first_letter.capitalize(),  # 'Mining'
    ]
    out: dict[str, Any] = {}
    for cand in candidates:
        if cand in d and isinstance(d[cand], dict):
            merged = dict(out); merged.update(d[cand])
            out = merged
    return out


def _parents(path: str):
    prev = None
    while prev != path:
        yield path
        prev = path
        path = os.path.dirname(path)


def _resolve_seed_placeholders(d: dict[str, Any], seed: int) -> dict[str, Any]:
    """Replace any string value of shape '${experiment.seed}' with the integer seed.

    Recurses through nested dicts/lists/tuples. Only exact-match strings are replaced;
    partial mid-string interpolation is intentionally unsupported to keep semantics explicit.
    """

    sentinel = "${experiment.seed}"

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return seed if v.strip() == sentinel else v
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, tuple):
            return tuple(walk(x) for x in v)
        return v

    return walk(d)


def load_config(path_or_obj: "str | ExperimentConfig") -> ExperimentConfig:
    """Convenience helper accepting either an already-built config object or a YAML path."""
    if isinstance(path_or_obj, ExperimentConfig):
        return path_or_obj
    return ExperimentConfig.from_file(str(path_or_obj))


# --------------------------------------------------------------------------- #
# Bucket label <-> ordinal level helpers shared across modules
# --------------------------------------------------------------------------- #
BUCKET_ORDINALS: dict[str, int] = {"imm": 0, "d1": 1, "d2": 2, "d3": 3, "d4": 4}


def bucket_label_from_delta(delta_value: int | float | None) -> str | None:
    """Map numeric Delta onto one of five canonical bucket labels.

    Returns None when input missing OR out of scope (>4). Negative-or-zero deltas collapse into 'imm'.
    """

    if delta_value is None:
        return None
    try:
        di = int(delta_value)
    except (TypeError, ValueError):
        return None
    if di > 4:
        return None
    return "imm" if di <= 0 else f"d{di}"


def ordinal_level(bucket_label: str) -> int:
    try:
        return BUCKET_ORDINALS[bucket_label]
    except KeyError as exc:
        raise ValueError(
            f"Unknown bucket label {bucket_label!r}; expected one of "
            f"{sorted(BUCKET_ORDINALS)}"
        ) from exc
