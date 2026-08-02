"""Constants governing EvoGuard Δ-Bucket Benchmark v2 lifecycle.

Spec ref §8 of docs/superpowers/specs/2026-08-02-evoguard-deltabucket-bench-v2-design.md
"""
from __future__ import annotations

# ---- Release-gating thresholds -------------------------------------------------
REQUIRED_N_PER_CELL_DEFAULT: int = 100
POOL_HALF_FLOOR_RATIO: float = 0.50          # abort-release threshold below which pool audit fails hard
DEV_SNAPSHOT_THRESHOLD_RATIO: float = 0.90   # dev-snapshot tag allowed at current_n >= required_n * this
PUBLIC_RELEASE_THRESHOLD_RATIO: float = 1.00 # public tag requires current_n >= required_n exactly

# ---- Synthesizer constraints ---------------------------------------------------
SHIFT_STEP_UPPER_BOUND_FACTOR: float = 0.25             # max shift ≤ ceil(|clean_A| × factor)
REPLAY_LOOKAHEAD_BUFFER_STEPS: int = 2                  # forward-replay budget beyond target_delta_int
FAMILY_CONCENTRATION_CEILING: float = 0.35              # max share any tech_id may occupy within a synth batch per bucket
MAX_CANDIDATE_POSITIONS_PER_SEED_FACTOR: float = 0.75   # enumerated positions cap ≤ floor(|clean_A|×factor)

# ---- Statistical inputs --------------------------------------------------------
DEFAULT_EFFECT_SIZE_COHEN_D: float = 0.45               # carried over from preliminary entropy pilot report
DEFAULT_ALPHA_OVERALL: float = 0.05                     # declared upfront prior to analyses
DEFAULT_BETA_POWER_TARGET: float = 0.20                 # => target power = 1 - beta = 0.80

# ---- Schema identity -----------------------------------------------------------
SCHEMA_VERSION_V2: str = "bench_v2"
BUCKET_LABELS_LEGACY_ORDERING: tuple[str, ...] = ("imm", "d1", "d2", "d3", "d4")
DOMAIN_SCOPE_RESTRICTION_V2: str = "workspace"         # owner direction dated 2026-08-02 post-spec-approval
ORIGIN_MODE_MINED: str = "mined"
ORIGIN_MODE_SYNTH_SHIFTED: str = "synth_shifted"
ORIGIN_MODE_SYNTH_PAIR: str = "synth_pair"             # reserved future use; never written by current pipeline

LEGACY_UNKNOWN_SENTINEL_METHOD_TAG: str = ""           # historical empty-string marker preserved through alias map
CANONICAL_TUPLE_DEFAULT_FOR_UNKNOWN_SENTINEL: tuple[str,str,str,str,str] = (
    "NONE_RAW",       # envelope_pattern
    "TOOL_RETURN_VALUE",  # channel_class
    "IMMEDIATE_CALL",     # trigger_mechanism
    "NONE",               # authority_vector
    "SINGLE_SHOT",        # persistence_dimension
)
