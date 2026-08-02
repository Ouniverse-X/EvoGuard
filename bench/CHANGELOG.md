# Bench Changelog

## bench-v2-dev-init — UNRELEASED

Initial migration toward `bench_v2` schema begun. Pending fulfillment of n≥100-per-bucket gate thresholds before tagging minor-release git refs.

### Added
- Canonical technique-family taxonomy (five-axis classifier producing stable tech_<hex12> IDs) replacing noisy ad-hoc method tags inherited from preliminary_bench_v1.
- Strategy-① Turn-Shift Replant Validation synthesizer pipeline capable of enriching starved high-Δ buckets (especially d3/d4) cheaply leveraging existing mined successes shifted geometrically along clean trajectories.
- Power-aware three-stage release gate preventing publication of undersized bucket configurations reproducing today's null-result pilot lesson structurally.

### Changed
- Sampling configuration `cap_per_bucket` lifted 40 → 110 granting headroom above n≥100 floor.
- Domain restriction narrowed exclusively to AgentDojo `workspace` subset per owner direction dated 2026-08-02.
- Schema version identifier updated `preliminary_bench_v1` → `bench_v2`; backward-compat adapter preserves legacy `bench/corpus_d<n>.jsonl` path resolution semantically redirecting to new `bench/scenarios/bucket_d<n>.jsonl` locations.

### Deprecated
- `bench/ipi_library/v1/library_*.jsonl` retained read-only heritage assets superseded by `bench/techniques/registry.jsonl`.

### Removed
Nothing yet pending deprecation grace period expiration.

### Fixed
Addressed severe under-powering of d3/d4 cells responsible for inconclusive alertness-entropy pilot findings reported earlier today.

### Security
No security-sensitive changes affecting runtime trust posture.
