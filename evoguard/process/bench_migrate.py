"""One-shot migration transforming preliminary_bench_v1 corpus into bench_v2 layout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evoguard.process.bench_constants import (
    BUCKET_LABELS_LEGACY_ORDERING,
    DOMAIN_SCOPE_RESTRICTION_V2,
    LEGACY_UNKNOWN_SENTINEL_METHOD_TAG,
    SCHEMA_VERSION_V2,
)
from evoguard.process.bench_schema import ScenarioRecordV2, SignalsRef
from evoguard.process.bench_taxonomy import AxisTuple, classify_method_tag, record_alias_entry


_V1_CORPUS_FILENAME_TEMPLATE = "{bucket}.jsonl"
_V1_CORPUS_PREFIXED_TEMPLATE = "corpus_{bucket}.jsonl"


def _v1_corpus_path(src_root: Path, bucket: str) -> Path:
    prefixed = src_root / _V1_CORPUS_PREFIXED_TEMPLATE.format(bucket=bucket)
    bare = src_root / _V1_CORPUS_FILENAME_TEMPLATE.format(bucket=bucket)
    return prefixed if prefixed.exists() else bare


def _hash_short(text: str, nchars: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:nchars]


def _build_toolkit_signature(prefix_actions: list[dict[str, Any]]) -> str:
    names: list[str] = []
    for act in prefix_actions:
        tc = act.get("tool_call") or {}
        nm = tc.get("name")
        if isinstance(nm, str):
            names.append(nm)
    seen: set[str] = set()
    ordered = [n for n in names if not (n in seen or seen.add(n))]
    return "|".join(ordered)


def migrate(*, src_dir: str, dst_dir: str,
            canonical_aliases_out: str) -> dict[str, Any]:
    """Transform preliminary_bench_v1 corpora into bench_v2 directory structure.

    Returns summary dict tracking counts useful for asserting zero-orphan guarantee.
    """
    # refactored-for-clarity-from-spec-original-preserving-semantics:
    # dropped unused ``import shutil`` and ``Iterable`` from spec-original import list;
    # both were carried forward verbatim by the template generator without any usage site.
    src_root = Path(src_dir); dst_root = Path(dst_dir)
    aliases_path = Path(canonical_aliases_out)
    scen_dir = dst_root / "scenarios"
    tech_dir = dst_root / "techniques"
    diag_dir = dst_root / "diagnostics"
    for p in (scen_dir, tech_dir, diag_dir, scen_dir / "_synthetic"):
        p.mkdir(parents=True, exist_ok=True)
    aliases_path.parent.mkdir(parents=True, exist_ok=True)
    aliases_path.unlink(missing_ok=True)

    total_processed = 0; orphan_count = 0; skipped_non_workspace = 0
    registry_acc: dict[str, AxisTuple] = {}

    for bucket in BUCKET_LABELS_LEGACY_ORDERING:
        src_path = _v1_corpus_path(src_root, bucket)
        if not src_path.exists():
            continue
        dst_path = scen_dir / f"bucket_{bucket}.jsonl"
        with open(dst_path, "w", encoding="utf-8") as wfh:
            with open(src_path, "r", encoding="utf-8") as rfh:
                for raw in rfh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    dom = rec.get("domain")
                    if dom != DOMAIN_SCOPE_RESTRICTION_V2:
                        skipped_non_workspace += 1
                        continue

                    total_processed += 1
                    meth = rec.get("method", LEGACY_UNKNOWN_SENTINEL_METHOD_TAG) or LEGACY_UNKNOWN_SENTINEL_METHOD_TAG
                    tech_id, axis_tuple = classify_method_tag(meth)

                    if not str(meth).strip():
                        reason_code = "empty-string-fallback-to-sentinel-tuple"
                    elif axis_tuple == (str(axis_tuple[0]), "TOOL_RETURN_VALUE", "IMMEDIATE_CALL", "NONE", "SINGLE_SHOT"):
                        reason_code = "default-fill-after-partial-miss"
                    else:
                        reason_code = "at-least-one-axis-matched-explicitly"
                    record_alias_entry(str(aliases_path), str(meth), (tech_id, axis_tuple), reason_code)

                    if tech_id.startswith("tech_"):
                        registry_acc.setdefault(tech_id, axis_tuple)
                    else:
                        orphan_count += 1
                        continue

                    signals_ref_block = rec.get("original_signals_for_reference", {})
                    sr = SignalsRef(
                        injection_point=int(signals_ref_block.get("injection_point", 0)),
                        turning_point=int(signals_ref_block.get("turning_point", 0)),
                        delta=int(signals_ref_block.get("delta", 0)),
                        delta_normalized=float(signals_ref_block.get("delta_normalized", 0.0)),
                        edit_distance=int(signals_ref_block.get("edit_distance", -1)),
                    )

                    poisoned_obs_txt = rec.get("_provenance", {}).get("poisoned_observation_sha256_first16", "")
                    payload_hash = poisoned_obs_txt or _hash_short((rec.get("method", "") + rec.get("task_id", "")), 16)

                    prov: dict[str, Any] = {"_provenance": dict(rec.get("_provenance", {}) or {})}
                    if "source" in rec:
                        prov["source"] = rec["source"]

                    toolkit_sig = _build_toolkit_signature(rec.get("context_prefix_actions", []) or [])

                    sc = ScenarioRecordV2(
                        scenario_id=rec.get("scenario_id", f"scn_{payload_hash}"),
                        bucket=bucket,
                        origin_mode="mined",
                        delta_value_orig=int(rec.get("delta_value_orig", sr.delta)),
                        canonical_technique_id=tech_id,
                        method_tag_raw=meth,
                        task_id=rec.get("task_id", ""),
                        domain=dom,
                        toolkit_signature=toolkit_sig,
                        goal_instruction=rec.get("goal_instruction", ""),
                        context_prefix_actions=rec.get("context_prefix_actions", []) or [],
                        poisoned_observation_text=poisoned_obs_txt,
                        injected_payload_sha256_first16=payload_hash,
                        injection_target_turn_index=int(rec.get("injection_target_turn_index", 0)),
                        signals_ref=sr,
                        provenance=prov,
                    )
                    wfh.write(sc.to_json_line() + "\n")

    reg_path = tech_dir / "registry.jsonl"
    with open(reg_path, "w", encoding="utf-8") as rh:
        for tid in sorted(registry_acc.keys()):
            tup = registry_acc[tid]
            entry = {"tech_id": tid,
                     "envelope_pattern": tup[0], "channel_class": tup[1],
                     "trigger_mechanism": tup[2], "authority_vector": tup[3],
                     "persistence_dimension": tup[4]}
            rh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    manifest_summary = {
        "summary_version": SCHEMA_VERSION_V2,
        "total_processed": total_processed,
        "skipped_non_workspace": skipped_non_workspace,
        "unique_canonical_families_written": len(registry_acc),
        "orphan_rate": orphan_count / max(total_processed, 1),
    }
    summary_local = {
        "records_total_processed": total_processed,
        "records_skipped_domain_filter": skipped_non_workspace,
        "families_registered": len(registry_acc),
        "records_orphan_after_classification": orphan_count,
    }
    (diag_dir / "migration_audit.json").write_text(
        json.dumps(manifest_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return summary_local


if __name__ == "__main__":
    raise SystemExit("Invoke programmatically via evoguard.process.bench_migrate.migrate(...)")
