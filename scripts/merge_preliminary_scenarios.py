"""Merge multiple ``all_scenarios.json`` files emitted by
``preliminary.collect`` into one consolidated bundle suitable for
feeding into ``preliminary.probe``.

Use case (Tier-1 multi-domain × longer-thought joint expansion):
    * An earlier workspace-only run produced 100 valid scenarios tagged
      implicitly via their ``task_id`` prefix but lacking an explicit
      ``domain`` field.
    * Newer runs targeting banking / slack / travel emit records carrying an
      explicit ``domain`` field thanks to the patched harness.
    * This script concatenates them while ensuring every output record carries
      a populated ``domain`` field (back-filled from ``task_id`` prefix when
      missing) and unique ``(task_id, scenario_idx)`` keys.

Duplicate-key policy: prefer the LATER entry in argv order so subsequent runs
can override older ones intentionally. A summary log lists any overrides.

Example::

    python scripts/merge_preliminary_scenarios.py \
        rounds/_preliminary/20260803_alertness_v3/all_scenarios.json \
        rounds/_preliminary/20260805_alertness_banking/all_scenarios.json \
        rounds/_preliminary/20260806_alertness_slack_travel/all_scenarios.json \
        --output rounds/_preliminary/20260807_tier1_multidomain/all_scenarios.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _infer_domain(record: dict) -> str | None:
    """Recover suite name either from explicit 'domain' or from task_id prefix."""
    d = record.get("domain")
    if isinstance(d, str) and d.strip():
        return d.strip()
    tid = str(record.get("task_id") or "")
    # Expected format: agentdojo:<suite>:<hex>
    parts = tid.split(":")
    if len(parts) >= 2 and parts[0] == "agentdojo":
        return parts[1]
    return None


def _patch_domain_field(records: list[dict]) -> tuple[int, int]:
    """Back-fill 'domain' on records missing it; returns (filled_count, total)."""
    filled = 0
    total = len(records)
    for r in records:
        inferred = _infer_domain(r)
        if not r.get("domain"):
            r["domain"] = inferred
            filled += 1
        elif not isinstance(r.get("domain"), str) or not r["domain"]:
            r["domain"] = inferred
            filled += 1
    return filled, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=str,
                    help="Two-or-more paths to all_scenarios.json bundles.")
    ap.add_argument("--output", required=True,
                    help="Destination merged JSON path; parent dir auto-created.")
    args = ap.parse_args()

    inputs = [Path(p).resolve() for p in args.inputs]
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys: dict[tuple[str, int], dict] = {}
    override_log: list[dict] = []
    totals_per_input = []

    for src_idx, src in enumerate(inputs):
        if not src.exists():
            print(f"[WARN] input #{src_idx} ({src}) does not exist; skipping",
                  file=sys.stderr)
            continue
        try:
            rows = json.loads(src.read_text())
        except json.JSONDecodeError as exc:
            print(f"[ERROR] failed parsing {src}: {exc}", file=sys.stderr)
            sys.exit(1)

        n_filled, n_total = _patch_domain_field(rows)
        overridden_here = 0
        appended_here = 0
        for r in rows:
            key = (str(r.get("task_id")), int(r.get("scenario_idx", -1)))
            prev = seen_keys.get(key)
            if prev is None:
                seen_keys[key] = r
                appended_here += 1
            else:
                override_log.append({
                    "key": list(key),
                    "from_source": prev.get("_source"),
                    "to_source": str(src),
                    "reason": "later-input-overrides-by-design"})

                # Stamp provenance so logs are debuggable post-merge.
                prev["_source"] = prev.get("_source", f"<input#{src_idx}>:{src}")
                r["_source"] = f"<input#{src_idx}>:{src}"
                seen_keys[key] = r
                overridden_here += 1

        totals_per_input.append({
            "source": str(src),
            "records_read": len(rows),
            "domains_filled": n_filled,
            "appended_unique": appended_here,
            "overrode_existing": overridden_here,
        })

    final_records = list(seen_keys.values())
    out_path.write_text(json.dumps(final_records, ensure_ascii=False,
                                   indent=2, default=str))

    domains_breakdown: dict[str,int] = {}
    t_strata_seen:set[float] = set()
    for r in final_records:
        d=r.get("domain"); domains_breakdown[d]=domains_breakdown.get(d,0)+1
        ts_list=(r.get("temperature_schedule") or [])
        for tv in ts_list:
            try:t_strata_seen.add(float(tv))
            except(TypeError,ValueError):pass

    summary={
        "inputs_processed":len(totals_per_input),
        "per_input_breakdown":totals_per_input,
        "override_log_excerpt_top_10":override_log[:10],
        "total_overrides":len(override_log),
        "merged_record_count":len(final_records),
        "domains_breakdown":dict(sorted(domains_breakdown.items())),
        "unique_temperature_strata_in_merged_bundle":
           sorted([float(t) for t in t_strata_seen]),
        "destination":str(out_path)}
    sidecar=out_path.with_suffix(".merge_summary.json")
    sidecar.write_text(json.dumps(summary,indent=2,default=str))

    print(f"\n=== Merged {len(final_records)} scenarios from "
          f"{len(totals_per_input)} sources ===")
    print(f"Domains breakdown:")
    for k,v in sorted(domains_breakdown.items()):
        print(f"  {k:>14s} : {v}")
    print(f"\nSummary written alongside:\n  {sidecar}")


if __name__ == "__main__":
    main()
