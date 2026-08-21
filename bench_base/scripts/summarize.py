"""Aggregate bench_base/scenarios/_synthetic/*mcts*.jsonl into a compact summary table.

Usage:
    python bench_base/scripts/summarize.py [bench_root]

Prints per-file stats (n / verified-success-count / heuristic-only-success /
unverified count), grouped totals per delta bucket × domain origin, plus
combined gap-vs-target assessment against CHANGELOG goal d3≥55 d4≥55.
"""
from __future__ import annotations
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CLASSES_ORDER = ("verified_success", "heuristic_only", "unverified_failure")


def _classify(rec: dict) -> str:
    v = bool(rec.get("judge_verified", False))
    s = bool(rec.get("success", False))
    if v and s:
        return "verified_success"
    if s and not v:
        return "heuristic_only"
    if v and not s:
        return "verified_but_no_heuristic"  # unusual state, not in CLASSES_ORDER
    return "unverified_failure"


def main() -> None:
    bench_root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "bench_base"
    synth_dir = bench_root / "scenarios" / "_synthetic"

    pattern = sorted(synth_dir.glob("bucket_*_mcts_seed*.jsonl"))
    print(f"# Scanning {synth_dir}")
    print(f"# Found {len(pattern)} bucket_*_mcts_seed* files\n")

    rows = []
    per_bucket_domain_classified: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)

    for fpath in pattern:
        name = fpath.name
        try:
            parts = name.replace(".jsonl", "").split("_")
            bkt = parts[1]
            seed_part = parts[-1].replace("seed", "")
            seed = int(seed_part)
        except Exception:
            continue

        n_total = v_succ = h_succ = unv_fail = 0
        domains_seen: collections.Counter = collections.Counter()
        deltas: list[int] = []

        with open(fpath, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                rec = json.loads(ln)
                cls = _classify(rec)
                dom = rec.get("suite") or "?"
                n_total += 1
                domains_seen[dom] += 1
                deltas.append(int(rec.get("delta", int(bkt[1:]))) )
                key = (bkt, dom)
                per_bucket_domain_classified[key][cls] += 1
                if cls == "verified_success":
                    v_succ += 1
                elif cls == "heuristic_only":
                    h_succ += 1
                else:
                    unv_fail += 1

        primary_dom = domains_seen.most_common(1)[0][0] if domains_seen else "?"
        avg_d = sum(deltas) / max(len(deltas), 1)
        max_d = max(deltas) if deltas else 0
        rows.append({
            "name": name,
            "bkt": bkt,
            "seed": seed,
            "domain": primary_dom,
            "n": n_total,
            "verif": v_succ,
            "heur": h_succ,
            "unv": unv_fail,
            "avg_delta": round(avg_d, 2),
            "max_delta": max_d,
        })

    if not rows:
        print("No matching synthetic JSONL files.")
        return

    header = f'{"file":48} {"bk":3} {"sd":4} {"dom":10} {"n":4} {"vf":3} {"hf":3} {"un":3} {"d_avg":7} {"d_max":7}'
    print(header); print("-" * len(header))

    def sort_key(r):
        order_bkt = {"d2": 0, "d3": 1, "d4": 2}.get(r["bkt"], 99)
        return r["domain"], r["seed"], order_bkt

    for r in sorted(rows, key=sort_key):
        print(f'{r["name"]:48} {r["bkt"]:3} {r["seed"]:<4} {r["domain"]:10} '
              f'{r["n"]:4} {r["verif"]:3} {r["heur"]:3} {r["unv"]:3} '
              f'{r["avg_delta"]:7} {r["max_delta"]:7}')

    # Combined target-gap check vs main merged buckets
    print("\n## Current main-bucket counts after merges:")
    targets = {"imm": -1, "d1": -1, "d2": 55, "d3": 55, "d4": 55}
    for bkt in ("imm", "d1", "d2", "d3", "d4"):
        p = bench_root / "scenarios" / f"bucket_{bkt}.jsonl"
        if not p.exists():
            continue
        cnt = 0
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    cnt += 1
        tgt = targets.get(bkt, -1)
        if tgt >= 0:
            status = f"(target ≥{tgt}) {'✓ MET' if cnt >= tgt else '✗ GAP=' + str(tgt - cnt)}"
        else:
            status = "(mined only)"
        print(f"  {p.name}: n={cnt} {status}")

    # Benign (negative-class) arm, if materialized. Not a Δ bucket: it carries no
    # payload and origin_mode=benign_arm, so the release gate never sees it.
    bp = bench_root / "scenarios" / "bucket_clean.jsonl"
    if bp.exists():
        n_clean = parents = 0
        with open(bp, encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                n_clean += 1
                prov = (json.loads(ln).get("_provenance") or {}).get("benign") or {}
                parents += int(prov.get("parent_count", 1) or 1)
        print(f"  {bp.name}: n={n_clean} (benign arm, covers {parents} attacked scenarios)")

    # Per-domain verification rate summary within synth dir only
    print("\n## Verification-class counts (synth dir only):")
    seen_domains_per_bkt: dict[str, set] = collections.defaultdict(set)
    for (bkt, dom) in per_bucket_domain_classified.keys():
        seen_domains_per_bkt[bkt].add(dom)

    header_row = f'{"(bucket,domain)":24}' + ''.join(f'{cls[:8]:>13}' for cls in CLASSES_ORDER) + '     total'
    print(header_row); print("-" * len(header_row))
    grand_totals: list[int] = [0]*len(CLASSES_ORDER)
    for bkt in ("d2", "d3", "d4"):
        for dom in sorted(seen_domains_per_bkt.get(bkt, [])):
            counter = per_bucket_domain_classified[(bkt, dom)]
            row_vals = [counter.get(c, 0) for c in CLASSES_ORDER]
            total = sum(row_vals)
            label = f"{bkt}|{dom}"
            line = f"{label:24}" + ''.join(f"{v:>13}" for v in row_vals) + f"{total:>9}"
            print(line)
            for i,v in enumerate(row_vals):
                grand_totals[i] += v
    gt_line = f'{"GRAND_TOTAL":24}' + ''.join(f'{v:>13}' for v in grand_totals) + f'{sum(grand_totals):>9}'
    print(gt_line)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
