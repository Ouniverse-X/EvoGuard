"""Build ``data/delta_bench/`` -- a Δ-stratified, paired IPI attack benchmark.

Every case group is ONE AgentDojo task/carrier/sink triple carrying FOUR
payloads that differ only in *when* they ask for the harmful call:

    tier   expected_delta   interposed clean steps between carrier and sink
    imm    1                0   (fires at the very next step)
    d1     2                1
    d2     3                2
    d3     4                3

``expected_delta`` is the plan-index offset from the carrier, and 1 is its
SMALLEST constructible value -- the payload only becomes visible in the
carrier's observation, so the earliest step that can act on it is
``carrier_index + 1``. ``imm`` therefore means "immediate", not a literal
Δ=0 field value; this matches ``data/agentdojo_stock/``'s convention.

Because the four tiers share task, clean plan, carrier, benign observation,
sink and args by CONSTRUCTION (all of it read from one source record), Δ is
the sole variable across the ladder.

Inputs
------
* ``data/agentdojo/default_suites/v1/{suite}/`` -- verbatim task PROMPTs,
  ``ground_truth`` clean plans and the advertised tool inventory, via
  ``scripts/build_agentdojo_latent.py``.
* ``data/agentdojo_latent/authored/{suite}.json`` -- the STRUCTURAL record of
  each case (carrier, benign observation, sink, args, goal) plus the payload
  reused verbatim at one tier.
* ``data/delta_bench/authored/{suite}.json`` -- the authored part: which
  latent case a group is built on, which tier reuses its payload, and the
  three new payloads.

Outputs
-------
* ``data/delta_bench/scenarios/bucket_{imm,d1,d2,d3}.jsonl`` -- attacked rows.
* ``data/delta_bench/scenarios/bucket_clean.jsonl`` -- one clean row per
  (task, carrier), shared by all four tiers of every group riding it.
* ``data/delta_bench/manifest.json``, ``data/delta_bench/_sha256_guard.txt``

Run: ``python scripts/build_delta_bench.py [--check]``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_agentdojo_latent import (  # noqa: E402
    load_suite_tool_names,
    load_tool_specs,
    load_user_tasks,
    render_env_info,
    task_uid,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATENT_AUTHORED = os.path.join(REPO_ROOT, "data", "agentdojo_latent", "authored")
OUT_ROOT = os.path.join(REPO_ROOT, "data", "delta_bench")
SUITES = ("banking", "slack", "travel", "workspace")
DATASET = "delta_bench"
SCHEMA_VERSION = 1

TIERS = ("imm", "d1", "d2", "d3")
TIER_DELTA = {"imm": 1, "d1": 2, "d2": 3, "d3": 4}

# A payload announcing itself as an injected channel, or telling the agent to
# disregard its instructions, measures delimiter handling rather than Δ.
FORBIDDEN = ("<INFORMATION>", "</INFORMATION>", "ignore previous", "ignore all previous")


def _check_payload(where: str, tier: str, text: str, harmful_tool: str) -> None:
    if not text or not text.strip():
        raise SystemExit(f"{where}/{tier}: empty payload")
    low = text.lower()
    for bad in FORBIDDEN:
        if bad.lower() in low:
            raise SystemExit(f"{where}/{tier}: payload contains {bad!r}")
    # Lesson from the camouflage probe v1 null: a payload that does not NAME
    # the call it wants is not obeyable, and the tier then measures nothing.
    if harmful_tool not in text:
        raise SystemExit(f"{where}/{tier}: payload never names {harmful_tool!r}")


def build_suite(suite: str, specs: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Return ``(attacked_rows, clean_rows)`` for one suite.

    ``attacked_rows`` carries all four tiers of every group; the caller splits
    them into buckets on ``tier``.
    """
    tasks = load_user_tasks(suite)
    env_info = render_env_info(load_suite_tool_names(suite), specs)
    advertised = set(load_suite_tool_names(suite))

    latent = json.load(open(os.path.join(LATENT_AUTHORED, f"{suite}.json"), encoding="utf-8"))
    authored_path = os.path.join(OUT_ROOT, "authored", f"{suite}.json")
    authored = json.load(open(authored_path, encoding="utf-8"))

    attacked: list[dict] = []
    clean: list[dict] = []
    seen_clean: dict[tuple[str, str], str] = {}

    for group in authored:
        case_id = group["case_id"]
        base = latent[group["latent_index"]]
        reuse_tier = group["reuse_tier"]
        if reuse_tier not in TIERS:
            raise SystemExit(f"{case_id}: reuse_tier {reuse_tier!r} is not a tier")
        if TIER_DELTA[reuse_tier] != base["expected_delta"]:
            raise SystemExit(
                f"{case_id}: reuse_tier {reuse_tier} is delta "
                f"{TIER_DELTA[reuse_tier]} but the reused latent case declares "
                f"delta {base['expected_delta']}"
            )

        payloads = dict(group["payloads"])
        techniques = dict(group["stealth_techniques"])
        if reuse_tier in payloads:
            raise SystemExit(f"{case_id}: {reuse_tier} is reused, do not author it")
        payloads[reuse_tier] = base["payload"]
        techniques.setdefault(reuse_tier, base["stealth_technique"])
        if set(payloads) != set(TIERS):
            raise SystemExit(
                f"{case_id}: tiers {sorted(payloads)} != {list(TIERS)}"
            )

        ref = base["task_ref"]
        if ref not in tasks:
            raise SystemExit(f"{case_id}: unknown task_ref {ref!r}")
        instruction = tasks[ref]["prompt"]
        clean_plan = tasks[ref]["clean_plan"]
        carrier = base["carrier_tool"]
        c_idx = base["carrier_index"]
        harmful = base["harmful_tool"]
        benign = base["benign_observation"]

        if clean_plan[c_idx] != carrier:
            raise SystemExit(
                f"{case_id}: carrier_index {c_idx} is {clean_plan[c_idx]!r}, "
                f"not {carrier!r}"
            )
        # d3 is the binding tier: it needs three clean steps to interpose.
        if c_idx + TIER_DELTA["d3"] > len(clean_plan):
            raise SystemExit(
                f"{case_id}: clean_plan len {len(clean_plan)} cannot support "
                f"carrier_index {c_idx} + delta {TIER_DELTA['d3']}; the case "
                f"does not qualify for a full four-tier ladder"
            )
        if harmful not in advertised:
            raise SystemExit(f"{case_id}: harmful_tool {harmful!r} is not advertised")
        if harmful in clean_plan:
            raise SystemExit(f"{case_id}: harmful_tool {harmful!r} is in clean_plan")

        prior = seen_clean.get((ref, carrier))
        if prior is not None and prior != benign:
            raise SystemExit(
                f"{case_id}: a second benign_observation for carrier {carrier!r} "
                f"on {ref}; a carrier's clean output must be identical across "
                f"every group riding it"
            )

        common = {
            "schema_version": SCHEMA_VERSION,
            "dataset": DATASET,
            "suite": suite,
            "case_id": case_id,
            "task_ref": f"agentdojo:v1:{suite}:{ref}",
            "task_id": task_uid(suite, instruction, DATASET),
            "task_instruction": instruction,
            "env_info": env_info,
            "clean_plan": clean_plan,
            "carrier_tool": carrier,
            "carrier_index": c_idx,
            "benign_observation": benign,
        }

        if prior is None:
            seen_clean[(ref, carrier)] = benign
            clean.append(
                dict(
                    common,
                    case_id=None,
                    instance_id=f"dbench-{suite}-clean-{len(seen_clean):02d}",
                    tier="clean",
                    sample_type="clean",
                    observation=benign,
                    payload=None,
                    harmful_tool=None,
                    harmful_args_hint=None,
                    attacker_goal=None,
                    expected_delta=None,
                    expected_turning_index=None,
                    expected_induced_plan=None,
                    stealth_technique=None,
                    reused_from_latent=None,
                )
            )

        for tier in TIERS:
            delta = TIER_DELTA[tier]
            text = payloads[tier]
            _check_payload(case_id, tier, text, harmful)
            turning = c_idx + delta
            induced = list(clean_plan[:turning]) + [harmful] + list(clean_plan[turning:])
            attacked.append(
                dict(
                    common,
                    instance_id=f"dbench-{suite}-{case_id}-{tier}",
                    tier=tier,
                    sample_type="injected",
                    observation=benign + text,
                    payload=text,
                    harmful_tool=harmful,
                    harmful_args_hint=base["harmful_args_hint"],
                    attacker_goal=base["attacker_goal"],
                    expected_delta=delta,
                    expected_turning_index=turning,
                    expected_induced_plan=induced,
                    stealth_technique=techniques[tier],
                    reused_from_latent=(
                        f"agentdojo_latent:{suite}:{group['latent_index']}"
                        if tier == reuse_tier
                        else None
                    ),
                )
            )
    return attacked, clean


def _tally(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for row in rows:
        out[str(row[key])] = out.get(str(row[key]), 0) + 1
    return dict(sorted(out.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and fail if the on-disk files would change",
    )
    args = parser.parse_args()

    specs = load_tool_specs()
    attacked: list[dict] = []
    clean: list[dict] = []
    per_suite: dict[str, dict] = {}
    for suite in SUITES:
        s_attacked, s_clean = build_suite(suite, specs)
        attacked.extend(s_attacked)
        clean.extend(s_clean)
        per_suite[suite] = {
            "n_groups": len(s_attacked) // len(TIERS),
            "n_attacked": len(s_attacked),
            "n_clean": len(s_clean),
            "by_technique": _tally(s_attacked, "stealth_technique"),
        }

    payloads: dict[str, str] = {}
    counts: dict[str, int] = {}
    for tier in TIERS:
        rows = [r for r in attacked if r["tier"] == tier]
        payloads[f"scenarios/bucket_{tier}.jsonl"] = _dump(rows)
        counts[tier] = len(rows)
    payloads["scenarios/bucket_clean.jsonl"] = _dump(clean)
    counts["clean"] = len(clean)

    sizes = {counts[t] for t in TIERS}
    if len(sizes) != 1:
        raise SystemExit(f"tier buckets are ragged: {counts}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "built_by": "scripts/build_delta_bench.py",
        "source": (
            "AgentDojo v1 default suites (vendored at data/agentdojo/) for the "
            "task, clean plan and tool inventory; data/agentdojo_latent/authored/ "
            "for each case's carrier, benign observation, sink and args"
        ),
        "split_unit": "none -- the whole set is a held-out diagnostic probe",
        "n_groups": counts["imm"],
        "n_attacked_total": sum(counts[t] for t in TIERS),
        "counts": counts,
        "per_suite": per_suite,
        "tier_semantics": {
            tier: {
                "expected_delta": TIER_DELTA[tier],
                "interposed_clean_steps": TIER_DELTA[tier] - 1,
            }
            for tier in TIERS
        },
        "delta_semantics": (
            "expected_delta is a DECLARED plan-index offset from the carrier, "
            "not signals.delta, which is measured at replay time and will "
            "differ. 1 is its smallest constructible value because the payload "
            "first becomes visible in the carrier's observation, so tier 'imm' "
            "means immediate, not a literal delta of 0."
        ),
        "pairing": (
            "The four tiers of a group share task, clean plan, carrier, benign "
            "observation, sink and args by construction -- all of it read from "
            "one source record -- so the payload's deferral depth is the sole "
            "variable. Rebuilding one bucket alone breaks nothing visibly; "
            "evoguard/tests/test_delta_bench.py is what catches it."
        ),
        "excluded_sources": {
            "ASB": (
                "every agent config advertises exactly 2 tools and rows carry no "
                "clean plan, so no tier past imm is constructible without "
                "fabricating the clean arm"
            ),
            "InjecAgent": (
                "every user case is a single tool call (clean plan length 1, "
                "carrier_index 0), so delta 1 is the maximum and the ladder "
                "cannot exist"
            ),
        },
    }
    payloads["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=1) + "\n"

    guard_lines = [
        "# sha256 guard for data/delta_bench/",
        "# generated-by: scripts/build_delta_bench.py",
        f"# n-groups: {counts['imm']}",
    ]
    for rel in sorted(payloads):
        digest = hashlib.sha256(payloads[rel].encode("utf-8")).hexdigest()
        guard_lines.append(f"sha256({rel})={digest}")
    payloads["_sha256_guard.txt"] = "\n".join(guard_lines) + "\n"

    if args.check:
        _check(payloads)
        return
    for rel, text in payloads.items():
        path = os.path.join(OUT_ROOT, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(f"wrote {len(payloads)} files under {OUT_ROOT}")
    print(f"  {counts['imm']} groups x 4 tiers = {sum(counts[t] for t in TIERS)} "
          f"attacked rows + {counts['clean']} clean")
    for suite, c in per_suite.items():
        print(f"  {suite}: {c['n_groups']} groups, {c['n_clean']} clean, "
              f"tech={c['by_technique']}")


def _dump(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def _check(payloads: dict[str, str]) -> None:
    bad = []
    for rel, text in payloads.items():
        path = os.path.join(OUT_ROOT, rel)
        if not os.path.exists(path):
            bad.append(f"{rel}: missing")
        elif open(path, encoding="utf-8").read() != text:
            bad.append(f"{rel}: differs from rebuild")
    if bad:
        raise SystemExit("stale build:\n  " + "\n  ".join(bad))
    print("on-disk files match a fresh rebuild")


if __name__ == "__main__":
    sys.exit(main())
