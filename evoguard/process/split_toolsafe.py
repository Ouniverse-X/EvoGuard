"""Stratified train/test split of the ToolSafe AgentDojo annotations.

The shipped ``data/toolsafe/agentdojo-tragj/*.json`` files are *segment*-level
records (1220 rows) that collapse to only **58 distinct task instructions** once
:class:`evoguard.envs.toolsafe._ToolSafeEnv` groups them (first-seen record per
instruction wins). Two consequences drive the design here:

1. The split must be taken over **instructions**, not rows -- splitting rows
   would leak the same task into both sides.
2. The existing implicit split (``driver._split_train_val``, tail fraction of
   load order) is *not* stratified: with files read alphabetically, the tail 15%
   lands entirely inside ``workspace``. So held-out ASR was measured on a single
   suite while training saw all four.

This module writes an explicit, reproducible split that keeps class proportions
intact along two axes -- ``suite`` and the record's ``score`` label (which the
env surfaces as ``metadata["annotated_score"]``). Per-stratum test counts use
largest-remainder allocation so each suite hits ``round(n_suite * ratio)``
exactly instead of drifting from repeated flooring.

Output tree (consumed by :class:`evoguard.envs.toolsafe.AgentDojoSplitEnv`)::

    data/toolsafe/agentdojo-tragjnew/
        train/{banking,slack,travel,workspace}.json
        val/{banking,slack,travel,workspace}.json      # see carve_val_split
        test/{banking,slack,travel,workspace}.json
        split_manifest.json

Every segment record belonging to a selected instruction is copied verbatim, in
original file order, so downstream consumers see byte-identical rows.

Run as::

    python -m evoguard.process.split_toolsafe [--ratio 0.2] [--seed 20260817]

**The ``val`` split is carved separately, never by re-running the two-way split.**
``test/`` is the baseline every held-out replay number in ``CLAUDE.md`` was
measured on, so it must stay byte-identical; re-running :func:`build_split` with
a third ratio would reshuffle it. :func:`carve_val_split` therefore reads the
*existing* ``train/`` files, moves whole instructions out of them into ``val/``,
and never opens ``test/`` for writing (the CLI verifies this with a checksum)::

    python -m evoguard.process.split_toolsafe --carve-val
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import OrderedDict, defaultdict

DEFAULT_SRC = os.path.join("data", "toolsafe", "agentdojo-tragj")
DEFAULT_DST = os.path.join("data", "toolsafe", "agentdojo-tragjnew")
DEFAULT_SUITES = ("banking", "slack", "travel", "workspace")

# Val task counts per suite (10 tasks out of the 47 train tasks). Chosen by hand
# rather than from a ratio so every suite contributes at least one task.
DEFAULT_VAL_COUNTS = {"banking": 1, "slack": 1, "travel": 1, "workspace": 7}
DEFAULT_VAL_SEED = 20260823


def _instruction_key(instruction: str) -> str:
    """Short stable id for manifest readability (mirrors env's uid digest)."""
    return hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:10]


def _group_by_instruction(records: list[dict]) -> "OrderedDict[str, list[dict]]":
    """Group segment rows by instruction, preserving first-seen order.

    Order matters: it is the same order ``_ToolSafeEnv._load`` uses, so a
    manifest produced here lines up with the env's task enumeration.
    """
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for rec in records:
        instruction = (rec.get("instruction") or "").strip()
        if not instruction:
            continue
        groups.setdefault(instruction, []).append(rec)
    return groups


def _stratum_of(rows: list[dict]) -> str:
    """Stratum label for an instruction group = first row's ``score``.

    The first row is what the env keeps as ``annotated_score``, so stratifying
    on it preserves exactly the label distribution the pipeline observes.
    """
    return str(rows[0].get("score"))


def _allocate_test_counts(sizes: dict[str, int], ratio: float) -> dict[str, int]:
    """Largest-remainder allocation of ``round(total*ratio)`` across strata.

    Plain per-stratum flooring loses items (e.g. workspace 9*0.2=1.8 and
    31*0.2=6.2 floor to 7, one short of the intended 8). Largest remainder
    keeps the suite total on target while still respecting proportions.
    """
    total = sum(sizes.values())
    target = int(round(total * ratio))
    target = max(0, min(total, target))
    base = {k: int(v * ratio) for k, v in sizes.items()}
    remainder = {k: v * ratio - base[k] for k, v in sizes.items()}
    deficit = target - sum(base.values())
    # Ties broken by stratum name for determinism across runs/platforms.
    order = sorted(sizes, key=lambda k: (-remainder[k], k))
    i = 0
    while deficit > 0 and order:
        k = order[i % len(order)]
        if base[k] < sizes[k]:
            base[k] += 1
            deficit -= 1
        i += 1
        if i > 4 * len(order) + target:                # pathological guard
            break
    return base


def split_suite(
    records: list[dict],
    ratio: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split one suite. Returns ``(train_rows, test_rows, manifest_entries)``."""
    groups = _group_by_instruction(records)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for instruction, rows in groups.items():
        by_stratum[_stratum_of(rows)].append(instruction)

    sizes = {s: len(v) for s, v in by_stratum.items()}
    n_test = _allocate_test_counts(sizes, ratio)

    test_instructions: set[str] = set()
    for stratum, instructions in by_stratum.items():
        pool = list(instructions)
        rng.shuffle(pool)
        test_instructions.update(pool[: n_test[stratum]])

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    manifest: list[dict] = []
    for instruction, rows in groups.items():
        split = "test" if instruction in test_instructions else "train"
        (test_rows if split == "test" else train_rows).extend(rows)
        manifest.append({
            "instruction_id": _instruction_key(instruction),
            "split": split,
            "stratum": _stratum_of(rows),
            "n_segments": len(rows),
            "instruction_preview": instruction[:120],
        })
    return train_rows, test_rows, manifest


def build_split(
    src_dir: str = DEFAULT_SRC,
    dst_dir: str = DEFAULT_DST,
    *,
    suites: tuple[str, ...] = DEFAULT_SUITES,
    ratio: float = 0.2,
    seed: int = 20260817,
) -> dict:
    """Write the stratified split under ``dst_dir`` and return the manifest."""
    os.makedirs(os.path.join(dst_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, "test"), exist_ok=True)

    manifest: dict = {
        "source_dir": src_dir,
        "test_ratio": ratio,
        "seed": seed,
        "stratified_by": ["suite", "score"],
        "split_unit": "distinct instruction",
        "suites": {},
    }

    for suite in suites:
        src_path = os.path.join(src_dir, f"{suite}.json")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(src_path)
        with open(src_path, encoding="utf-8") as fh:
            records = json.load(fh)

        # Per-suite RNG seeded from (seed, suite) so adding or reordering suites
        # never perturbs another suite's assignment.
        rng = random.Random(f"{seed}:{suite}")
        train_rows, test_rows, entries = split_suite(records, ratio, rng)

        for split, rows in (("train", train_rows), ("test", test_rows)):
            out_path = os.path.join(dst_dir, split, f"{suite}.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, ensure_ascii=False, indent=1)

        n_train_tasks = sum(1 for e in entries if e["split"] == "train")
        n_test_tasks = sum(1 for e in entries if e["split"] == "test")
        manifest["suites"][suite] = {
            "n_tasks": len(entries),
            "n_train_tasks": n_train_tasks,
            "n_test_tasks": n_test_tasks,
            "n_train_segments": len(train_rows),
            "n_test_segments": len(test_rows),
            "tasks": entries,
        }

    manifest["totals"] = {
        "n_tasks": sum(v["n_tasks"] for v in manifest["suites"].values()),
        "n_train_tasks": sum(v["n_train_tasks"] for v in manifest["suites"].values()),
        "n_test_tasks": sum(v["n_test_tasks"] for v in manifest["suites"].values()),
    }
    with open(os.path.join(dst_dir, "split_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


def _allocate_absolute(sizes: dict[str, int], target: int) -> dict[str, int]:
    """Largest-remainder allocation of an *absolute* ``target`` across strata.

    :func:`_allocate_test_counts` takes a ratio; the val carve is specified as a
    task count per suite instead (the counts were chosen by hand so every suite
    contributes at least one task -- a ratio of 0.2 on ``workspace``'s 32 train
    tasks rounds to 6, not the 7 we want).
    """
    total = sum(sizes.values())
    target = max(0, min(total, target))
    if total <= 0 or target == 0:
        return {k: 0 for k in sizes}
    base = {k: int(v * target / total) for k, v in sizes.items()}
    remainder = {k: v * target / total - base[k] for k, v in sizes.items()}
    deficit = target - sum(base.values())
    order = sorted(sizes, key=lambda k: (-remainder[k], k))
    i = 0
    while deficit > 0 and order:
        k = order[i % len(order)]
        if base[k] < sizes[k]:
            base[k] += 1
            deficit -= 1
        i += 1
        if i > 4 * len(order) + target:                # pathological guard
            break
    return base


def carve_val_split(
    dst_dir: str = DEFAULT_DST,
    *,
    suites: tuple[str, ...] = DEFAULT_SUITES,
    val_counts: dict[str, int] | None = None,
    seed: int = DEFAULT_VAL_SEED,
) -> dict:
    """Move whole instructions out of ``train/`` into a new ``val/``.

    ``test/`` is never read or written here -- it is the baseline for every
    held-out replay number on record, so it must stay byte-identical. The val
    tasks are drawn from the *current* ``train/`` files, stratified by ``score``
    inside each suite, with a per-suite RNG so changing one suite's count leaves
    the others untouched.
    """
    manifest_path = os.path.join(dst_dir, "split_manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    os.makedirs(os.path.join(dst_dir, "val"), exist_ok=True)
    counts = dict(DEFAULT_VAL_COUNTS if val_counts is None else val_counts)

    for suite in suites:
        train_path = os.path.join(dst_dir, "train", f"{suite}.json")
        if not os.path.isfile(train_path):
            raise FileNotFoundError(train_path)
        with open(train_path, encoding="utf-8") as fh:
            rows = json.load(fh)

        groups = _group_by_instruction(rows)
        by_stratum: dict[str, list[str]] = defaultdict(list)
        for instruction, g in groups.items():
            by_stratum[_stratum_of(g)].append(instruction)
        sizes = {s: len(v) for s, v in by_stratum.items()}
        per_stratum = _allocate_absolute(sizes, counts.get(suite, 0))

        rng = random.Random(f"{seed}:val:{suite}")
        val_instructions: set[str] = set()
        for stratum in sorted(by_stratum):             # sorted => deterministic
            pool = list(by_stratum[stratum])
            rng.shuffle(pool)
            val_instructions.update(pool[: per_stratum[stratum]])

        train_rows: list[dict] = []
        val_rows: list[dict] = []
        for instruction, g in groups.items():
            (val_rows if instruction in val_instructions else train_rows).extend(g)

        for split, out_rows in (("train", train_rows), ("val", val_rows)):
            out_path = os.path.join(dst_dir, split, f"{suite}.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(out_rows, fh, ensure_ascii=False, indent=1)

        val_ids = {_instruction_key(i) for i in val_instructions}
        info = manifest["suites"][suite]
        for entry in info["tasks"]:
            if entry["instruction_id"] in val_ids:
                if entry["split"] != "train":
                    raise RuntimeError(
                        f"{suite}: instruction {entry['instruction_id']} is "
                        f"'{entry['split']}', refusing to relabel it 'val'")
                entry["split"] = "val"
        info["n_train_tasks"] = sum(1 for e in info["tasks"] if e["split"] == "train")
        info["n_val_tasks"] = sum(1 for e in info["tasks"] if e["split"] == "val")
        info["n_train_segments"] = len(train_rows)
        info["n_val_segments"] = len(val_rows)

    manifest["val_counts"] = counts
    manifest["val_seed"] = seed
    manifest["val_carved_from"] = "train"
    manifest["totals"]["n_train_tasks"] = sum(
        v["n_train_tasks"] for v in manifest["suites"].values())
    manifest["totals"]["n_val_tasks"] = sum(
        v.get("n_val_tasks", 0) for v in manifest["suites"].values())
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--ratio", type=float, default=0.2, help="test fraction")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--carve-val", action="store_true",
                    help="do NOT re-split; move val tasks out of the existing "
                         "train/ split and leave test/ byte-identical")
    ap.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED)
    args = ap.parse_args()

    if args.carve_val:
        import hashlib as _hl

        def _digests() -> dict[str, str]:
            out = {}
            for suite in DEFAULT_SUITES:
                p = os.path.join(args.dst, "test", f"{suite}.json")
                with open(p, "rb") as fh:
                    out[suite] = _hl.sha256(fh.read()).hexdigest()
            return out

        before = _digests()
        manifest = carve_val_split(args.dst, seed=args.val_seed)
        after = _digests()
        if before != after:
            raise SystemExit("FATAL: test/ changed; the carve must never touch it")
        for suite, info in manifest["suites"].items():
            print(f"{suite:<10} tasks={info['n_tasks']:<3} "
                  f"train={info['n_train_tasks']:<3} "
                  f"val={info.get('n_val_tasks', 0):<3} "
                  f"test={info['n_test_tasks']:<3} "
                  f"segments={info['n_train_segments']}/"
                  f"{info.get('n_val_segments', 0)}/{info['n_test_segments']}")
        t = manifest["totals"]
        print(f"{'TOTAL':<10} tasks={t['n_tasks']:<3} train={t['n_train_tasks']:<3} "
              f"val={t['n_val_tasks']:<3} test={t['n_test_tasks']}")
        print("test/ sha256 unchanged")
        return

    manifest = build_split(args.src, args.dst, ratio=args.ratio, seed=args.seed)
    for suite, info in manifest["suites"].items():
        strata = defaultdict(lambda: [0, 0])
        for e in info["tasks"]:
            strata[e["stratum"]][0 if e["split"] == "train" else 1] += 1
        detail = " ".join(f"score={k}:{v[0]}/{v[1]}" for k, v in sorted(strata.items()))
        print(f"{suite:<10} tasks={info['n_tasks']:<3} "
              f"train={info['n_train_tasks']:<3} test={info['n_test_tasks']:<3} "
              f"segments={info['n_train_segments']}/{info['n_test_segments']}  {detail}")
    t = manifest["totals"]
    print(f"{'TOTAL':<10} tasks={t['n_tasks']:<3} "
          f"train={t['n_train_tasks']:<3} test={t['n_test_tasks']}")
    print(f"written -> {args.dst}")


if __name__ == "__main__":
    main()
