"""Offline invariants for the vendored (frozen) attacker ablation backend.

Run: ``python -m evoguard.tests.test_vendored_attacker`` (optionally ``-k name``).
No GPU, no network -- reads only ``data/toolsafe/agentdojo-tragjnew/train``.
"""

from __future__ import annotations

import os
import sys

from evoguard.attacks import build_attacker
from evoguard.attacks.vendored import VendoredAttacker
from evoguard.config import AttackerConfig
from evoguard.core.types import Task

TRAIN_DIR = "data/toolsafe/agentdojo-tragjnew/train"
VAL_DIR = "data/toolsafe/agentdojo-tragjnew/val"


def _env():
    from evoguard.envs import build_env
    from evoguard.config import EnvConfig

    return build_env(EnvConfig(dataset="agentdojo_split", data_root="data",
                              max_tasks=200))


def _cfg(dataset_dir: str = TRAIN_DIR) -> AttackerConfig:
    return AttackerConfig(search_method="vendored",
                          vendored_dataset_dir=dataset_dir)


def _first_task_with_attacks(env, cfg, *, want: int = 1):
    """Return (task, tools, attacker) for the first task carrying >= want."""
    for t in env.get_tasks():
        if t.metadata.get("split") != "train":
            continue
        tools = env.get_tools(t)
        atk = VendoredAttacker(task=t, tools=tools, config=cfg,
                               defense_max_turns=8)
        if len(atk.current_population()) >= want:
            return t, tools, atk
    raise AssertionError(f"no train task carries >= {want} vendored attacks")


def test_factory_selects_vendored():
    """search_method='vendored' reaches VendoredAttacker via build_attacker."""
    env = _env()
    task = next(t for t in env.get_tasks() if t.metadata.get("split") == "train")
    atk = build_attacker(task, env.get_tools(task), None, _cfg(),
                         defense_max_turns=8)
    assert isinstance(atk, VendoredAttacker), type(atk)
    print("  factory -> VendoredAttacker, generator=None accepted (unused)")


def test_unknown_method_still_rejected():
    try:
        build_attacker(Task(task_id="x:y:z", instruction="i"), [], None,
                       AttackerConfig(search_method="nope"))
    except ValueError as exc:
        assert "vendored" in str(exc), str(exc)
        print(f"  unknown method rejected, message lists vendored: {exc}")
        return
    raise AssertionError("expected ValueError on unknown search_method")


def test_empty_dataset_dir_is_a_hard_error():
    """An unset dir must RAISE, not fall back to the unsplit tree."""
    env = _env()
    task = next(t for t in env.get_tasks() if t.metadata.get("split") == "train")
    try:
        VendoredAttacker(task=task, tools=env.get_tools(task),
                         config=AttackerConfig(search_method="vendored"),
                         defense_max_turns=8)
    except ValueError as exc:
        assert "vendored_dataset_dir" in str(exc)
        print("  empty vendored_dataset_dir raises (no silent val/test leak)")
        return
    raise AssertionError("expected ValueError on empty vendored_dataset_dir")


def test_population_is_frozen_across_evolve():
    """evolve() must not change the population -- byte-identical every round."""
    env = _env()
    _t, _tools, atk = _first_task_with_attacks(env, _cfg(), want=2)
    before = [s.to_dict() for s in atk.current_population()]
    for _ in range(3):
        atk.evolve([])
    after = [s.to_dict() for s in atk.current_population()]
    # attack_id is generated per spec at construction, so compare the genome.
    keys = ("task_id", "target_turn", "injection_channel", "method",
            "payload", "goal", "origin")
    assert [{k: d[k] for k in keys} for d in before] == \
           [{k: d[k] for k in keys} for d in after]
    assert all(d["generation"] == 0 for d in after), "generation must stay 0"
    print(f"  population frozen over 3 evolve() calls (n={len(after)})")


def test_specs_carry_payload_goal_and_provenance():
    env = _env()
    _t, _tools, atk = _first_task_with_attacks(env, _cfg())
    for spec in atk.current_population():
        assert "<INFORMATION>" in spec.payload, spec.payload[:120]
        assert spec.goal.startswith("call "), spec.goal[:80]
        assert spec.origin == "vendored" and spec.method == "vendored_replay"
        assert spec.injection_channel == "tool_result"
        assert spec.metadata.get("vendored") is True
        assert spec.metadata.get("target_tool")
        assert spec.metadata.get("mal_tool")
    print(f"  {len(atk.current_population())} specs carry payload/goal/metadata")


def test_target_turn_within_ceiling():
    """Every target_turn must be inside [0, max_turns-1] after sanitize."""
    env = _env()
    cfg = _cfg()
    seen = 0
    for t in env.get_tasks():
        if t.metadata.get("split") != "train":
            continue
        atk = VendoredAttacker(task=t, tools=env.get_tools(t), config=cfg,
                               defense_max_turns=8)
        for spec in atk.current_population():
            assert 0 <= spec.target_turn <= 7, spec.target_turn
            seen += 1
    assert seen > 0
    print(f"  {seen} specs all have target_turn in [0,7]")


def test_carrier_index_is_dataset_derived_not_constant():
    """The turn comes from the transcript, so it must vary across scenarios."""
    from evoguard.process.vendored_attack_loaders import load_vendored_scenarios

    scen = load_vendored_scenarios(dataset_dir=TRAIN_DIR,
                                   dataset="agentdojo_split")
    idxs = {va.carrier_index for va in scen}
    assert len(idxs) > 1, f"carrier_index collapsed to {idxs}"
    assert min(idxs) >= 0
    print(f"  carrier_index over {len(scen)} scenarios: "
          f"{len(idxs)} distinct, range [{min(idxs)},{max(idxs)}]")


def test_train_and_val_corpora_are_disjoint_dirs():
    """Sanity: the train split must not be the val split (cache key check)."""
    env = _env()
    task = next(t for t in env.get_tasks() if t.metadata.get("split") == "train")
    tools = env.get_tools(task)
    tr = VendoredAttacker(task=task, tools=tools, config=_cfg(TRAIN_DIR),
                          defense_max_turns=8).current_population()
    va = VendoredAttacker(task=task, tools=tools, config=_cfg(VAL_DIR),
                          defense_max_turns=8).current_population()
    assert len(tr) > 0
    assert [s.payload for s in tr] != [s.payload for s in va], \
        "per-dir cache key is not distinguishing the splits"
    print(f"  train={len(tr)} val={len(va)} specs for the same task, distinct")


def test_task_coverage_on_train_split():
    """Report coverage; a task with zero attacks must NOT raise."""
    env = _env()
    cfg = _cfg()
    n_tasks = n_with = n_specs = 0
    for t in env.get_tasks():
        if t.metadata.get("split") != "train":
            continue
        n_tasks += 1
        pop = VendoredAttacker(task=t, tools=env.get_tools(t), config=cfg,
                               defense_max_turns=8).current_population()
        n_specs += len(pop)
        n_with += 1 if pop else 0
    assert n_with > 0 and n_specs > 0
    print(f"  train tasks={n_tasks} with_attacks={n_with} total_specs={n_specs} "
          f"(zero-attack tasks keep their clean arm)")


def main() -> int:
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    only = None
    if "-k" in sys.argv:
        only = sys.argv[sys.argv.index("-k") + 1]
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        if only and only not in fn.__name__:
            continue
        print(f"[ RUN ] {fn.__name__}")
        try:
            fn()
            print(f"[  OK ] {fn.__name__}")
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print(f"[FAIL ] {fn.__name__}: {type(exc).__name__}: {exc}")
    print("FAILED" if failed else "ALL PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
