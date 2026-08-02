"""Offline tests for preliminary.kld_stats on synthetic data. Run via:

    /ssd1/conda_envs/evoguard/bin/python -m preliminary.tests.test_kld_stats
"""
from __future__ import annotations
import os, sys, csv, tempfile

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import numpy as np
from preliminary.kld_stats import analyze, pairwise_mwu, kruskal_wallallis


def _write_synth_csv(path, groups):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trajectory_id","segment_id","bucket","kl_nats","obs_token_len","dropped"])
        w.writeheader()
        i = 0
        for bucket, vals in groups.items():
            for v in vals:
                w.writerow({"trajectory_id": i, "segment_id": 0, "bucket": bucket,
                            "kl_nats": f"{v:.4f}", "obs_token_len": 10, "dropped": False})
                i += 1


def test_kruskal_wallis_significant_for_separated_groups():
    g = {"A": [0.1]*20, "B": [5.0]*20, "C": [0.2]*20}
    H, p = kruskal_wallallis(g)
    assert p < 0.05, f"expected significant, H={H} p={p}"


def test_kruskal_wallas_null_for_identical_groups():
    g = {"A": [1.0]*15, "B": [1.0]*15, "C": [1.0]*15}
    _, p = kruskal_wallallis(g)
    assert p > 0.05


def test_pairwise_mwu_direction():
    g = {"AttackFail": [5.0]*15, "AttackSuccess": [0.1]*15}
    res = pairwise_mwu(g["AttackFail"], g["AttackSuccess"])
    assert res["p_two_sided"] < 0.05
    assert res["median_a"] > res["median_b"]


def test_analyze_end_to_end_on_synth():
    groups = {
        "NormalClean": np.random.default_rng(1).normal(0.2, 0.1, 30).tolist(),
        "AttackSuccess": np.random.default_rng(2).normal(0.3, 0.1, 30).tolist(),
        "AttackFail": np.random.default_rng(3).normal(1.5, 0.4, 15).tolist(),
    }
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "kl_per_scenario.csv")
        _write_synth_csv(csv_path, groups)
        out = analyze(csv_path, output_dir=td)
        assert "kruskal_wallis" in out
        assert "pairwise" in out
        assert out["n_per_bucket"]["AttackFail"] == 15
        # AttackFail vs AttackSuccess should be significant & directional.
        af_as = out["pairwise"].get("AttackFail_vs_AttackSuccess")
        assert af_as is not None
        assert af_as["p_one_sided_greater"] < 0.05


def main():
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1; print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1; import traceback; print(f"FAIL {t.__name__}: {e}\n{traceback.format_exc(limit=3)}")
        except Exception as e:
            failed += 1; import traceback; print(f"ERROR {t.__name__}: {e!r}\n{traceback.format_exc(limit=3)}")
    print(f"\n=== SUMMARY === pass={passed} fail/error={failed} total={len(tests)}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
