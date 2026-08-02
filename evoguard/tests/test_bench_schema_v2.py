"""Tests for bench_v2 constants & schemas. Runnable as python -m or pytest."""
from __future__ import annotations

import importlib


def _bc():
    return importlib.import_module("evoguard.process.bench_constants")


def test_required_n_per_cell_default_is_100():
    assert _bc().REQUIRED_N_PER_CELL_DEFAULT == 100


def test_family_concentration_ceiling_value():
    assert _bc().FAMILY_CONCENTRATION_CEILING == 0.35


def test_shift_step_upper_bound_factor_value():
    assert _bc().SHIFT_STEP_UPPER_BOUND_FACTOR == 0.25


def test_dev_snapshot_threshold_ratio_value():
    assert _bc().DEV_SNAPSHOT_THRESHOLD_RATIO == 0.90


def test_public_release_threshold_ratio_value():
    assert _bc().PUBLIC_RELEASE_THRESHOLD_RATIO == 1.00


def test_legacy_unknown_sentinel_method_tag_is_empty_string():
    assert _bc().LEGACY_UNKNOWN_SENTINEL_METHOD_TAG == ""


def _main() -> int:
    fns = [(k,v) for k,v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for name, fn in fns:
        try:
            fn(); print(f"PASS {name}")
        except Exception as exc:
            print(f"FAIL {name}: {exc!r}"); failures.append(name)
    print(f"\n{len(fns)-len(failures)}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_main())
