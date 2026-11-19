"""Tests for compute_ccf gap masking and filter_and_triggers excursion marking.

Run:  python -m pytest tests/test_gap_mask_and_excursions.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import chronos.scripts.compute_ccf as ccf
from chronos.scripts.filter_and_triggers import mark_excursions

FS = 8.0


def test_coverage_mask_marks_only_real_samples():
    # data 01:00-02:00 and 05:00-06:00 on a 24-h grid (seconds from day start)
    n = int(86400 * FS) + 1
    m = ccf._coverage_mask([(3600.0, 7200.0), (18000.0, 21600.0)], 0.0, FS, n)
    assert m.sum() == 2 * (3600 * 8 + 1)
    assert m[int(3600 * FS)] and m[int(7200 * FS)] and not m[int(7200 * FS) + 1]


def test_process_day_skips_mostly_gap_windows():
    rng = np.random.default_rng(0)
    n = int(6 * 3600 * FS)
    a = rng.standard_normal(n)
    b = np.roll(a, 3) + 0.1 * rng.standard_normal(n)
    cov_b = np.ones(n, bool)
    cov_b[int(2 * 3600 * FS):int(5 * 3600 * FS)] = False   # 3-h gap in b
    b_gap = b.copy()
    b_gap[~cov_b] = 1e-12 * rng.standard_normal((~cov_b).sum())  # filter residue
    full, _ = ccf.process_day(a, b, FS, 0.0)
    masked, t = ccf.process_day(a, b_gap, FS, 0.0, np.ones(n, bool), cov_b)
    assert len(masked) < len(full)
    # every kept window's midpoint lies where b has >= 50 % real data
    half = ccf.CC_LEN / 2
    for ti in np.asarray(t) * 86400:
        i0, i1 = int((ti - half) * FS), int((ti + half) * FS)
        assert cov_b[i0:i1].mean() >= ccf.MIN_WINDOW_COVERAGE


def _periods(rows):
    return pd.DataFrame(rows, columns=["start_day", "end_day", "duration_days",
                                       "start_index", "end_index",
                                       "max_abs_jump_in_period",
                                       "num_triggered_steps_merged"])


def test_coherent_excursion_marked_undefined():
    raw = np.zeros(100)
    raw[40:44] = -3.4                       # 4-h coherent excursion
    raw[70] = 25.0                          # single noisy pick
    raw[80:83] = [30.0, -20.0, 5.0]         # incoherent noisy run
    model = np.zeros(100)
    mask = np.zeros(100, bool)
    mask[[40, 41, 42, 43, 70, 80, 81, 82]] = True
    out, per, exc = mark_excursions(raw, model, mask, _periods([]))
    assert exc == [(40, 43, -3.4)]
    assert np.isnan(out[40:44]).all() and np.isfinite(out[44:]).all()
    # trigger runs to the start of the next good hour (sample h covers [h, h+1))
    assert per[["start_index", "end_index"]].values.tolist() == [[40, 44]]


def test_excursion_merges_with_adjacent_trigger():
    raw = np.zeros(100)
    raw[50:53] = 4.0
    model = np.zeros(100)
    model[53:60] = np.nan                   # existing trigger 53..59
    mask = np.zeros(100, bool)
    mask[50:53] = True
    per0 = _periods([[53 / 24, 59 / 24, 6 / 24, 53, 59, 2.0, 1]])
    _, per, exc = mark_excursions(raw, model, mask, per0)
    assert len(exc) == 1
    assert per[["start_index", "end_index"]].values.tolist() == [[50, 59]]
