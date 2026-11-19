"""Regression tests for ClockModel.interp_delta_t and correct._resample.

Run:  python -m pytest chronfix/tests -q
"""
from __future__ import annotations

import numpy as np
from obspy import Trace, UTCDateTime

from chronfix.clock_model import ClockModel
from chronfix.correct import correct_trace

H = np.timedelta64(1, "h")
T0 = np.datetime64("2024-01-01T00", "h")


def _model(dt, triggers=()):
    """dt: per-hour values from T0. triggers: (start_idx, end_idx) pairs."""
    dt = np.asarray(dt, dtype=float)
    hours = T0 + np.arange(len(dt)) * H
    for a, b in triggers:
        dt[a:b + 1] = np.nan
    ts = np.array([hours[a] for a, _ in triggers], dtype="datetime64[s]")
    te = np.array([hours[b] for _, b in triggers], dtype="datetime64[s]")
    return ClockModel(hour_times=hours, delta_t=dt, trigger_starts=ts, trigger_ends=te)


def _q(hours_from_t0):
    return (T0.astype("datetime64[us]")
            + (np.atleast_1d(hours_from_t0) * 3.6e9).astype("timedelta64[us]"))


# ----------------------------- interp_delta_t -----------------------------

def test_linear_between_adjacent_hours():
    m = _model(np.arange(10.0))
    np.testing.assert_allclose(m.interp_delta_t(_q([0, 2.5, 7.25])), [0, 2.5, 7.25])


def test_does_not_bridge_long_nan_gap():
    dt = np.zeros(48)
    dt[10:30] = np.nan           # 20-h undefined stretch (the 2022-11-08 case)
    dt[30:] = 60.0
    m = _model(dt)
    out = m.interp_delta_t(_q([9.5, 10.0, 10.5, 20.0, 29.0, 29.5]))
    assert out[0] == 0.0 and out[1] == 0.0     # edge hold <= 1 h after hour 9
    assert np.isnan(out[2]) and np.isnan(out[3])
    assert out[4] == 60.0 and out[5] == 60.0   # edge hold <= 1 h before hour 30


def test_short_gap_is_bridged():
    dt = np.arange(10.0)
    dt[4:6] = np.nan                       # finite at 3 and 6: 3 h apart
    m = _model(dt)
    np.testing.assert_allclose(m.interp_delta_t(_q([4.5])), [4.5])


def test_never_interpolates_across_trigger():
    dt = np.r_[np.full(10, 7.0), np.zeros(10)]
    m = _model(dt, triggers=[(10, 10)])    # one-hour trigger at hour 10
    out = m.interp_delta_t(_q([9.5, 10.0, 10.5, 11.0]))
    assert out[0] == 7.0                   # held from the left, not blended with 0
    assert np.isnan(out[1])                # inside the trigger
    assert out[2] == 0.0 and out[3] == 0.0 # held from the right, not blended with 7


def test_scalar_and_utcdatetime_input():
    m = _model(np.arange(10.0))
    assert m.interp_delta_t(UTCDateTime("2024-01-01T02:30:00")) == 2.5


# ------------------------------- _resample -------------------------------

FS = 8.0


def _signal(t):
    """Smooth, band-limited test signal of true UTC time t (s)."""
    return np.sin(2 * np.pi * 0.37 * t) + 0.5 * np.sin(2 * np.pi * 1.13 * t + 1.0)


def _recorded_trace(true_dt, utc0, n, dtype=np.float64):
    """Trace recorded by a clock with error true_dt(utc) (s): a sample taken
    at true time u is stamped u + true_dt(u). Stamps are regular (fs grid in
    apparent time), so solve u + true_dt(u) = stamp for each stamp."""
    app_start = utc0 + true_dt(utc0)
    stamps = app_start + np.arange(n) / FS
    u = stamps - true_dt(stamps)
    for _ in range(6):
        u = stamps - true_dt(u)
    tr = Trace(data=_signal(u - utc0).astype(dtype))
    tr.stats.sampling_rate = FS
    tr.stats.starttime = UTCDateTime(app_start)
    return tr


def test_resample_recovers_true_timing_and_grid():
    # Drift 0 -> 2 s over 6 h, i.e. a large, smooth clock error.
    m = _model(np.linspace(0.0, 2.0, 7))
    t_h0 = UTCDateTime("2024-01-01T00:00:00").timestamp
    true_dt = lambda u: (np.asarray(u) - t_h0) / 3600.0 * (2.0 / 6.0)
    utc0 = UTCDateTime("2024-01-01T00:30:00").timestamp + 0.0371  # off-grid
    tr = _recorded_trace(true_dt, utc0, int(4 * 3600 * FS))
    out = correct_trace(tr, m)
    assert len(out) == 1
    o = out[0]
    # on the absolute k/fs grid
    assert o.stats.starttime.ns % int(1e9 / FS) == 0
    t = o.stats.starttime.timestamp + np.arange(o.stats.npts) / FS
    def rms(lag):
        e = o.data - _signal(t - utc0 + lag)
        return np.sqrt(np.mean(e[100:-100] ** 2))
    # Linear interpolation at 8 Hz leaves a small amplitude error (~0.025
    # here); timing is what matters: the misfit must be smallest at lag 0.
    assert rms(0.0) < 0.04
    assert rms(0.0) < rms(0.005) and rms(0.0) < rms(-0.005)


def test_resample_splits_at_model_gap_instead_of_truncating():
    dt = np.full(30, 3.0)
    dt[8:20] = np.nan                     # 12-h undefined stretch mid-chunk
    m = _model(dt)
    utc0 = UTCDateTime("2024-01-01T02:00:00").timestamp
    tr = _recorded_trace(lambda u: 3.0 + 0 * np.asarray(u), utc0, int(24 * 3600 * FS))
    out = correct_trace(tr, m)
    assert len(out) == 2                  # data after the gap is kept
    assert out[0].stats.endtime <= UTCDateTime("2024-01-01T09:00:00")
    assert out[1].stats.starttime >= UTCDateTime("2024-01-01T19:00:00")


def test_integer_data_is_rounded_not_truncated():
    m = _model(np.full(10, 0.0625))       # half-sample shift
    tr = Trace(data=np.array([0, 3] * 400, dtype=np.int32))
    tr.stats.sampling_rate = FS
    tr.stats.starttime = UTCDateTime("2024-01-01T01:00:00")
    o = correct_trace(tr, m)[0]
    assert o.data.dtype == np.int32
    # midpoint of 0 and 3 is 1.5 -> rint gives 2 (banker's) or 2; never 1
    assert set(np.unique(o.data[1:-1])) <= {2}
