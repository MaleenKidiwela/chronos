"""Apply a ClockModel to ObsPy traces.

Splits each input trace at trigger boundaries (real clock discontinuities)
and writes one corrected sub-trace per stable segment overlap. Two methods:

- ``method="resample"`` (default, recommended for hourly Δt input):
  resamples the trace data onto a regular UTC grid, accounting for
  within-segment drift via linear interp of Δt(t).

- ``method="shift_only"``: subtracts Δt at the chunk start from the
  starttime; data is bit-identical. Within-chunk drift is approximated
  as constant. Useful for spot-check / validation.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from obspy import Trace, Stream, UTCDateTime

from chronfix.clock_model import ClockModel


def _to_utc(t: np.datetime64) -> UTCDateTime:
    return UTCDateTime(t.astype("datetime64[us]").astype(str))


def correct_trace(
    tr: Trace,
    model: ClockModel,
    method: Literal["resample", "shift_only"] = "resample",
    interpolator: Literal["linear"] = "linear",
) -> list[Trace]:
    """Return one corrected sub-trace per stable-segment overlap.

    The output traces have:
        - starttime aligned to true UTC (resample) or apparent-start - Δt
          (shift_only).
        - Same nominal sampling_rate as input.
        - Data dtype preserved.

    Days where Δt is unavailable (NaN throughout the segment) yield no
    output.
    """
    if tr.stats.npts == 0:
        return []

    fs = float(tr.stats.sampling_rate)
    t_apparent_start = tr.stats.starttime
    t_apparent_end = tr.stats.endtime

    # Find stable intervals overlapping the trace's apparent time range.
    stables = model.stable_intervals(t_apparent_start.datetime, t_apparent_end.datetime)
    out: list[Trace] = []

    for stable_t0, stable_t1 in stables:
        s0 = _to_utc(stable_t0)
        s1 = _to_utc(stable_t1)
        sub_t0 = max(s0, t_apparent_start)
        sub_t1 = min(s1, t_apparent_end)
        if sub_t1 <= sub_t0:
            continue

        try:
            sub_tr = tr.slice(sub_t0, sub_t1, nearest_sample=False)
        except Exception:
            continue
        if sub_tr.stats.npts < 2:
            continue

        if method == "shift_only":
            corrected = _shift_only(sub_tr, model)
            if corrected is not None:
                out.append(corrected)
        elif method == "resample":
            out.extend(_resample(sub_tr, model))
        else:
            raise ValueError(f"unknown method: {method!r}")

    return out


def correct_stream(
    st: Stream,
    model: ClockModel,
    method: Literal["resample", "shift_only"] = "resample",
) -> Stream:
    """Apply correct_trace to every trace and return a flat Stream of chunks."""
    chunks: list[Trace] = []
    for tr in st:
        chunks.extend(correct_trace(tr, model, method=method))
    return Stream(chunks)


# --------------------------- internals ---------------------------

def _shift_only(tr: Trace, model: ClockModel) -> Trace | None:
    """Subtract Δt at the trace start from starttime; leave data alone."""
    dt0 = model.interp_delta_t(np.datetime64(tr.stats.starttime.datetime, "us"))
    if not np.isfinite(dt0):
        return None
    new = tr.copy()
    new.stats.starttime = tr.stats.starttime - float(dt0)
    return new


def _utc_grid_start(t: UTCDateTime, fs: float) -> UTCDateTime:
    """First time >= t on the absolute sample grid k / fs (k integer).

    Putting corrected output on the canonical grid means downstream tools
    that snap to the nearest sample (e.g. ``trim(nearest_sample=True)``)
    do not add up to half a sample of timing error. Falls back to ``t``
    when 1/fs is not a whole number of nanoseconds.
    """
    period_ns = 1e9 / fs
    if abs(period_ns - round(period_ns)) > 1e-6:
        return t
    period_ns = int(round(period_ns))
    k = -(-t.ns // period_ns)  # ceil division
    return UTCDateTime(ns=k * period_ns)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of the True runs in a boolean array."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _resample(tr: Trace, model: ClockModel) -> list[Trace]:
    """Resample the trace onto a regular UTC grid using linear interp.

    For each output sample at true UTC time `t_utc` (on the absolute
    k / fs grid), look up the input trace at apparent time
    `t_apparent ≈ t_utc + Δt(t_utc)`. Output samples where Δt is
    undefined or the target falls outside the input are dropped, and the
    output is split into one trace per contiguous valid run.
    """
    fs = float(tr.stats.sampling_rate)
    n_in = tr.stats.npts
    if n_in < 2:
        return []

    apparent_start = tr.stats.starttime
    app_offsets = np.arange(n_in, dtype=np.float64) / fs
    app_start_us = np.datetime64(apparent_start.datetime, "us")

    # Δt along the apparent axis, to find the UTC span this chunk maps to.
    # Every 8th sample is plenty for the span; endpoints always included.
    probe = np.unique(np.r_[np.arange(0, n_in, 8), n_in - 1])
    dt_probe = model.interp_delta_t(
        app_start_us + (app_offsets[probe] * 1e6).astype("timedelta64[us]"))
    ok = np.isfinite(dt_probe)
    if not ok.any():
        return []
    utc_offsets_probe = app_offsets[probe][ok] - dt_probe[ok]   # rel. apparent_start
    utc_lo = apparent_start + float(utc_offsets_probe.min())
    utc_hi = apparent_start + float(utc_offsets_probe.max())

    utc_start = _utc_grid_start(utc_lo, fs)
    if utc_start > utc_hi:
        return []
    n_out = int(np.floor((utc_hi - utc_start) * fs + 1e-6)) + 1
    if n_out < 2:
        return []
    t_utc_offsets = np.arange(n_out, dtype=np.float64) / fs  # rel. utc_start

    utc_start_us = np.datetime64(utc_start.datetime, "us")
    dt_at_utc = model.interp_delta_t(
        utc_start_us + (t_utc_offsets * 1e6).astype("timedelta64[us]"))

    # Apparent-time targets, relative to apparent_start.
    targets = t_utc_offsets + dt_at_utc + float(utc_start - apparent_start)

    # Snap float-precision overshoots at the input boundaries back onto the
    # input range so the first/last output samples are not lost.
    eps = 0.5 / fs
    last = app_offsets[-1]
    targets = np.where((targets < 0) & (targets > -eps), 0.0, targets)
    targets = np.where((targets > last) & (targets < last + eps), last, targets)

    valid = np.isfinite(targets) & (targets >= 0) & (targets <= last)
    data_in = np.asarray(tr.data, dtype=np.float64)
    is_int = np.issubdtype(tr.data.dtype, np.integer)

    out: list[Trace] = []
    for a, b in _runs(valid):
        if b - a < 2:
            continue
        vals = np.interp(targets[a:b], app_offsets, data_in)
        if is_int:
            vals = np.rint(vals)
        new = Trace(data=vals.astype(tr.data.dtype, copy=False))
        new.stats = tr.stats.copy()
        new.stats.starttime = utc_start + a / fs
        new.stats.npts = b - a
        out.append(new)
    return out
