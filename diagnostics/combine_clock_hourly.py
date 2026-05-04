#!/usr/bin/env python
"""Build the canonical hourly HYS14 clock-error estimate.

Same primary/cross-check structure as `combine_clock.py`, but at hourly
resolution. The primary estimate comes from HYS12-HYS14 hourly peak lags;
HYS14-HYSB1_lowband is reported only as a cross-check.

Pipeline:
    1. Load each pair's hourly peak-lag track and time axis.
    2. Anchor each track to its late-window median (clock fixed by ~May 2025).
    3. Convert peak-lag → shift = anchor - peak; sign-flip per pair so that
       Δt_HYS14 > 0 means HYS14 reports times "late" relative to true UTC.
    4. Align both pair estimates on a master hourly time axis.
    5. Pick primary = pair A only. Compute residual against pair B.
    6. Detect resync events as hours where |Δt[h+1] - Δt[h]| exceeds an
       absolute or relative threshold; emit those as segment boundaries
       for chronfix.

Inputs (from `peak_lag_hourly.py`):
    data/peak_lag_hourly/HYS12-HYS14/peak_lag_hourly_global.npy + hour_times.npy
    data/peak_lag_hourly/HYS14-HYSB1_lowband/peak_lag_hourly_global.npy + hour_times.npy

Outputs (under data/clock_estimate/HYS14/):
    delta_t_hourly.npy                 primary hourly Δt_HYS14 (s)
    hour_times.npy                     master hourly axis (datetime64[h])
    dt_hourly_from_HYS12_HYS14.npy     primary, sign-flipped
    dt_hourly_from_HYS14_HYSB1.npy     cross-check
    residual_hourly.npy                primary - cross-check, NaN otherwise
    segment_breaks.npy                 datetime64[h] of resync events
    plot_hourly.png                    4-panel diagnostic
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

CHRONOS_ROOT = Path("/home/seismic/chronos")
PEAK_HOURLY_ROOT = CHRONOS_ROOT / "data" / "peak_lag_hourly"
OUT_DIR = CHRONOS_ROOT / "data" / "clock_estimate" / "HYS14"

PAIR_A_TAG = "HYS12-HYS14"             # primary
PAIR_B_TAG = "HYS14-HYSB1_lowband"     # cross-check
SIGN_A = -1.0   # HYS14 is B-side: dt_HYS14 = -shifts
SIGN_B = +1.0   # HYS14 is A-side: dt_HYS14 = +shifts

LOG = logging.getLogger("combine_clock_hourly")


def load_pair(tag: str) -> tuple[np.ndarray, np.ndarray]:
    d = PEAK_HOURLY_ROOT / tag
    peak = np.load(d / "peak_lag_hourly_global.npy")
    times = np.load(d / "hour_times.npy")
    return peak, times


def anchor_lag(
    peak: np.ndarray, lag_extent: float, ref_window: int,
) -> float:
    """Median peak lag over the most recent valid hours, excluding picks
    clipped at the maxlag boundary."""
    valid = ~np.isnan(peak) & (np.abs(peak) < 0.99 * lag_extent)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        raise RuntimeError("no valid hourly peak picks for anchoring")
    tail = idx[-min(ref_window, len(idx)):]
    return float(np.median(peak[tail]))


def shifts_from_peak(
    peak: np.ndarray, lag_extent: float, ref_window: int, max_shift: float,
) -> tuple[np.ndarray, float]:
    anchor = anchor_lag(peak, lag_extent, ref_window)
    shifts = anchor - peak
    half = lag_extent
    bad = np.isnan(peak) | (np.abs(peak) > 0.99 * half) | (np.abs(shifts) > max_shift)
    shifts[bad] = np.nan
    return shifts, anchor


def align_on_master(
    estimates: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[np.ndarray]]:
    all_times = sorted(set().union(*(set(t.tolist()) for _, t in estimates)))
    master = np.array(all_times, dtype="datetime64[h]")
    aligned = []
    for arr, times in estimates:
        out = np.full(len(master), np.nan, dtype=np.float64)
        idx = np.searchsorted(master, times)
        out[idx] = arr
        aligned.append(out)
    return master, aligned


def detect_resyncs(
    times: np.ndarray, dt: np.ndarray,
    abs_jump_s: float, rel_jump_factor: float,
) -> np.ndarray:
    """Return datetime64[h] timestamps marking the start of each new segment
    (i.e. the hour just after the resync)."""
    valid = ~np.isnan(dt)
    vt = times[valid]
    vd = dt[valid]
    diffs = np.diff(vd)
    if len(diffs) == 0:
        return np.array([], dtype=times.dtype)
    # Use the 90th-percentile |diff| as the noise floor — robust to the median
    # being 0 in long calm stretches with quantised peak picks.
    ad = np.abs(diffs)
    ref_grad = float(np.percentile(ad[~np.isnan(ad)], 90))
    threshold = max(abs_jump_s, rel_jump_factor * ref_grad)
    jumps = ad >= threshold
    LOG.debug("resync detection threshold = %.3f s (abs=%g, rel*p90=%g)",
              threshold, abs_jump_s, rel_jump_factor * ref_grad)
    return vt[1:][jumps]


def run(
    abs_jump_s: float, rel_jump_factor: float,
    ref_window: int, max_shift: float,
) -> None:
    peak_a, times_a = load_pair(PAIR_A_TAG)
    peak_b, times_b = load_pair(PAIR_B_TAG)

    # The lag axis extent isn't carried in peak_lag_hourly outputs; recover
    # it from the cc/lags file for the same pair.
    lag_axis_a = np.load(CHRONOS_ROOT / "data" / "ccf" / PAIR_A_TAG / "lags.npy")
    lag_axis_b = np.load(CHRONOS_ROOT / "data" / "ccf" / PAIR_B_TAG / "lags.npy")
    extent_a, extent_b = float(lag_axis_a[-1]), float(lag_axis_b[-1])

    shifts_a, anchor_a = shifts_from_peak(peak_a, extent_a, ref_window, max_shift)
    shifts_b, anchor_b = shifts_from_peak(peak_b, extent_b, ref_window, max_shift)
    LOG.info("[%s] anchor = %+0.3f s", PAIR_A_TAG, anchor_a)
    LOG.info("[%s] anchor = %+0.3f s", PAIR_B_TAG, anchor_b)

    dt_a = SIGN_A * shifts_a
    dt_b = SIGN_B * shifts_b

    master, (a_m, b_m) = align_on_master([(dt_a, times_a), (dt_b, times_b)])

    # Primary = pair A only. Cross-check = pair B.
    delta_t = a_m.copy()
    valid_a, valid_b = ~np.isnan(a_m), ~np.isnan(b_m)
    both = valid_a & valid_b
    residual = np.full_like(a_m, np.nan)
    residual[both] = a_m[both] - b_m[both]
    res_rms = float(np.sqrt(np.nanmean(residual[both] ** 2))) if both.any() else float("nan")

    LOG.info(
        "primary=%s: %d valid; cross-check=%s: %d valid; both: %d",
        PAIR_A_TAG, int(valid_a.sum()), PAIR_B_TAG, int(valid_b.sum()),
        int(both.sum()),
    )
    LOG.info("residual RMS (primary - cross-check): %.3f s", res_rms)

    breaks = detect_resyncs(master, delta_t, abs_jump_s, rel_jump_factor)
    LOG.info("detected %d resync events with abs>=%g s or rel>=%g x median",
             len(breaks), abs_jump_s, rel_jump_factor)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "delta_t_hourly.npy", delta_t)
    np.save(OUT_DIR / "hour_times.npy", master)
    np.save(OUT_DIR / "dt_hourly_from_HYS12_HYS14.npy", a_m)
    np.save(OUT_DIR / "dt_hourly_from_HYS14_HYSB1.npy", b_m)
    np.save(OUT_DIR / "residual_hourly.npy", residual)
    np.save(OUT_DIR / "segment_breaks.npy", breaks)
    LOG.info("wrote outputs to %s", OUT_DIR)

    _plot(master, a_m, b_m, delta_t, residual, res_rms, breaks)


def _plot(times, a, b, combined, residual, res_rms, breaks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    x = np.arange(len(times))
    n = len(times)
    tick_idx = np.linspace(0, n - 1, min(10, n)).astype(int)
    tick_labels = [str(times[i].astype("datetime64[D]")) for i in tick_idx]

    # Helper: convert datetime64[h] break stamps to x indices
    if len(breaks):
        break_idx = np.searchsorted(times, breaks)
    else:
        break_idx = np.array([], dtype=int)

    ax = axes[0]
    va, vb = ~np.isnan(a), ~np.isnan(b)
    ax.plot(x[va], a[va], ".", ms=1.5, color="C0", label=f"{PAIR_A_TAG} primary")
    ax.plot(x[vb], b[vb], ".", ms=1.5, color="C3", label=f"{PAIR_B_TAG} cross-check")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel(r"$\Delta t_{HYS14}$ (s)")
    ax.set_title("Primary estimate and cross-check (both sign-flipped to HYS14 convention)")
    ax.legend(loc="upper right", fontsize=8, markerscale=3)

    ax = axes[1]
    vc = ~np.isnan(combined)
    ax.plot(x[vc], combined[vc], ".", ms=1.5, color="C2")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel(r"$\Delta t_{HYS14}$ (s)")
    ax.set_title(f"Canonical hourly Δt: primary ({PAIR_A_TAG}) only")

    ax = axes[2]
    vc = ~np.isnan(combined)
    ax.plot(x[vc], combined[vc], ".", ms=1.0, color="C2")
    for bi in break_idx:
        ax.axvline(bi, color="k", lw=0.5, alpha=0.3)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel(r"$\Delta t_{HYS14}$ (s)")
    ax.set_title(f"Canonical Δt with detected resync events ({len(breaks)} segment breaks)")

    ax = axes[3]
    vr = ~np.isnan(residual)
    ax.plot(x[vr], residual[vr], ".", ms=1.5, color="C4")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel("primary - cross-check (s)")
    ax.set_xlabel("Date")
    ax.set_title(f"Validation residual where both pairs valid  (RMS = {res_rms:.3f} s)")

    for ax in axes:
        ax.set_xticks(tick_idx)
    axes[-1].set_xticklabels(tick_labels, rotation=30, ha="right")

    fig.tight_layout()
    out = OUT_DIR / "plot_hourly.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    LOG.info("plot -> %s", out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--abs-jump", type=float, default=5.0,
                   help="Absolute hourly Δt jump (s) above which a resync is declared.")
    p.add_argument("--rel-jump-factor", type=float, default=10.0,
                   help="Hourly Δt jump must also exceed this many times the median |gradient|.")
    p.add_argument("--ref-window", type=int, default=24 * 30,
                   help="Trailing valid-hour window for anchor median (default 30 days of hours).")
    p.add_argument("--max-shift", type=float, default=50.0,
                   help="Drop hours whose required shift exceeds this magnitude.")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args.abs_jump, args.rel_jump_factor, args.ref_window, args.max_shift)
    return 0


if __name__ == "__main__":
    sys.exit(main())
