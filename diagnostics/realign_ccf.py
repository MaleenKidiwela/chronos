#!/usr/bin/env python
"""Realign daily CCFs so the ballistic peak sits at a fixed anchor lag.

For each day, computes a time shift = anchor_lag - measured_peak_lag and
applies it to that day's CCF via linear interpolation on the lag axis. The
result is a corrected daily-stack tensor in which the ballistic packet sits
at the anchor lag on every day, which is the realignment that any waveform-
domain timing correction must achieve.

Inputs (from `hys_ccf.py` and `peak_lag.py`):
    data/ccf/<pair>/cc_daily.npy
    data/ccf/<pair>/lags.npy
    data/ccf/<pair>/cc_dates.npy
    data/peak_lag/<pair>/peak_lag_global.npy

Outputs (under data/ccf_realigned/<pair>/):
    cc_daily.npy        corrected daily-stack tensor
    cc_ref.npy          corrected reference (linear stack of corrected days)
    lags.npy            unchanged lag axis (copied through)
    cc_dates.npy        unchanged date axis (copied through)
    shifts.npy          per-day shift in seconds (NaN where excluded)

A diagnostic figure data/peak_lag/<pair>/realigned_overview.png compares the
original and corrected 2D heatmaps and the per-day applied shifts.

The anchor lag defaults to the median peak lag over a stable window at the
end of the time series (where the clock issue appears to be resolved).
Override with --anchor.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

CHRONOS_ROOT = Path("/home/seismic/chronos")
CC_ROOT = CHRONOS_ROOT / "data" / "ccf"
PEAK_ROOT = CHRONOS_ROOT / "data" / "peak_lag"
OUT_ROOT = CHRONOS_ROOT / "data" / "ccf_realigned"

LOG = logging.getLogger("realign_ccf")


def shift_row(cc: np.ndarray, lags: np.ndarray, shift: float) -> np.ndarray:
    """Linear-interp time shift: corrected(tau) = original(tau - shift).

    A positive shift moves the waveform toward later lag.
    Out-of-range samples are filled with 0.
    """
    return np.interp(lags - shift, lags, cc, left=0.0, right=0.0)


def auto_anchor(
    peak: np.ndarray, lags: np.ndarray, ref_window: int = 90,
) -> float:
    """Median of peak lag over the last `ref_window` valid days, excluding
    boundary-clipped picks (where the peak landed at +/- maxlag)."""
    half = float(lags[-1])  # +maxlag
    valid = ~np.isnan(peak) & (np.abs(peak) < 0.99 * half)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        raise RuntimeError("no valid peak-lag picks to anchor on")
    tail = idx[-min(ref_window, len(idx)):]
    return float(np.median(peak[tail]))


def realign(
    pair_tag: str, anchor: float | None, max_shift: float, ref_window: int,
) -> None:
    in_dir = CC_ROOT / pair_tag
    cc_daily = np.load(in_dir / "cc_daily.npy")
    lags = np.load(in_dir / "lags.npy")
    dates = np.load(in_dir / "cc_dates.npy")
    peak = np.load(PEAK_ROOT / pair_tag / "peak_lag_global.npy")

    if anchor is None:
        anchor = auto_anchor(peak, lags, ref_window=ref_window)
    LOG.info("[%s] anchor lag = %+0.3f s", pair_tag, anchor)

    half = float(lags[-1])
    shifts = anchor - peak  # what we want to apply
    # Exclude days whose pick was clipped at the analysis-window edge or
    # whose required shift would slide the waveform past the edge.
    clipped = np.abs(peak) > 0.99 * half
    too_far = np.abs(shifts) > max_shift
    bad = np.isnan(peak) | clipped | too_far
    shifts[bad] = np.nan
    LOG.info(
        "[%s] applying shifts: %d days good, %d clipped/over-range/missing",
        pair_tag, int((~bad).sum()), int(bad.sum()),
    )

    out = np.full_like(cc_daily, np.nan)
    for i in range(cc_daily.shape[0]):
        if bad[i] or np.all(np.isnan(cc_daily[i])):
            continue
        out[i] = shift_row(cc_daily[i].astype(np.float64), lags, shifts[i]).astype(
            cc_daily.dtype, copy=False
        )

    valid_rows = ~np.isnan(out).any(axis=1)
    cc_ref = np.nanmean(out[valid_rows], axis=0).astype(cc_daily.dtype)

    out_dir = OUT_ROOT / pair_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cc_daily.npy", out)
    np.save(out_dir / "cc_ref.npy", cc_ref)
    np.save(out_dir / "lags.npy", lags)
    np.save(out_dir / "cc_dates.npy", dates)
    np.save(out_dir / "shifts.npy", shifts)
    LOG.info("[%s] wrote corrected stack to %s", pair_tag, out_dir)

    _plot(pair_tag, dates, lags, cc_daily, out, shifts, anchor)


def _plot(pair_tag, dates, lags, cc_orig, cc_corr, shifts, anchor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    # Color scale shared between heatmaps
    vmax = np.nanpercentile(np.abs(cc_orig), 99)

    ax = axes[0]
    im = ax.imshow(
        cc_orig, aspect="auto", origin="lower",
        extent=[lags[0], lags[-1], 0, len(dates)],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax.set_title(f"{pair_tag}  original daily CCFs")
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Date")
    fig.colorbar(im, ax=ax, fraction=0.02)
    n = len(dates)
    tick_idx = np.linspace(0, n - 1, min(8, n)).astype(int)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([str(dates[i]) for i in tick_idx])

    ax = axes[1]
    im = ax.imshow(
        cc_corr, aspect="auto", origin="lower",
        extent=[lags[0], lags[-1], 0, len(dates)],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax.axvline(anchor, color="k", lw=0.7, ls="--")
    ax.set_title(f"realigned to anchor = {anchor:+.3f} s")
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Date")
    fig.colorbar(im, ax=ax, fraction=0.02)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([str(dates[i]) for i in tick_idx])

    ax = axes[2]
    days = np.arange(len(dates))
    valid = ~np.isnan(shifts)
    ax.plot(days[valid], shifts[valid], ".", ms=2.5, color="C0")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Applied shift (s)")
    ax.set_title("per-day shift = anchor - peak_lag (NaN excluded)")
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(dates[i]) for i in tick_idx], rotation=30, ha="right")

    fig.tight_layout()
    out = PEAK_ROOT / pair_tag / "realigned_overview.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    LOG.info("[%s] plot -> %s", pair_tag, out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pair", default="HYS12-HYS14")
    p.add_argument(
        "--anchor", type=float, default=None,
        help="Anchor lag in seconds. Default: auto-detect from late-window median.",
    )
    p.add_argument(
        "--max-shift", type=float, default=50.0,
        help="Days requiring |shift| larger than this are dropped (no correction applied).",
    )
    p.add_argument(
        "--ref-window", type=int, default=90,
        help="Number of trailing valid days used to auto-detect the anchor lag.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    realign(args.pair, args.anchor, args.max_shift, args.ref_window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
