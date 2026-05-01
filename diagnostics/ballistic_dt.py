#!/usr/bin/env python
"""Track lag-time deviation of the ballistic-wave peak across daily CCFs.

The standard OBS clock-drift diagnostic: for each daily CCF, locate the
ballistic peak and measure dt = peak_lag(day) - peak_lag(reference). A
constant offset = clock offset, a step = clock jump, a slope = clock drift.

Two methods:

1. **Peak-pick** — within a search window around the reference peak, return
   the lag of the daily-stack maximum (parabolic sub-sample interpolation).
   Fast, robust to amplitude, sensitive to lobe-hopping at low SNR.

2. **Stretching** — scan dilation factors eps; pick the eps that maximizes
   correlation between daily and reference within the coda window. Returns
   `dt = -eps * t_center` for the chosen window. More stable for drift-like
   timing errors but less obvious for jumps.

Inputs come from `hys_ccf.py` outputs at data/ccf/<pair_tag>/.

Outputs (per pair) at data/ballistic_dt/<pair_tag>/:
    dt_pos.npy, dt_neg.npy        peak-pick dt (causal / acausal), s
    cc_pos.npy, cc_neg.npy        max correlation in search window
    dt_stretch_pos.npy            stretching dt (causal), s
    cc_stretch_pos.npy            best correlation
    ref_peak_pos.npy / _neg.npy   reference peak lag (s) per side
    plot.png                      dt(t) figure

Usage:
    python ballistic_dt.py --pair HYS12-HYS14
    python ballistic_dt.py --pair HYS14-HYSB1 --window 1.5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

CHRONOS_ROOT = Path("/home/seismic/chronos")
CC_ROOT = CHRONOS_ROOT / "data" / "ccf"
OUT_ROOT = CHRONOS_ROOT / "data" / "ballistic_dt"

LOG = logging.getLogger("ballistic_dt")


# =========================== peak finding ===========================

def parabolic_subsample(y: np.ndarray, i: int) -> float:
    """Sub-sample shift via parabolic fit to (i-1, i, i+1). Returns delta in samples."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (a - c) / denom


def locate_ref_peak(ref: np.ndarray, lags: np.ndarray, side: str) -> tuple[float, int]:
    """Return (peak_lag_seconds, peak_index)."""
    if side == "pos":
        sel = lags > 0
    elif side == "neg":
        sel = lags < 0
    else:
        sel = np.ones_like(lags, dtype=bool)
    idx_local = int(np.argmax(np.abs(ref[sel])))
    idx_global = np.where(sel)[0][idx_local]
    delta = parabolic_subsample(np.abs(ref), idx_global)
    dt_samp = idx_global + delta
    fs_inv = lags[1] - lags[0]
    return float(lags[0] + dt_samp * fs_inv), idx_global


def peak_pick_dt(
    daily: np.ndarray, ref: np.ndarray, lags: np.ndarray,
    side: str, window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-day peak lag in a window around the reference peak."""
    ref_lag, ref_idx = locate_ref_peak(ref, lags, side)
    fs_inv = lags[1] - lags[0]
    win_n = int(round(window_s / fs_inv))
    lo = max(0, ref_idx - win_n)
    hi = min(len(lags), ref_idx + win_n + 1)
    sign = 1.0 if ref[ref_idx] >= 0 else -1.0  # follow polarity of reference

    n_days = daily.shape[0]
    dt = np.full(n_days, np.nan, dtype=np.float64)
    cc = np.full(n_days, np.nan, dtype=np.float64)
    ref_win = ref[lo:hi]
    ref_norm = np.linalg.norm(ref_win) + 1e-30
    for k in range(n_days):
        row = daily[k]
        if np.all(np.isnan(row)):
            continue
        seg = row[lo:hi]
        # peak by signed amplitude (so we don't flip onto a sidelobe)
        idx_local = int(np.argmax(sign * seg))
        idx_global = lo + idx_local
        delta = parabolic_subsample(sign * row, idx_global)
        dt[k] = (lags[0] + (idx_global + delta) * fs_inv) - ref_lag
        cc[k] = float(np.dot(seg, ref_win) / ((np.linalg.norm(seg) + 1e-30) * ref_norm))
    return dt, cc, ref_lag


# =========================== stretching ===========================

def stretch_dt(
    daily: np.ndarray, ref: np.ndarray, lags: np.ndarray,
    side: str, t_center: float, t_half: float,
    eps_max: float = 0.05, n_eps: int = 401,
) -> tuple[np.ndarray, np.ndarray]:
    """Stretching method on a fixed coda window (one side).

    Returns (dt_seconds_at_t_center, max_correlation).
    dt = -eps * t_center (positive eps stretches the trace, equivalent to
    delaying the medium / advancing the clock).
    """
    if side == "pos":
        t_lo, t_hi = t_center - t_half, t_center + t_half
    else:
        t_lo, t_hi = -(t_center + t_half), -(t_center - t_half)
    sel = (lags >= t_lo) & (lags <= t_hi)
    win_lags = lags[sel]
    ref_win = ref[sel]
    ref_norm = np.linalg.norm(ref_win) + 1e-30
    eps_grid = np.linspace(-eps_max, eps_max, n_eps)

    n_days = daily.shape[0]
    dt = np.full(n_days, np.nan, dtype=np.float64)
    cc = np.full(n_days, np.nan, dtype=np.float64)
    for k in range(n_days):
        row = daily[k]
        if np.all(np.isnan(row)):
            continue
        best_cc = -np.inf
        best_eps = 0.0
        for eps in eps_grid:
            sampled = np.interp(win_lags * (1.0 + eps), lags, row,
                                left=0.0, right=0.0)
            c = float(np.dot(sampled, ref_win) /
                      ((np.linalg.norm(sampled) + 1e-30) * ref_norm))
            if c > best_cc:
                best_cc = c
                best_eps = eps
        dt[k] = -best_eps * t_center
        cc[k] = best_cc
    return dt, cc


# =========================== driver ===========================

def run_pair(
    pair_tag: str, window_s: float, coda_t: float, coda_half: float,
    eps_max: float, do_stretch: bool, plot: bool,
) -> None:
    in_dir = CC_ROOT / pair_tag
    daily = np.load(in_dir / "cc_daily.npy")
    ref = np.load(in_dir / "cc_ref.npy")
    lags = np.load(in_dir / "lags.npy")
    dates = np.load(in_dir / "cc_dates.npy")

    out_dir = OUT_ROOT / pair_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("[%s] daily=%s lags=[%g,%g]s ref RMS=%.3g",
             pair_tag, daily.shape, lags[0], lags[-1], np.sqrt(np.mean(ref ** 2)))

    dt_pos, cc_pos, rp_pos = peak_pick_dt(daily, ref, lags, "pos", window_s)
    dt_neg, cc_neg, rp_neg = peak_pick_dt(daily, ref, lags, "neg", window_s)
    LOG.info("[%s] reference peak: pos=%.3fs neg=%.3fs", pair_tag, rp_pos, rp_neg)

    np.save(out_dir / "dt_pos.npy", dt_pos)
    np.save(out_dir / "dt_neg.npy", dt_neg)
    np.save(out_dir / "cc_pos.npy", cc_pos)
    np.save(out_dir / "cc_neg.npy", cc_neg)
    np.save(out_dir / "ref_peak_pos.npy", np.array([rp_pos]))
    np.save(out_dir / "ref_peak_neg.npy", np.array([rp_neg]))

    if do_stretch:
        ds_pos, cs_pos = stretch_dt(daily, ref, lags, "pos", coda_t, coda_half, eps_max)
        ds_neg, cs_neg = stretch_dt(daily, ref, lags, "neg", coda_t, coda_half, eps_max)
        np.save(out_dir / "dt_stretch_pos.npy", ds_pos)
        np.save(out_dir / "dt_stretch_neg.npy", ds_neg)
        np.save(out_dir / "cc_stretch_pos.npy", cs_pos)
        np.save(out_dir / "cc_stretch_neg.npy", cs_neg)

    if plot:
        _plot(out_dir, pair_tag, dates, daily, ref, lags,
              dt_pos, cc_pos, dt_neg, cc_neg, rp_pos, rp_neg, window_s)


def _plot(out_dir, pair_tag, dates, daily, ref, lags,
          dt_pos, cc_pos, dt_neg, cc_neg, rp_pos, rp_neg, window_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)

    # Reference + search windows
    ax = axes[0]
    ax.plot(lags, ref, lw=0.7)
    ax.axvspan(rp_pos - window_s, rp_pos + window_s, color="C1", alpha=0.2)
    ax.axvspan(rp_neg - window_s, rp_neg + window_s, color="C2", alpha=0.2)
    ax.axvline(rp_pos, color="C1", ls="--", lw=0.8)
    ax.axvline(rp_neg, color="C2", ls="--", lw=0.8)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Reference CC")
    ax.set_title(f"{pair_tag}  ref pos={rp_pos:.2f}s  neg={rp_neg:.2f}s")

    # 2D daily stack
    ax = axes[1]
    im = ax.imshow(
        daily, aspect="auto", origin="lower",
        extent=[lags[0], lags[-1], 0, len(dates)],
        cmap="RdBu_r",
        vmin=-np.nanpercentile(np.abs(daily), 99),
        vmax=np.nanpercentile(np.abs(daily), 99),
    )
    ax.axvline(rp_pos, color="k", ls="--", lw=0.6)
    ax.axvline(rp_neg, color="k", ls="--", lw=0.6)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Day index")
    fig.colorbar(im, ax=ax, fraction=0.02)

    # dt time series
    ax = axes[2]
    days = np.arange(len(dates))
    ax.plot(days, dt_pos, ".", ms=2, label="dt causal", color="C1")
    ax.plot(days, dt_neg, ".", ms=2, label="dt acausal", color="C2")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Day index (date axis below)")
    ax.set_ylabel("dt (s)")
    ax.set_title("Ballistic-peak deviation from reference (positive = clock late)")
    ax.legend(loc="upper right", fontsize=8)
    # sparse date ticks
    n = len(dates)
    tick_idx = np.linspace(0, n - 1, min(8, n)).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(dates[i]) for i in tick_idx], rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(out_dir / "plot.png", dpi=140)
    plt.close(fig)
    LOG.info("[%s] plot -> %s", pair_tag, out_dir / "plot.png")


# =========================== CLI ===========================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pair", action="append", required=False,
                   help="e.g. HYS12-HYS14; repeatable. Default: all under data/ccf/.")
    p.add_argument("--window", type=float, default=2.0,
                   help="Peak-search half-window around reference peak (s).")
    p.add_argument("--coda-t", type=float, default=15.0,
                   help="Coda-window center for stretching (s).")
    p.add_argument("--coda-half", type=float, default=10.0,
                   help="Coda-window half-width (s).")
    p.add_argument("--eps-max", type=float, default=0.05,
                   help="Stretching eps grid range (+/-).")
    p.add_argument("--no-stretch", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    pairs = args.pair or sorted(d.name for d in CC_ROOT.iterdir() if d.is_dir())
    if not pairs:
        LOG.error("No pairs found under %s", CC_ROOT)
        return 1
    for tag in pairs:
        run_pair(
            tag, args.window, args.coda_t, args.coda_half,
            args.eps_max,
            do_stretch=not args.no_stretch,
            plot=not args.no_plot,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
