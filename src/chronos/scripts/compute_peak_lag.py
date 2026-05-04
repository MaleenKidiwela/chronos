#!/usr/bin/env python
"""Per-hour ballistic peak lag (instead of per-day).

Re-stacks the 30-min CCs from `hys_ccf.py` into hourly bins, applies the
envelope-of-CC^2 picker per hour, and writes the resulting hourly peak-lag
time series.

The hourly stacks have ~24x lower SNR than the daily stacks because each
hour contains ~7-8 30-min CCs (with 75% overlap) versus ~190 per day. Use
this only on pairs whose daily reference is well above the noise floor;
otherwise the hourly picks are noise.

Inputs (from `hys_ccf.py`):
    data/ccf/<pair>/cc_30min.npy
    data/ccf/<pair>/cc_30min_times.npy   fractional day index of segment mids
    data/ccf/<pair>/cc_dates.npy         datetime64[D] day axis
    data/ccf/<pair>/lags.npy

Outputs (under data/peak_lag_hourly/<pair>/):
    cc_hourly.npy            (N_hours, n_lags) per-lag median per hour
    hour_times.npy           datetime64[h] timestamp per hour bin
    peak_lag_hourly_<side>.npy
    plot.png

Usage:
    python peak_lag_hourly.py --pair HYS12-HYS14
    python peak_lag_hourly.py --pair HYS14-HYSB1_lowband --side pos
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

CHRONOS_ROOT = Path("/home/seismic/chronos")
CC_ROOT = CHRONOS_ROOT / "data" / "ccf"
OUT_ROOT = CHRONOS_ROOT / "data" / "peak_lag_hourly"

LOG = logging.getLogger("peak_lag_hourly")


def stack_hourly(
    cc30: np.ndarray, times_days: np.ndarray, dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Median-stack 30-min CCs into hourly bins.

    Returns (cc_hourly, hour_times) where hour_times is datetime64[h].
    """
    hour_idx = np.floor(times_days * 24.0).astype(np.int64)
    unique_hours, inv = np.unique(hour_idx, return_inverse=True)
    n_h = len(unique_hours)
    n_lags = cc30.shape[1]

    cc_hourly = np.full((n_h, n_lags), np.nan, dtype=np.float32)
    # Group rows by hour bucket. argsort + segment slices is O(N log N) once
    # and avoids the O(N_h * N) of repeated boolean masks.
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    boundaries = np.concatenate(
        [[0], np.where(np.diff(sorted_inv) != 0)[0] + 1, [len(order)]]
    )
    for i in range(n_h):
        rows = order[boundaries[i]:boundaries[i + 1]]
        cc_hourly[i] = np.median(cc30[rows], axis=0)

    t0 = np.datetime64(str(dates[0]), "h")
    hour_times = t0 + unique_hours.astype("timedelta64[h]")
    return cc_hourly, hour_times


def envelope_squared(cc: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(cc.astype(np.float64) ** 2, axis=-1))


def argmax_in_window(env: np.ndarray, lags: np.ndarray, side: str) -> np.ndarray:
    if side == "global":
        sel = np.ones_like(lags, dtype=bool)
    elif side == "pos":
        sel = lags > 0
    elif side == "neg":
        sel = lags < 0
    else:
        raise ValueError(side)
    sel_idx = np.where(sel)[0]
    sub = env[:, sel]
    local = np.argmax(sub, axis=-1)
    out = lags[sel_idx[local]].astype(np.float64)
    invalid = np.all(np.isnan(env), axis=-1) | np.all(env == 0, axis=-1)
    out[invalid] = np.nan
    return out


def run(pair_tag: str, side: str) -> None:
    in_dir = CC_ROOT / pair_tag
    LOG.info("[%s] loading cc_30min...", pair_tag)
    cc30 = np.load(in_dir / "cc_30min.npy")
    times = np.load(in_dir / "cc_30min_times.npy")
    lags = np.load(in_dir / "lags.npy")
    dates = np.load(in_dir / "cc_dates.npy")
    LOG.info("[%s] cc_30min=%s", pair_tag, cc30.shape)

    cc_hourly, hour_times = stack_hourly(cc30, times, dates)
    LOG.info("[%s] hourly bins: %d", pair_tag, cc_hourly.shape[0])

    env = envelope_squared(cc_hourly)
    peak = argmax_in_window(env, lags, side)

    out_dir = OUT_ROOT / pair_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cc_hourly.npy", cc_hourly)
    np.save(out_dir / "hour_times.npy", hour_times)
    np.save(out_dir / f"peak_lag_hourly_{side}.npy", peak)
    LOG.info("[%s] wrote outputs to %s", pair_tag, out_dir)

    _plot(out_dir, pair_tag, hour_times, peak, side)


def _plot(out_dir, pair_tag, hour_times, peak, side):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = {"global": "global peak", "pos": "causal", "neg": "acausal"}[side]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    valid = ~np.isnan(peak)
    ax.plot(np.arange(len(hour_times))[valid], peak[valid],
            ".", ms=1.5, color="C0", label=label)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Peak lag (s)")
    ax.set_title(f"{pair_tag}  hourly ballistic peak lag ({side})")
    ax.legend(loc="upper right", fontsize=8, markerscale=3)
    n = len(hour_times)
    tick_idx = np.linspace(0, n - 1, min(10, n)).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [str(hour_times[i].astype("datetime64[D]")) for i in tick_idx],
        rotation=30, ha="right",
    )
    fig.tight_layout()
    out = out_dir / "plot.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    LOG.info("[%s] plot -> %s", pair_tag, out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pair", default="HYS12-HYS14")
    p.add_argument("--side", choices=["global", "pos", "neg"], default="global")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args.pair, args.side)
    return 0


if __name__ == "__main__":
    sys.exit(main())
