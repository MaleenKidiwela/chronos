#!/usr/bin/env python
"""Per-day ballistic peak lag from the envelope of |CC|^2.

For each daily CCF, compute the Hilbert envelope of CC**2 and report the lag
of its maximum. No reference, no deviation; just the raw peak lag per day.

Default: global argmax across all lags. Use --side {pos,neg} to restrict the
search to one side of zero lag.

Usage:
    python peak_lag.py --pair HYS12-HYS14
    python peak_lag.py --pair HYS14-HYSB1_lowband --side pos
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
OUT_ROOT = CHRONOS_ROOT / "data" / "peak_lag"

LOG = logging.getLogger("peak_lag")


def envelope_squared(cc_daily: np.ndarray) -> np.ndarray:
    """|Hilbert(CC**2)| per day."""
    sq = cc_daily.astype(np.float64) ** 2
    return np.abs(hilbert(sq, axis=-1))


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
    global_idx = sel_idx[local]
    # Mask all-nan rows
    invalid = np.all(np.isnan(env), axis=-1) | np.all(env == 0, axis=-1)
    out = lags[global_idx].astype(np.float64)
    out[invalid] = np.nan
    return out


def run(pair_tag: str, side: str) -> None:
    in_dir = CC_ROOT / pair_tag
    cc = np.load(in_dir / "cc_daily.npy")
    lags = np.load(in_dir / "lags.npy")
    dates = np.load(in_dir / "cc_dates.npy")

    out_dir = OUT_ROOT / pair_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("[%s] daily=%s lags=[%g,%g]s side=%s",
             pair_tag, cc.shape, lags[0], lags[-1], side)

    env = envelope_squared(cc)
    peak = argmax_in_window(env, lags, side)
    np.save(out_dir / f"peak_lag_{side}.npy", peak)
    _plot(out_dir, pair_tag, dates, peak, side)


def _plot(out_dir: Path, pair_tag: str, dates: np.ndarray,
          peak: np.ndarray, side: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = {
        "global": "global peak",
        "pos": "causal (lag > 0)",
        "neg": "acausal (lag < 0)",
    }[side]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    days = np.arange(len(dates))
    valid = ~np.isnan(peak)
    ax.plot(days[valid], peak[valid], ".", ms=2.5, color="C0", label=label)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Peak lag (s)")
    ax.set_title(f"{pair_tag}  ballistic peak lag ({side}) from envelope of CC**2")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    n = len(dates)
    tick_idx = np.linspace(0, n - 1, min(10, n)).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(dates[i]) for i in tick_idx], rotation=30, ha="right")
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
