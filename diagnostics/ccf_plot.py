#!/usr/bin/env python
"""Two-panel diagnostic plot for a CCF pair: reference stack + 2D daily heatmap.

Reads the outputs of `hys_ccf.py` from data/ccf/<pair>/ and writes
data/peak_lag/<pair>/ccf_overview.png.

Usage:
    python ccf_plot.py --pair HYS12-HYS14
    python ccf_plot.py --pair HYS14-HYSB1_lowband
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

CHRONOS_ROOT = Path("/home/seismic/chronos")
CC_ROOT = CHRONOS_ROOT / "data" / "ccf"
OUT_ROOT = CHRONOS_ROOT / "data" / "peak_lag"

LOG = logging.getLogger("ccf_plot")


def run(pair_tag: str) -> None:
    in_dir = CC_ROOT / pair_tag
    daily = np.load(in_dir / "cc_daily.npy")
    ref = np.load(in_dir / "cc_ref.npy")
    lags = np.load(in_dir / "lags.npy")
    dates = np.load(in_dir / "cc_dates.npy")

    out_dir = OUT_ROOT / pair_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    ax = axes[0]
    ax.plot(lags, ref, lw=0.7)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Reference CC")
    ax.set_title(f"{pair_tag}  long-term reference stack (RMS={np.sqrt(np.mean(ref**2)):.3g})")

    ax = axes[1]
    vmax = np.nanpercentile(np.abs(daily), 99)
    im = ax.imshow(
        daily, aspect="auto", origin="lower",
        extent=[lags[0], lags[-1], 0, len(dates)],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Day index")
    fig.colorbar(im, ax=ax, fraction=0.02)
    n = len(dates)
    tick_idx = np.linspace(0, n - 1, min(8, n)).astype(int)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([str(dates[i]) for i in tick_idx])

    fig.tight_layout()
    out = out_dir / "ccf_overview.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    LOG.info("[%s] plot -> %s", pair_tag, out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pair", default="HYS12-HYS14")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.pair)
    return 0


if __name__ == "__main__":
    sys.exit(main())
