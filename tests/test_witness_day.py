"""Step 4 diagnostic: compare hourly CCF for one witness hour across both
pipelines (uncorrected vs. chronfix-corrected) to localize what is
degrading the per-hour CCF.

Witness picked in the conversation: 2023-10-11 hour 13 UTC. Uncorrected
peak-lag = +11.25 s (riding the drift ramp), corrected = -49.875 s
(jumped to a far noise lobe).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/seismic/chronos/data/peak_lag_hourly")
PAIR_UC = ROOT / "HYS12-HYS14"
PAIR_CO = ROOT / "HYS12-HYS14_corrected"
OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)

WITNESS = np.datetime64("2023-10-11T13", "h")


def load_pair(pair_dir: Path):
    cc = np.load(pair_dir / "cc_hourly.npy")
    ht = np.load(pair_dir / "hour_times.npy")
    # lags axis is implicit; rebuild from compute_ccf.MAXLAG=60, fs=8.
    fs = 8.0
    half = int(round(60.0 * fs))
    lags = np.arange(-half, half + 1) / fs
    assert cc.shape[1] == lags.size, (cc.shape, lags.size)
    return cc, ht, lags


def main() -> int:
    cc_u, ht_u, lags = load_pair(PAIR_UC)
    cc_c, ht_c, _ = load_pair(PAIR_CO)

    iu = np.searchsorted(ht_u, WITNESS)
    ic = np.searchsorted(ht_c, WITNESS)
    assert ht_u[iu] == WITNESS, ht_u[iu]
    assert ht_c[ic] == WITNESS, ht_c[ic]

    a = cc_u[iu]
    b = cc_c[ic]
    print(f"witness hour: {WITNESS}")
    print(f"uncorrected: max|cc|={np.nanmax(np.abs(a)):.4f}, argmax_lag={lags[np.nanargmax(np.abs(a))]:+.3f}s")
    print(f"corrected  : max|cc|={np.nanmax(np.abs(b)):.4f}, argmax_lag={lags[np.nanargmax(np.abs(b))]:+.3f}s")

    # Envelope^2 (what compute_peak_lag actually picks on)
    from scipy.signal import hilbert
    env_u = np.abs(hilbert(a)) ** 2
    env_c = np.abs(hilbert(b)) ** 2

    # Plot.
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(lags, a, lw=0.8, label="raw cc")
    axes[0].plot(lags, env_u / env_u.max() * np.abs(a).max(),
                 lw=0.8, alpha=0.6, label="env² (norm)")
    axes[0].set_title(f"UNCORRECTED  hour={WITNESS}  picked={lags[np.argmax(env_u)]:+.3f}s")
    axes[0].axvline(lags[np.argmax(env_u)], color="r", lw=0.7)
    axes[0].legend(loc="upper right")

    axes[1].plot(lags, b, lw=0.8, label="raw cc")
    axes[1].plot(lags, env_c / env_c.max() * np.abs(b).max(),
                 lw=0.8, alpha=0.6, label="env² (norm)")
    axes[1].set_title(f"CORRECTED    hour={WITNESS}  picked={lags[np.argmax(env_c)]:+.3f}s")
    axes[1].axvline(lags[np.argmax(env_c)], color="r", lw=0.7)
    axes[1].set_xlabel("lag (s)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    out = OUT_DIR / "witness_hour_cc.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    # Also: ratio of true-peak (near uncorrected pick) energy to global max.
    # In the corrected hour, was the near-zero peak still present but weaker?
    peak_u_lag = lags[np.argmax(env_u)]
    # Pick a small window around uncorrected lag in corrected env to see if peak survives.
    win_lo, win_hi = peak_u_lag - 2.0, peak_u_lag + 2.0
    sel = (lags >= win_lo) & (lags <= win_hi)
    if sel.any():
        local_peak_c = env_c[sel].max()
        global_peak_c = env_c.max()
        print(f"corrected env² near uncorrected pick ({peak_u_lag:+.2f}±2s): "
              f"local={local_peak_c:.4g} global={global_peak_c:.4g} "
              f"ratio={local_peak_c/global_peak_c:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
