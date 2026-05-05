"""Regenerate the chronfix-example figures (v3) bundled in
chronfix/examples/HYS14/. Run from /home/seismic/chronos."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CHRONOS = Path("/home/seismic/chronos")
CHRONFIX = Path("/home/seismic/chronfix")
CLOCK = CHRONOS / "data/clock_estimate/HYS14"
CCF_BEFORE = CHRONOS / "data/ccf/HYS12-HYS14"
CCF_AFTER = CHRONOS / "data/ccf/HYS12-HYS14_corrected"

OUT_FIG = CHRONFIX / "examples/HYS14/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)


def correction_function():
    dt = np.load(CLOCK / "delta_t_hourly_clean.npy")
    ht = np.load(CLOCK / "hour_times.npy").astype("datetime64[us]")
    trig = pd.read_csv(CLOCK / "trigger_periods.csv")
    t = ht.astype("datetime64[D]").astype("O")  # for matplotlib

    fig, ax = plt.subplots(figsize=(13, 4))
    valid = np.isfinite(dt)
    ax.plot(t[valid], dt[valid], lw=0.7, color="C0",
            label="Δt(t) applied to HYS14 timestamps")
    for _, row in trig.iterrows():
        x0 = ht[int(row["start_index"])].astype("O")
        x1 = ht[int(row["end_index"])].astype("O")
        ax.axvspan(x0, x1, color="lightcoral", alpha=0.30,
                   label="trigger interval" if _ == 0 else None)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Δt (s)")
    ax.set_xlabel("UTC date")
    ax.set_title("HYS14 correction function (v3 segment-modeled hourly Δt)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    p = OUT_FIG / "correction_function.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"wrote {p}")
    # mirror copy into chronos data tree
    (CLOCK / "correction_function.png").write_bytes(p.read_bytes())


def before_after_ccf():
    lags = np.load(CCF_BEFORE / "lags.npy")
    ref_b = np.load(CCF_BEFORE / "cc_ref.npy")
    ref_a = np.load(CCF_AFTER / "cc_ref.npy")
    daily_b = np.load(CCF_BEFORE / "cc_daily.npy")
    daily_a = np.load(CCF_AFTER / "cc_daily.npy")
    dates_b = np.load(CCF_BEFORE / "cc_dates.npy")
    dates_a = np.load(CCF_AFTER / "cc_dates.npy")

    rms_b = float(np.sqrt(np.nanmean(ref_b ** 2)))
    rms_a = float(np.sqrt(np.nanmean(ref_a ** 2)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                             gridspec_kw={"height_ratios": [1, 2]})
    axes[0, 0].plot(lags, ref_b, lw=0.7)
    axes[0, 0].set_title(f"Long-term reference: BEFORE  (RMS={rms_b:.2f})")
    axes[0, 0].set_xlim(lags.min(), lags.max())
    axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(lags, ref_a, lw=0.7, color="C2")
    axes[0, 1].set_title(f"Long-term reference: AFTER (v3)  (RMS={rms_a:.2f})")
    axes[0, 1].set_xlim(lags.min(), lags.max())
    axes[0, 1].grid(alpha=0.3)

    def heatmap(ax, daily, dates, title):
        # robust amplitude for symmetric color limits
        vmax = np.nanpercentile(np.abs(daily), 99.5)
        d_num = mdates.date2num(dates.astype("datetime64[D]").astype("O"))
        ax.imshow(daily, origin="lower", aspect="auto",
                  extent=[lags.min(), lags.max(), d_num.min(), d_num.max()],
                  vmin=-vmax, vmax=vmax, cmap="RdBu_r", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Lag (s)")
        ax.yaxis_date()
        ax.yaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    heatmap(axes[1, 0], daily_b, dates_b, "Daily CCF: BEFORE")
    heatmap(axes[1, 1], daily_a, dates_a, "Daily CCF: AFTER (v3)")
    axes[0, 0].set_xlabel("Lag (s)")
    axes[0, 1].set_xlabel("Lag (s)")

    fig.suptitle("HYS12-HYS14 cross-correlations: before vs after chronfix (v3)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    p = OUT_FIG / "before_after_ccf.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"wrote {p}  (RMS {rms_b:.2f} -> {rms_a:.2f}, {rms_a/rms_b:.2f}x)")


def peak_lag_after():
    src = CHRONOS / "data/peak_lag_hourly/HYS12-HYS14_corrected/plot.png"
    dst = OUT_FIG / "peak_lag_after.png"
    dst.write_bytes(src.read_bytes())
    print(f"copied {src} -> {dst}")


def peak_lag_before():
    src = CHRONOS / "data/peak_lag_hourly/HYS12-HYS14/plot.png"
    dst = OUT_FIG / "peak_lag_before.png"
    dst.write_bytes(src.read_bytes())
    print(f"copied {src} -> {dst}")


if __name__ == "__main__":
    correction_function()
    before_after_ccf()
    peak_lag_after()
    peak_lag_before()
