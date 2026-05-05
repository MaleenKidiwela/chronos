"""Residuals-vs-fit diagnostic for v3 segment modeling.

Per inter-trigger segment, compute residual = (raw cleaned Δt) - (v3
modeled Δt) on the hours where raw Δt is finite. If the smoother is
adequately tracking real drift shape, residuals should look like
zero-mean noise consistent with the picker quantum (~±0.125 s).
Systematic structure (drift, curvature) in the residuals would mean the
smoother is leaving real timing signal on the floor.

Outputs:
- tests/out/v3_residuals_overview.png — full-timeline panel:
  raw cleaned Δt, v3 model overlaid, residuals below.
- tests/out/v3_residuals_per_segment.png — for the longest few segments
  in seconds-of-curvature, zoom-in panels.
- console summary: per-segment residual stats.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLOCK = Path("/home/seismic/chronos/data/clock_estimate/HYS14")
ARCH = Path("/home/seismic/chronos/data/results_archive")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def main() -> int:
    dt_raw = np.load(CLOCK / "delta_t_hourly_filtered_raw.npy")  # post-Hampel, pre-smoothing
    dt_v3 = np.load(CLOCK / "delta_t_hourly_clean.npy")          # v3 modeled
    trig = pd.read_csv(CLOCK / "trigger_periods.csv")

    n = len(dt_raw)
    t_days = np.arange(n) / 24.0

    # Build segment list mirroring filter_and_triggers.model_segments.
    segs: list[tuple[int, int]] = []
    cursor = 0
    for _, row in trig.sort_values("start_index").iterrows():
        s = int(row["start_index"]); e = int(row["end_index"])
        if s > cursor:
            segs.append((cursor, s))
        cursor = e + 1
    if cursor < n:
        segs.append((cursor, n))

    # Per-segment residual stats.
    print(f"{'segment':>10}  {'lo..hi':>14}  {'days':>6}  "
          f"{'n_valid':>8}  {'med_res':>8}  {'mad_res':>8}  "
          f"{'p95_res':>8}  {'curvature':>10}")
    rows = []
    for i, (lo, hi) in enumerate(segs):
        seg_raw = dt_raw[lo:hi]
        seg_mod = dt_v3[lo:hi]
        valid = np.isfinite(seg_raw) & np.isfinite(seg_mod)
        if valid.sum() < 5:
            continue
        res = seg_raw[valid] - seg_mod[valid]
        med = float(np.median(res))
        mad = float(np.median(np.abs(res - med)))
        p95 = float(np.percentile(np.abs(res - med), 95))
        # Curvature proxy: how much the model itself moves over the segment.
        curv = float(np.nanmax(seg_mod) - np.nanmin(seg_mod))
        days = (hi - lo) / 24.0
        rows.append({"i": i, "lo": lo, "hi": hi, "days": days,
                     "n_valid": int(valid.sum()), "med_res": med,
                     "mad_res": mad, "p95_res": p95, "curv": curv})
        print(f"{i:>10}  {lo:>5}..{hi:>5}  {days:6.1f}  {int(valid.sum()):>8}  "
              f"{med:+8.3f}  {mad:8.3f}  {p95:8.3f}  {curv:10.3f}")

    # Overview plot.
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    valid_raw = np.isfinite(dt_raw)
    valid_mod = np.isfinite(dt_v3)
    axes[0].scatter(t_days[valid_raw], dt_raw[valid_raw],
                    s=2, alpha=0.4, label="raw cleaned Δt (post-Hampel)")
    axes[0].plot(t_days[valid_mod], dt_v3[valid_mod],
                 lw=1.0, color="red", label="v3 modeled (rolling med + MA)")
    for _, row in trig.iterrows():
        axes[0].axvspan(row["start_day"], row["end_day"],
                        color="lightcoral", alpha=0.18)
        axes[1].axvspan(row["start_day"], row["end_day"],
                        color="lightcoral", alpha=0.18)
    axes[0].set_ylabel("Δt (s)")
    axes[0].set_title("v3 segment model vs raw cleaned Δt — full timeline")
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].grid(alpha=0.3)

    res_full = dt_raw - dt_v3
    valid = np.isfinite(res_full)
    axes[1].scatter(t_days[valid], res_full[valid], s=2, alpha=0.5)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].axhline(+0.125, color="gray", lw=0.5, ls="--", alpha=0.7,
                    label="±picker quantum (0.125 s)")
    axes[1].axhline(-0.125, color="gray", lw=0.5, ls="--", alpha=0.7)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_ylabel("residual (s)")
    axes[1].set_xlabel("Elapsed time (days)")
    axes[1].set_title("residual = raw cleaned Δt − v3 model")
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    p = OUT / "v3_residuals_overview.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"\nwrote {p}")

    # Per-segment zoom-ins for the segments with the most curvature
    # (those are the ones where linear v2 was failing).
    rows.sort(key=lambda r: -r["curv"])
    top = rows[:6]
    if top:
        fig, axes = plt.subplots(len(top), 1, figsize=(12, 2.2 * len(top)),
                                 sharey=False)
        if len(top) == 1:
            axes = [axes]
        for ax, r in zip(axes, top):
            lo, hi = r["lo"], r["hi"]
            t = t_days[lo:hi]
            raw = dt_raw[lo:hi]
            mod = dt_v3[lo:hi]
            ax.scatter(t[np.isfinite(raw)], raw[np.isfinite(raw)],
                       s=4, alpha=0.5, label="raw cleaned")
            ax.plot(t[np.isfinite(mod)], mod[np.isfinite(mod)],
                    lw=1.2, color="red", label="v3 model")
            ax.set_title(f"seg {r['i']}  days={r['days']:.1f}  "
                         f"curvature={r['curv']:.2f}s  "
                         f"med_res={r['med_res']:+.3f}  mad_res={r['mad_res']:.3f}  "
                         f"p95|res|={r['p95_res']:.3f}")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Elapsed time (days)")
        fig.tight_layout()
        p = OUT / "v3_residuals_per_segment.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
