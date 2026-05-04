#!/usr/bin/env python3
"""Filter hourly Δt and detect threshold-triggered intervals.

Replaces the previous (mark_sudden_changes + clean_within_segments + find_resyncs)
workflow with a single multi-pass Hampel-style filter followed by interval
trigger detection on the differences between consecutive retained points.

Workflow:
    1. Load dt_hourly_from_HYS12_HYS14.npy (chronos primary, sign-flipped Δt).
    2. Apply final outlier filter:
       - 3-pass Hampel filter (windows 168 / 72 / 25 hours)
       - 3-point continuity residual pass
       - 1 extra "bit less-strict" 240-hour Hampel pass
    3. Difference consecutive retained points (NaNs ignored):
           Δdt_k = retained[k] - retained[k-1]
    4. Trigger when |Δdt_k| > threshold (default 1.0 s).
    5. Trigger interval is [t(k-1), t(k)]; merge overlapping intervals.
    6. Save:
       - filtered .npy
       - outlier mask .npy
       - trigger-period CSV
       - full-time trigger plot

Run from `/home/seismic/chronos`:

    python -m chronfix.scripts.filter_and_triggers
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Paths and parameters
# ============================================================

CLOCK_DIR = Path("/home/seismic/chronos/data/clock_estimate/HYS14")

INPUT_FILE = CLOCK_DIR / "dt_hourly_from_HYS12_HYS14.npy"

# Hourly data: convert sample index to elapsed days
SAMPLES_PER_DAY = 24.0

# Trigger threshold (seconds of |Δdt| between consecutive retained points)
TRIGGER_THRESHOLD = 1.0

# Plot/output names
FILTERED_NPY = CLOCK_DIR / "delta_t_hourly_clean.npy"
OUTLIER_MASK_NPY = CLOCK_DIR / "delta_t_hourly_outlier_mask.npy"
TRIGGER_CSV = CLOCK_DIR / "trigger_periods.csv"
PLOT_FILE = CLOCK_DIR / "filter_and_triggers.png"


# ============================================================
# Helper functions
# ============================================================

def hampel_mask(x, window, threshold_sigma=4.0, min_abs_deviation=0.5):
    """Rolling Hampel-style outlier detector."""
    s = pd.Series(x)

    rolling_median = s.rolling(
        window=window,
        center=True,
        min_periods=max(5, window // 4),
    ).median()

    residual = (s - rolling_median).abs()

    rolling_mad = residual.rolling(
        window=window,
        center=True,
        min_periods=max(5, window // 4),
    ).median()

    robust_sigma = 1.4826 * rolling_mad

    threshold = np.maximum(
        threshold_sigma * robust_sigma,
        min_abs_deviation,
    )

    return np.isfinite(x) & (residual.to_numpy() > threshold.to_numpy())


def apply_final_filter(dt_orig):
    """Final selected outlier filter.

    Stage 1: strict multi-pass Hampel filter (7d / 3d / 1d windows) plus a
    3-point continuity residual.
    Stage 2: extra 10-day local-MAD pass with sigma=6.5 and min_abs=1.30
    (the "bit less-strict" final pass).
    """
    dt_orig = np.asarray(dt_orig, dtype=float)

    # Stage 1: previous strict filter
    dt_strict = dt_orig.copy()
    strict_mask = np.zeros(len(dt_orig), dtype=bool)

    strict_passes = [
        {"window": 168, "threshold_sigma": 4.5, "min_abs_deviation": 0.75},  # 7 days
        {"window": 72,  "threshold_sigma": 4.0, "min_abs_deviation": 0.60},  # 3 days
        {"window": 25,  "threshold_sigma": 3.5, "min_abs_deviation": 0.50},  # ~1 day
    ]

    for p in strict_passes:
        m = hampel_mask(
            dt_strict,
            window=p["window"],
            threshold_sigma=p["threshold_sigma"],
            min_abs_deviation=p["min_abs_deviation"],
        )
        strict_mask |= m
        dt_strict[m] = np.nan

    # 3-point continuity residual
    neighbor_median = np.full_like(dt_strict, np.nan)
    for i in range(1, len(dt_strict) - 1):
        if np.isfinite(dt_strict[i - 1]) and np.isfinite(dt_strict[i + 1]):
            neighbor_median[i] = np.nanmedian([dt_strict[i - 1], dt_strict[i + 1]])

    continuity_residual = np.abs(dt_strict - neighbor_median)
    continuity_mask = (
        np.isfinite(dt_strict)
        & np.isfinite(neighbor_median)
        & (continuity_residual > 3.0)
    )
    strict_mask |= continuity_mask
    dt_strict[continuity_mask] = np.nan

    # Stage 2: final bit less-strict extra pass
    window = 240                  # 10 days for hourly data
    threshold_sigma = 6.5
    min_abs_deviation = 1.30

    s = pd.Series(dt_strict)

    rolling_median = s.rolling(
        window=window,
        center=True,
        min_periods=window // 5,
    ).median()

    residual = (s - rolling_median).abs()

    rolling_mad = residual.rolling(
        window=window,
        center=True,
        min_periods=window // 5,
    ).median()

    robust_sigma = 1.4826 * rolling_mad

    threshold = np.maximum(
        threshold_sigma * robust_sigma,
        min_abs_deviation,
    )

    extra_mask = np.isfinite(dt_strict) & (residual.to_numpy() > threshold.to_numpy())

    dt_clean = dt_strict.copy()
    dt_clean[extra_mask] = np.nan

    final_mask = np.isfinite(dt_orig) & ~np.isfinite(dt_clean)

    return dt_clean, final_mask, strict_mask, extra_mask


def compute_trigger_periods(dt_clean, threshold=1.0, samples_per_day=24.0):
    """Compute trigger intervals from differences of consecutive retained points."""
    dt_clean = np.asarray(dt_clean, dtype=float)
    sample_index = np.arange(len(dt_clean))
    t_days = sample_index / samples_per_day

    valid_idx = np.where(np.isfinite(dt_clean))[0]
    valid_t = t_days[valid_idx]
    valid_dt = dt_clean[valid_idx]

    diff_ignore_nans = np.full(len(valid_dt), np.nan)
    diff_ignore_nans[1:] = valid_dt[1:] - valid_dt[:-1]
    trigger = np.abs(diff_ignore_nans) > threshold

    intervals = []
    for k in range(1, len(valid_dt)):
        if trigger[k]:
            intervals.append((
                valid_t[k - 1],
                valid_t[k],
                int(valid_idx[k - 1]),
                int(valid_idx[k]),
                float(diff_ignore_nans[k]),
            ))

    merged = []
    for start, end, idx0, idx1, jump in intervals:
        if not merged:
            merged.append([start, end, idx0, idx1, [jump]])
        else:
            last_start, last_end, last_idx0, last_idx1, last_jumps = merged[-1]
            if start <= last_end:
                merged[-1][1] = max(last_end, end)
                merged[-1][3] = idx1
                merged[-1][4].append(jump)
            else:
                merged.append([start, end, idx0, idx1, [jump]])

    periods = pd.DataFrame([
        {
            "start_day": start,
            "end_day": end,
            "duration_days": end - start,
            "start_index": idx0,
            "end_index": idx1,
            "max_abs_jump_in_period": float(np.max(np.abs(jumps))),
            "num_triggered_steps_merged": len(jumps),
        }
        for start, end, idx0, idx1, jumps in merged
    ])

    return periods, valid_t, valid_dt, diff_ignore_nans, intervals


def plot_filtered_and_triggers(
    dt_clean, periods, valid_t, diff_ignore_nans,
    threshold=1.0, samples_per_day=24.0, outfile="trigger_plot.png",
):
    """Plot filtered Δt and retained-point differences with trigger intervals shaded."""
    sample_index = np.arange(len(dt_clean))
    t_days = sample_index / samples_per_day

    mask_filtered = np.isfinite(dt_clean)
    mask_diff = np.isfinite(diff_ignore_nans)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for j, row in periods.iterrows():
        x0, x1 = row["start_day"], row["end_day"]
        ax1.axvspan(x0, x1, color="lightcoral", alpha=0.25,
                    label="Triggered interval: [t(k-1), t(k)]" if j == 0 else None)
        ax2.axvspan(x0, x1, color="lightcoral", alpha=0.25)

    ax1.scatter(t_days[mask_filtered], dt_clean[mask_filtered], s=5, label="Filtered dt")
    ax1.set_ylabel("Filtered dt")
    ax1.set_title("Filtered HYS12-HYS14 dt, full time range")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, loc="upper right")

    ax2.scatter(valid_t[mask_diff], diff_ignore_nans[mask_diff], s=5,
                label="Δdt between retained points")
    ax2.axhline(threshold, linewidth=1, linestyle="--", alpha=0.8)
    ax2.axhline(-threshold, linewidth=1, linestyle="--", alpha=0.8)
    ax2.axhline(0, linewidth=1, alpha=0.6)

    ax2.set_xlabel("Elapsed time (days)")
    ax2.set_ylabel("Δdt between\nretained points")
    ax2.set_title("Difference between consecutive retained filtered points, ignoring NaNs")
    ax2.grid(True, alpha=0.3)

    if len(valid_t) > 0:
        ax2.set_xlim(np.nanmin(valid_t), np.nanmax(valid_t))

    plt.tight_layout()
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    dt_orig = np.load(INPUT_FILE).astype(float)

    dt_clean, final_mask, strict_mask, extra_mask = apply_final_filter(dt_orig)

    periods, valid_t, valid_dt, diff_ignore_nans, raw_intervals = compute_trigger_periods(
        dt_clean, threshold=TRIGGER_THRESHOLD, samples_per_day=SAMPLES_PER_DAY,
    )

    np.save(FILTERED_NPY, dt_clean)
    np.save(OUTLIER_MASK_NPY, final_mask)
    periods.to_csv(TRIGGER_CSV, index=False)

    plot_filtered_and_triggers(
        dt_clean, periods, valid_t, diff_ignore_nans,
        threshold=TRIGGER_THRESHOLD, samples_per_day=SAMPLES_PER_DAY,
        outfile=PLOT_FILE,
    )

    orig_finite = np.isfinite(dt_orig)
    clean_finite = np.isfinite(dt_clean)

    print("Final filter summary")
    print("--------------------")
    print(f"Input file: {INPUT_FILE}")
    print(f"Original samples: {len(dt_orig)}")
    print(f"Original finite samples: {orig_finite.sum()}")
    print(f"Strict-stage removed: {strict_mask.sum()}")
    print(f"Extra removed after strict: {extra_mask.sum()}")
    print(f"Total removed: {final_mask.sum()}")
    print(f"Remaining finite samples: {clean_finite.sum()}")
    print()

    print("Trigger summary")
    print("---------------")
    print(f"Threshold: |Δdt| > {TRIGGER_THRESHOLD}")
    print(f"Triggered pair-intervals before merging: {len(raw_intervals)}")
    print(f"Shaded periods after merging: {len(periods)}")
    print()

    if len(periods) > 0:
        print("Trigger periods:")
        print(periods.round({
            "start_day": 2, "end_day": 2, "duration_days": 2,
            "max_abs_jump_in_period": 3,
        }).to_string(index=False))
        print()

    print("Saved outputs")
    print("-------------")
    print(f"Filtered data: {FILTERED_NPY}")
    print(f"Outlier mask:  {OUTLIER_MASK_NPY}")
    print(f"Trigger CSV:   {TRIGGER_CSV}")
    print(f"Trigger plot:  {PLOT_FILE}")


if __name__ == "__main__":
    main()
