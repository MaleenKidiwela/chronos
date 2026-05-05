#!/usr/bin/env python3
"""Filter hourly Δt and detect threshold-triggered intervals.

A single multi-pass Hampel-style filter followed by interval trigger
detection on the differences between consecutive retained points. This
is the chronos→chronfix bridge: it consumes the hourly Δt produced by
combine_clock and produces the correction file chronfix consumes.

Workflow:
    1. Load primary Δt (sign-flipped to target's clock) from a per-pair
       file like dt_hourly_from_<primary-pair>.npy. Default: derived
       from --target and --primary-pair.
    2. Apply final outlier filter:
       - 3-pass Hampel filter (windows 168 / 72 / 25 hours)
       - 3-point continuity residual pass
       - 1 extra "bit less-strict" 240-hour Hampel pass
    3. Difference consecutive retained points (NaNs ignored):
           Δdt_k = retained[k] - retained[k-1]
    4. Trigger when |Δdt_k| > threshold (default 1.0 s).
    5. Trigger interval is [t(k-1), t(k)]; merge overlapping intervals.
    6. Save:
       - filtered .npy           (delta_t_hourly_clean.npy)
       - outlier mask .npy
       - trigger-period CSV       (trigger_periods.csv)
       - full-time trigger plot

Run from anywhere:

    python -m chronos.scripts.filter_and_triggers --target HYS14 \
        --primary-pair HYS12-HYS14
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_CHRONOS_ROOT = Path("/home/seismic/chronos")
SAMPLES_PER_DAY = 24.0  # one Δt sample per hour
DEFAULT_TRIGGER_THRESHOLD = 1.0  # |Δdt| (s) between consecutive retained points
# Below this segment slope, replace the segment with its robust median.
# Above it, replace with a robust linear fit. Picker quantum is 0.125 s,
# so anything below ~0.05 s/day is indistinguishable from flat in noise.
DEFAULT_SLOPE_THRESHOLD_PER_DAY = 0.05
DEFAULT_MIN_SEG_VALID = 5  # skip modeling if fewer valid samples than this
DEFAULT_SMOOTHER_WINDOW_HOURS = 24   # rolling robust median window
DEFAULT_SMOOTHER_MA_HOURS = 6        # short MA on top of rolling median


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


def model_segments(
    dt_clean,
    periods,
    slope_threshold_per_day=DEFAULT_SLOPE_THRESHOLD_PER_DAY,
    samples_per_day=SAMPLES_PER_DAY,
    min_valid=DEFAULT_MIN_SEG_VALID,
    smoother_window=DEFAULT_SMOOTHER_WINDOW_HOURS,
    smoother_ma=DEFAULT_SMOOTHER_MA_HOURS,
):
    """Replace each inter-trigger segment of Δt with a robust model.

    Segments are defined by the trigger intervals from
    `compute_trigger_periods`. Inside a trigger interval the output stays
    NaN (chronfix splits its corrected output at those boundaries). Within
    each stable inter-trigger segment:

      1. robust slope is estimated by Theil-Sen on the finite samples,
      2. if |slope| < `slope_threshold_per_day` -> segment replaced by its
         robust median (a flat segment is most honestly a constant; this
         also avoids end-effect wobble of the rolling smoother),
      3. otherwise -> segment replaced by a rolling robust median (window
         `smoother_window` hours, centered) followed by a short moving
         average (window `smoother_ma` hours, centered). This is shape-
         preserving: it removes sub-quantum hour-to-hour jitter without
         imposing any functional form on the drift, so curved drift
         segments are tracked instead of approximated by a straight line.
         Edge gaps from the rolling windows are filled with the segment's
         end-extrapolated values rather than NaN, so the modeled series
         is dense within the segment.

    The modeled series has the same picker scale as the raw cleaned
    series, except sub-quantum hourly jitter is removed -- which is what
    was destroying CCF coherence after chronfix-resample.

    Returns
    -------
    dt_model : np.ndarray (float64)
        Same shape as dt_clean. NaN inside trigger intervals or segments
        with too few finite samples; modeled values elsewhere.
    info : list of dicts
        Per-segment diagnostics: (lo, hi, n_valid, slope_per_day,
        mode in {"median","rolling","skipped"}, value).
    """
    try:
        from scipy.stats import theilslopes
        _have_theil = True
    except Exception:
        _have_theil = False

    n = len(dt_clean)
    out = np.full(n, np.nan, dtype=np.float64)

    # Build the list of inter-trigger segments [lo, hi).
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    if periods is not None and len(periods) > 0:
        # periods is expected to be sorted by start_index.
        sorted_periods = periods.sort_values("start_index")
        for _, row in sorted_periods.iterrows():
            s = int(row["start_index"])
            e = int(row["end_index"])
            if s > cursor:
                boundaries.append((cursor, s))
            cursor = e + 1
    if cursor < n:
        boundaries.append((cursor, n))

    info: list[dict] = []
    for lo, hi in boundaries:
        seg = dt_clean[lo:hi]
        finite_idx = np.where(np.isfinite(seg))[0]
        if len(finite_idx) < min_valid:
            info.append({
                "lo": lo, "hi": hi, "n_valid": int(len(finite_idx)),
                "slope_per_day": np.nan, "mode": "skipped", "value": np.nan,
            })
            continue

        x = finite_idx.astype(np.float64)
        y = seg[finite_idx].astype(np.float64)
        if _have_theil:
            slope, _intercept, _, _ = theilslopes(y, x)
        else:
            slope, _intercept = np.polyfit(x, y, 1)
        slope_per_day = slope * samples_per_day

        if abs(slope_per_day) < slope_threshold_per_day:
            value = float(np.nanmedian(seg))
            out[lo:hi] = value
            info.append({
                "lo": lo, "hi": hi, "n_valid": int(len(finite_idx)),
                "slope_per_day": float(slope_per_day),
                "mode": "median", "value": value,
            })
            continue

        # Rolling robust median + short MA. Operate on the full segment
        # (NaNs from picker filtering count as missing; pandas rolling
        # ignores them with min_periods).
        s_seg = pd.Series(seg)
        win_med = max(3, int(smoother_window))
        win_ma = max(1, int(smoother_ma))
        smoothed = (
            s_seg.rolling(window=win_med, center=True,
                          min_periods=max(3, win_med // 4)).median()
                 .rolling(window=win_ma, center=True,
                          min_periods=max(1, win_ma // 2)).mean()
        )

        # Fill any leading/trailing NaN by holding the nearest valid
        # smoothed value; this keeps the segment dense for chronfix and
        # avoids end-of-segment dropouts.
        smoothed = smoothed.ffill().bfill()
        out[lo:hi] = smoothed.to_numpy(dtype=np.float64)
        info.append({
            "lo": lo, "hi": hi, "n_valid": int(len(finite_idx)),
            "slope_per_day": float(slope_per_day),
            "mode": "rolling", "value": float(np.nanmedian(seg)),
        })
    return out, info


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
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True,
                   help="Station whose clock is being corrected (e.g. HYS14).")
    p.add_argument("--primary-pair", required=True,
                   help="Pair tag whose Δt was sign-flipped to target. "
                        "Reads dt_hourly_from_<primary-pair>.npy.")
    p.add_argument("--chronos-root", default=str(DEFAULT_CHRONOS_ROOT))
    p.add_argument("--input-file", default=None,
                   help="Override input Δt npy. Default: "
                        "<chronos-root>/data/clock_estimate/<target>/dt_hourly_from_<primary-pair>.npy")
    p.add_argument("--clock-dir", default=None,
                   help="Override output directory. Default: "
                        "<chronos-root>/data/clock_estimate/<target>/")
    p.add_argument("--threshold", type=float, default=DEFAULT_TRIGGER_THRESHOLD,
                   help="|Δdt| threshold between consecutive retained hourly samples (s).")
    p.add_argument("--no-segment-model", action="store_true",
                   help="Disable per-segment modeling. The cleaned hourly Δt "
                        "is written verbatim to delta_t_hourly_clean.npy "
                        "(legacy behavior; not recommended -- the picker "
                        "quantum injects sub-second jitter that chronfix "
                        "_resample turns into CCF-coherence damage).")
    p.add_argument("--slope-threshold-per-day", type=float,
                   default=DEFAULT_SLOPE_THRESHOLD_PER_DAY,
                   help="Per-segment slope (s/day) below which a segment is "
                        "modeled as a constant (robust median). Above it, "
                        "modeled by a rolling robust smoother (shape-"
                        "preserving).")
    p.add_argument("--smoother-window-hours", type=int,
                   default=DEFAULT_SMOOTHER_WINDOW_HOURS,
                   help="Rolling robust median window (hours) for the "
                        "drift-segment smoother. Picks up curvature on "
                        "timescales longer than this; flattens jitter on "
                        "shorter timescales.")
    p.add_argument("--smoother-ma-hours", type=int,
                   default=DEFAULT_SMOOTHER_MA_HOURS,
                   help="Short moving-average window (hours) applied on "
                        "top of the rolling median.")
    args = p.parse_args()

    clock_dir = Path(args.clock_dir) if args.clock_dir else (
        Path(args.chronos_root) / "data" / "clock_estimate" / args.target
    )
    input_file = Path(args.input_file) if args.input_file else (
        clock_dir / f"dt_hourly_from_{args.primary_pair}.npy"
    )

    filtered_npy = clock_dir / "delta_t_hourly_clean.npy"
    raw_filtered_npy = clock_dir / "delta_t_hourly_filtered_raw.npy"
    outlier_mask_npy = clock_dir / "delta_t_hourly_outlier_mask.npy"
    trigger_csv = clock_dir / "trigger_periods.csv"
    plot_file = clock_dir / "filter_and_triggers.png"

    clock_dir.mkdir(parents=True, exist_ok=True)
    dt_orig = np.load(input_file).astype(float)

    dt_clean, final_mask, strict_mask, extra_mask = apply_final_filter(dt_orig)

    periods, valid_t, valid_dt, diff_ignore_nans, raw_intervals = compute_trigger_periods(
        dt_clean, threshold=args.threshold, samples_per_day=SAMPLES_PER_DAY,
    )

    if args.no_segment_model:
        dt_for_chronfix = dt_clean
        seg_info = None
    else:
        dt_for_chronfix, seg_info = model_segments(
            dt_clean, periods,
            slope_threshold_per_day=args.slope_threshold_per_day,
            samples_per_day=SAMPLES_PER_DAY,
            smoother_window=args.smoother_window_hours,
            smoother_ma=args.smoother_ma_hours,
        )
        # Always keep the unmodeled cleaned series for QC.
        np.save(raw_filtered_npy, dt_clean)

    np.save(filtered_npy, dt_for_chronfix)
    np.save(outlier_mask_npy, final_mask)
    periods.to_csv(trigger_csv, index=False)

    plot_filtered_and_triggers(
        dt_for_chronfix, periods, valid_t, diff_ignore_nans,
        threshold=args.threshold, samples_per_day=SAMPLES_PER_DAY,
        outfile=plot_file,
    )

    orig_finite = np.isfinite(dt_orig)
    clean_finite = np.isfinite(dt_clean)

    print(f"Final filter summary  (target={args.target}, primary={args.primary_pair})")
    print("-" * 60)
    print(f"Input file: {input_file}")
    print(f"Original samples: {len(dt_orig)}")
    print(f"Original finite samples: {orig_finite.sum()}")
    print(f"Strict-stage removed: {strict_mask.sum()}")
    print(f"Extra removed after strict: {extra_mask.sum()}")
    print(f"Total removed: {final_mask.sum()}")
    print(f"Remaining finite samples: {clean_finite.sum()}")
    print()

    print("Trigger summary")
    print("-" * 15)
    print(f"Threshold: |Δdt| > {args.threshold}")
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

    if seg_info is not None:
        n_roll = sum(1 for s in seg_info if s["mode"] == "rolling")
        n_med = sum(1 for s in seg_info if s["mode"] == "median")
        n_skip = sum(1 for s in seg_info if s["mode"] == "skipped")
        print("Segment modeling")
        print("-" * 16)
        print(f"Slope threshold: {args.slope_threshold_per_day} s/day")
        print(f"Smoother window: {args.smoother_window_hours} h median, "
              f"{args.smoother_ma_hours} h MA")
        print(f"Segments: {len(seg_info)}  rolling={n_roll}  "
              f"median={n_med}  skipped={n_skip}")
        print()

    print("Saved outputs (this is the correction file chronfix consumes)")
    print("-" * 60)
    print(f"Filtered data: {filtered_npy}  "
          f"({'segment-modeled' if seg_info is not None else 'raw cleaned'})")
    if seg_info is not None:
        print(f"Raw filtered : {raw_filtered_npy}")
    print(f"Outlier mask:  {outlier_mask_npy}")
    print(f"Trigger CSV:   {trigger_csv}")
    print(f"Trigger plot:  {plot_file}")
    print()
    print("Pass --correction-dir", clock_dir, "to chronfix.scripts.apply_correction.")


if __name__ == "__main__":
    sys.exit(main())
