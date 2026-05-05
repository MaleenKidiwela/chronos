"""Prototype: smooth Δt within each inter-trigger segment, simulate the
chronfix resample with the smoothed series, and re-test the witness hour.

If the smoothed-Δt resample recovers the CCF peak amplitude close to
variant B (integer-shift) from test_resample_vs_intshift.py, the
filter_and_triggers segment-smoothing fix is validated end-to-end without
having to rerun the full chronfix → CCF pipeline.

This script does NOT modify production code or files — it loads, simulates
in memory, and reports.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime, read_inventory

sys.path.insert(0, "/home/seismic/chronos/src")
from chronos.scripts import compute_ccf as ccf

CHRONOS_ROOT = Path("/home/seismic/chronos")
RAW = Path("/data/wsd02/maleen_data/OOI-Data")
INV_DIR = RAW / "StationXML"
CLOCK_DIR = CHRONOS_ROOT / "data/clock_estimate/HYS14"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

YEAR, DOY = 2023, 284
DAY = UTCDateTime(2023, 10, 11)
HOUR = 13
FS = 8.0


def load_clock_model():
    ht = np.load(CLOCK_DIR / "hour_times.npy")
    dt = np.load(CLOCK_DIR / "delta_t_hourly_clean.npy").astype(np.float64)
    trig = pd.read_csv(CLOCK_DIR / "trigger_periods.csv")
    return ht, dt, trig


def model_segments(ht: np.ndarray, dt: np.ndarray, trig: pd.DataFrame,
                   slope_thresh_per_day: float = 0.25) -> np.ndarray:
    """Per-segment smoothing of Δt(t).

    For each inter-trigger segment:
      - mask out the trigger interval itself (NaN)
      - estimate slope by robust linear fit; if |slope| < threshold per
        day -> replace with segment median; else replace with linear fit.
    """
    out = dt.copy()
    n = len(ht)
    # Boundaries: list of (segment_start_idx, segment_end_idx_exclusive),
    # plus mark trigger intervals NaN.
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for _, row in trig.iterrows():
        ts = int(row["start_index"])
        te = int(row["end_index"])
        if ts > cursor:
            boundaries.append((cursor, ts))
        # NaN inside trigger
        out[ts:te + 1] = np.nan
        cursor = te + 1
    if cursor < n:
        boundaries.append((cursor, n))

    fs_hours_per_day = 24.0
    for lo, hi in boundaries:
        seg = out[lo:hi]
        valid = np.isfinite(seg)
        if valid.sum() < 5:
            continue
        x = np.arange(hi - lo, dtype=np.float64)
        xv = x[valid]
        yv = seg[valid]
        # robust linear fit (Theil–Sen via numpy: use median of slopes)
        # Cheap version: ordinary least squares is fine after Hampel-cleaning;
        # the input is already cleaned, so use polyfit.
        slope, intercept = np.polyfit(xv, yv, 1)
        slope_per_day = slope * fs_hours_per_day  # s per day
        if abs(slope_per_day) < slope_thresh_per_day:
            out[lo:hi] = np.nanmedian(seg)
        else:
            out[lo:hi] = slope * x + intercept
    return out


def interp_dt_at_utc(t_utc_us: np.ndarray, ht: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Linear interpolation of Δt at given UTC times (datetime64[us])."""
    h_s = ht.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    q_s = t_utc_us.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    valid = np.isfinite(dt)
    return np.interp(q_s, h_s[valid], dt[valid], left=np.nan, right=np.nan)


def resample_day(raw_data: np.ndarray, raw_t0: UTCDateTime,
                 ht: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, UTCDateTime]:
    """Mimic chronfix._resample for one day-long input trace.

    raw_data: full day on the apparent-time grid at FS, starting at raw_t0.
    Returns (corrected samples on UTC grid, UTC start time).
    """
    n_in = len(raw_data)
    apparent_start = raw_t0
    apparent_end = raw_t0 + (n_in - 1) / FS

    valid = np.isfinite(dt)
    h_s = ht.astype("datetime64[s]").astype(np.int64).astype(np.float64)[valid]
    dv = dt[valid]
    dt_start = float(np.interp(apparent_start.timestamp, h_s, dv))
    dt_end = float(np.interp(apparent_end.timestamp, h_s, dv))
    utc_start = apparent_start - dt_start
    utc_end = apparent_end - dt_end
    n_out = int(np.floor((utc_end - utc_start) * FS)) + 1
    t_utc_offsets = np.arange(n_out, dtype=np.float64) / FS

    utc_start_us = np.datetime64(utc_start.datetime, "us")
    utc_query = utc_start_us + (t_utc_offsets * 1e6).astype("timedelta64[us]")
    dt_at_utc = interp_dt_at_utc(utc_query, ht, dt)
    if np.isnan(dt_at_utc).any():
        raise RuntimeError("Δt has NaNs in the UTC range")

    apparent_offsets = t_utc_offsets + dt_at_utc + (utc_start - apparent_start)
    apparent_grid = np.arange(n_in, dtype=np.float64) / FS
    # Clamp tiny float-precision overshoots at the boundaries so np.interp
    # doesn't return NaN for samples that are effectively at t=0 or t=end.
    eps = 0.5 / FS
    near_left  = (apparent_offsets < apparent_grid[0])  & (apparent_offsets > apparent_grid[0]  - eps)
    near_right = (apparent_offsets > apparent_grid[-1]) & (apparent_offsets < apparent_grid[-1] + eps)
    apparent_offsets[near_left]  = apparent_grid[0]
    apparent_offsets[near_right] = apparent_grid[-1]
    data_out = np.interp(apparent_offsets, apparent_grid,
                         raw_data.astype(np.float64),
                         left=np.nan, right=np.nan)

    valid = np.isfinite(data_out)
    if not valid.all():
        last = int(np.argmax(~valid))
        data_out = data_out[:last]
    return data_out, utc_start


def load_full_day_processed_from_array(arr: np.ndarray, t0: UTCDateTime, inv,
                                       station: str) -> np.ndarray:
    """Build an ObsPy Trace from a numpy array and run the same processing
    as compute_ccf.load_day_z."""
    from obspy import Trace
    tr = Trace(data=arr.astype(np.float64))
    tr.stats.network = "OO"
    tr.stats.station = station
    tr.stats.channel = "MHZ"
    tr.stats.sampling_rate = FS
    tr.stats.starttime = t0
    ccf.taper_gaps(tr, ccf.GAP_TAPER_S)
    tr.detrend("demean")
    tr.detrend("linear")
    tr.filter("highpass", freq=ccf.HP_FREQ, zerophase=True)
    nyq = tr.stats.sampling_rate / 2.0
    pre_filt = (ccf.PRE_FILT[0], ccf.PRE_FILT[1],
                min(ccf.PRE_FILT[2], 0.9 * nyq),
                min(ccf.PRE_FILT[3], 0.98 * nyq))
    tr.remove_response(inventory=inv, output="VEL", pre_filt=pre_filt)
    tr.trim(starttime=DAY, endtime=DAY + 86400.0,
            pad=True, fill_value=0.0, nearest_sample=True)
    return np.asarray(tr.data, dtype=np.float64)


def hour_window(arr: np.ndarray, hour: int) -> np.ndarray:
    n_per_h = int(3600 * FS)
    return arr[hour * n_per_h: (hour + 1) * n_per_h].copy()


def cc_one_hour(a: np.ndarray, b: np.ndarray):
    cc_list, _ = ccf.process_day(a, b, FS, day_offset_days=0)
    if not cc_list:
        return None
    return np.median(np.asarray(cc_list), axis=0)


def main() -> int:
    ht, dt_clean, trig = load_clock_model()

    # Show segment containing witness day.
    target_idx = int(np.searchsorted(ht.astype("datetime64[h]"),
                                     np.datetime64("2023-10-11T00", "h")))
    print(f"witness Δt sample @ {ht[target_idx]} = {dt_clean[target_idx]:.4f}")

    # Smooth Δt across all segments. (Threshold is 0.25 s/day — anything
    # slower than that is treated as a flat segment.)
    dt_smooth = model_segments(ht, dt_clean, trig, slope_thresh_per_day=0.05)
    print(f"dt around witness day (cleaned vs smoothed):")
    for j in range(target_idx - 2, target_idx + 26):
        print(f"  {ht[j]}  clean={dt_clean[j]:+.4f}  smooth={dt_smooth[j]:+.4f}")

    # Save smoothed for reuse.
    np.save(OUT / "dt_smooth.npy", dt_smooth)

    # Build raw HYS14 + HYS12 day arrays via processed pipeline.
    inv12 = read_inventory(str(INV_DIR / "OO.HYS12..MHZ.xml"))
    inv14 = read_inventory(str(INV_DIR / "OO.HYS14..MHZ.xml"))
    p12 = RAW / f"HYS12/{YEAR}/{DOY:03d}/HYS12.OO.{YEAR}.{DOY:03d}.MHZ"
    p14 = RAW / f"HYS14/{YEAR}/{DOY:03d}/HYS14.OO.{YEAR}.{DOY:03d}.MHZ"

    st12 = read(str(p12))
    st12.merge(method=1, fill_value=0.0)
    a12_full = load_full_day_processed_from_array(
        st12[0].data, st12[0].stats.starttime, inv12, "HYS12")

    st14 = read(str(p14))
    st14.merge(method=1, fill_value=0.0)
    raw14 = np.asarray(st14[0].data, dtype=np.float64)
    raw14_t0 = st14[0].stats.starttime

    # Variant D: chronfix-style resample using SMOOTHED Δt.
    out_smooth, utc_start_smooth = resample_day(raw14, raw14_t0, ht, dt_smooth)
    print(f"smoothed-resample utc_start={utc_start_smooth}  npts={len(out_smooth)}")
    a14_smooth = load_full_day_processed_from_array(
        out_smooth, utc_start_smooth, inv14, "HYS14")

    # Variant C': chronfix-style resample using JITTERY (clean) Δt — should
    # reproduce the production result up to numerical noise.
    out_jit, utc_start_jit = resample_day(raw14, raw14_t0, ht, dt_clean)
    a14_jit = load_full_day_processed_from_array(
        out_jit, utc_start_jit, inv14, "HYS14")

    # CCFs.
    cc_jit    = cc_one_hour(hour_window(a12_full, HOUR), hour_window(a14_jit, HOUR))
    cc_smooth = cc_one_hour(hour_window(a12_full, HOUR), hour_window(a14_smooth, HOUR))

    half = int(round(ccf.MAXLAG * FS))
    lags = np.arange(-half, half + 1) / FS

    def stats(label, cc):
        i = int(np.argmax(np.abs(cc)))
        print(f"{label}: max|cc|={np.abs(cc).max():.2f} at lag={lags[i]:+.3f}s")

    print()
    stats("C' resample with JITTERY clean Δt (reproduces production)", cc_jit)
    stats("D  resample with SMOOTHED Δt                              ", cc_smooth)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
    for ax, label, cc in zip(axes,
                             ["C' resample with jittery cleaned Δt",
                              "D  resample with segment-smoothed Δt"],
                             [cc_jit, cc_smooth]):
        ax.plot(lags, cc, lw=0.7)
        ipk = int(np.argmax(np.abs(cc)))
        ax.axvline(lags[ipk], color="r", lw=0.7)
        ax.set_title(f"{label}    max|cc|={np.abs(cc).max():.1f}  pick={lags[ipk]:+.3f}s")
    axes[-1].set_xlabel("lag (s)")
    fig.suptitle(f"Hour {HOUR} UTC of 2023-10-11 — jittery vs smoothed Δt")
    fig.tight_layout()
    out = OUT / "smoothed_vs_jittery_dt.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
