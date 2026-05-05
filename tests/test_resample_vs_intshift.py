"""Localize the corrected-CCF damage: is it the linear-interp time-warp
in chronfix._resample, or something else?

Experiment: re-run hour-13 of 2023-10-11 three ways, all using HYS12 raw
as the reference station and identical CCF processing:

  A. HYS14 raw                                (uncorrected baseline)
  B. HYS14 raw, integer-sample shift only     (no fractional interp,
                                              no zigzag — single Δt per day)
  C. HYS14 chronfix-corrected (already on disk, what produced the bad pick)

If B's peak amplitude matches A's (just shifted to ~0 lag) and C's is much
weaker, the damage is in chronfix's per-hour zigzag time-warp. That
falsifies "smoothing wouldn't help" and points the fix at the clock-model
stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime, read_inventory

import sys
sys.path.insert(0, "/home/seismic/chronos/src")
from chronos.scripts import compute_ccf as ccf

RAW = Path("/data/wsd02/maleen_data/OOI-Data")
COR = Path("/data/wsd02/maleen_data/OOI-Data-corrected")
INV_DIR = RAW / "StationXML"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

YEAR, DOY = 2023, 284  # 2023-10-11
DAY = UTCDateTime(2023, 10, 11)
HOUR = 13
FS = 8.0
DT_AT_DAY_START = 10.75  # seconds; from delta_t_hourly_clean[2023-10-11T00]
DT_INT_SAMPLES = int(round(DT_AT_DAY_START * FS))  # 86 samples


def load_full_day_processed(path: Path, inv) -> np.ndarray:
    """Replicate compute_ccf.load_day_z exactly."""
    st = read(str(path))
    for tr in st:
        if tr.data.dtype != np.float64:
            tr.data = tr.data.astype(np.float64, copy=False)
    st.merge(method=1, fill_value=0.0)
    tr = st.select(channel="MHZ")[0].copy()

    ccf.taper_gaps(tr, ccf.GAP_TAPER_S)
    tr.detrend("demean")
    tr.detrend("linear")
    tr.filter("highpass", freq=ccf.HP_FREQ, zerophase=True)

    nyq = tr.stats.sampling_rate / 2.0
    pre_filt = (ccf.PRE_FILT[0], ccf.PRE_FILT[1],
                min(ccf.PRE_FILT[2], 0.9 * nyq),
                min(ccf.PRE_FILT[3], 0.98 * nyq))
    tr.remove_response(inventory=inv, output="VEL", pre_filt=pre_filt)

    cur_fs = tr.stats.sampling_rate
    if abs(cur_fs - FS) > 1e-6:
        if cur_fs > FS:
            tr.resample(FS, no_filter=False)
        else:
            tr.interpolate(sampling_rate=FS, method="lanczos", a=20)

    tr.trim(starttime=DAY, endtime=DAY + 86400.0,
            pad=True, fill_value=0.0, nearest_sample=True)
    return np.asarray(tr.data, dtype=np.float64)


def hour_window(arr: np.ndarray, hour: int) -> np.ndarray:
    """Extract samples for a given UTC hour from a 86400.125 s array."""
    n_per_h = int(3600 * FS)
    start = hour * n_per_h
    return arr[start:start + n_per_h].copy()


def cc_one_hour(a_full: np.ndarray, b_full: np.ndarray, hour: int):
    """Run compute_ccf.process_day's logic but only over the chosen hour;
    return the per-hour median CCF (matches what cc_hourly stores)."""
    a_h = hour_window(a_full, hour)
    b_h = hour_window(b_full, hour)
    cc_list, _ = ccf.process_day(a_h, b_h, FS, day_offset_days=0)
    if not cc_list:
        return None
    cc = np.asarray(cc_list)
    return np.median(cc, axis=0)


def main() -> int:
    inv12 = read_inventory(str(INV_DIR / f"OO.HYS12..MHZ.xml"))
    inv14 = read_inventory(str(INV_DIR / f"OO.HYS14..MHZ.xml"))

    p12 = RAW / f"HYS12/{YEAR}/{DOY:03d}/HYS12.OO.{YEAR}.{DOY:03d}.MHZ"
    p14 = RAW / f"HYS14/{YEAR}/{DOY:03d}/HYS14.OO.{YEAR}.{DOY:03d}.MHZ"
    p14c = COR / f"HYS14/{YEAR}/{DOY:03d}/HYS14.OO.{YEAR}.{DOY:03d}.MHZ"

    print("loading...")
    a12 = load_full_day_processed(p12, inv12)
    a14_raw = load_full_day_processed(p14, inv14)
    a14_cor = load_full_day_processed(p14c, inv14)
    print(f"shapes: {a12.shape} {a14_raw.shape} {a14_cor.shape}")

    # Variant B: integer-sample shift of raw HYS14 by Δt(00:00)=+10.75s.
    # Convention from chronfix: corrected sample at UTC t = raw sample at apparent t + Δt.
    # In array space (full day), apparent index = utc_index + DT_INT_SAMPLES.
    # So "corrected[utc_idx] = raw[utc_idx + DT]" => roll left by DT.
    a14_intshift = np.roll(a14_raw, -DT_INT_SAMPLES)
    a14_intshift[-DT_INT_SAMPLES:] = 0.0  # zero the wrap-around region

    half = int(round(ccf.MAXLAG * FS))
    lags = np.arange(-half, half + 1) / FS

    cc_A = cc_one_hour(a12, a14_raw,      HOUR)  # uncorrected baseline
    cc_B = cc_one_hour(a12, a14_intshift, HOUR)  # integer-shift only
    cc_C = cc_one_hour(a12, a14_cor,      HOUR)  # chronfix resample

    def stats(label, cc):
        if cc is None:
            print(f"{label}: NO CCF"); return
        i = int(np.argmax(np.abs(cc)))
        print(f"{label}: max|cc|={np.abs(cc).max():.2f} at lag={lags[i]:+.3f}s")

    stats("A raw uncorrected         ", cc_A)
    stats("B integer-sample shift   ", cc_B)
    stats("C chronfix resample (file)", cc_C)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, sharey=True)
    for ax, label, cc in zip(axes,
                             ["A: HYS12 vs HYS14 raw (uncorrected)",
                              "B: HYS12 vs HYS14 integer-shift -86 samples",
                              "C: HYS12 vs HYS14 chronfix-resampled"],
                             [cc_A, cc_B, cc_C]):
        if cc is None:
            ax.set_title(f"{label}  [no CCF]")
            continue
        ax.plot(lags, cc, lw=0.7)
        ipk = int(np.argmax(np.abs(cc)))
        ax.axvline(lags[ipk], color="r", lw=0.7)
        ax.set_title(f"{label}    max|cc|={np.abs(cc).max():.1f}  pick={lags[ipk]:+.3f}s")
    axes[-1].set_xlabel("lag (s)")
    fig.suptitle(f"Hour {HOUR} UTC of 2023-10-11 — three correction variants")
    fig.tight_layout()
    out = OUT / "resample_vs_intshift.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
