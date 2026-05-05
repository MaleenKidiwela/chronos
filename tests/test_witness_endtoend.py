"""End-to-end witness-day verification: compare hour-13 CCF using HYS14
corrected by the REAL chronfix.apply_correction pipeline before the fix
(file under /data/wsd02/maleen_data/OOI-Data-corrected) vs. after the
fix (file under /tmp/chronfix_witness, produced with the new smoothed
delta_t_hourly_clean.npy and the leading-NaN snap in chronfix.correct).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime, read_inventory

sys.path.insert(0, "/home/seismic/chronos/src")
from chronos.scripts import compute_ccf as ccf

RAW = Path("/data/wsd02/maleen_data/OOI-Data")
COR_OLD = Path("/data/wsd02/maleen_data/OOI-Data-corrected")
COR_NEW = Path("/tmp/chronfix_witness")
INV_DIR = RAW / "StationXML"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

YEAR, DOY = 2023, 284
DAY = UTCDateTime(2023, 10, 11)
HOUR = 13
FS = 8.0


def load_processed(path: Path, inv) -> np.ndarray:
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
    tr.trim(starttime=DAY, endtime=DAY + 86400.0,
            pad=True, fill_value=0.0, nearest_sample=True)
    return np.asarray(tr.data, dtype=np.float64)


def hour_window(arr, hour):
    n = int(3600 * FS)
    return arr[hour * n: (hour + 1) * n].copy()


def cc(a, b):
    cc_list, _ = ccf.process_day(a, b, FS, day_offset_days=0)
    return np.median(np.asarray(cc_list), axis=0) if cc_list else None


def main() -> int:
    inv12 = read_inventory(str(INV_DIR / "OO.HYS12..MHZ.xml"))
    inv14 = read_inventory(str(INV_DIR / "OO.HYS14..MHZ.xml"))
    p12 = RAW / f"HYS12/{YEAR}/{DOY:03d}/HYS12.OO.{YEAR}.{DOY:03d}.MHZ"
    p14 = RAW / f"HYS14/{YEAR}/{DOY:03d}/HYS14.OO.{YEAR}.{DOY:03d}.MHZ"
    p14_old = COR_OLD / f"HYS14/{YEAR}/{DOY:03d}/HYS14.OO.{YEAR}.{DOY:03d}.MHZ"
    p14_new = COR_NEW / f"HYS14/{YEAR}/{DOY:03d}/HYS14.OO.{YEAR}.{DOY:03d}.MHZ"

    a12     = load_processed(p12, inv12)
    a14_raw = load_processed(p14, inv14)
    a14_old = load_processed(p14_old, inv14)
    a14_new = load_processed(p14_new, inv14)

    half = int(round(ccf.MAXLAG * FS))
    lags = np.arange(-half, half + 1) / FS

    cc_raw = cc(hour_window(a12, HOUR), hour_window(a14_raw, HOUR))
    cc_old = cc(hour_window(a12, HOUR), hour_window(a14_old, HOUR))
    cc_new = cc(hour_window(a12, HOUR), hour_window(a14_new, HOUR))

    def stats(label, c):
        i = int(np.argmax(np.abs(c)))
        print(f"{label}: max|cc|={np.abs(c).max():.1f} at lag={lags[i]:+.3f}s")

    stats("raw (uncorrected)            ", cc_raw)
    stats("OLD (jittery Δt + buggy chronfix)", cc_old)
    stats("NEW (smoothed Δt + fixed chronfix)", cc_new)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, sharey=True)
    for ax, label, c in zip(axes,
                            ["raw uncorrected (HYS12 vs HYS14 raw)",
                             "OLD: jittery Δt + buggy chronfix",
                             "NEW: smoothed Δt + chronfix snap fix"],
                            [cc_raw, cc_old, cc_new]):
        if c is None:
            ax.set_title(f"{label}  [no CCF]"); continue
        ax.plot(lags, c, lw=0.7)
        i = int(np.argmax(np.abs(c)))
        ax.axvline(lags[i], color="r", lw=0.7)
        ax.set_title(f"{label}    max|cc|={np.abs(c).max():.1f}  pick={lags[i]:+.3f}s")
    axes[-1].set_xlabel("lag (s)")
    fig.suptitle(f"Hour {HOUR} UTC of 2023-10-11 — end-to-end old vs new")
    fig.tight_layout()
    out = OUT / "witness_endtoend.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
