#!/usr/bin/env python
"""Daily inter-station ZZ cross-correlations for OO.HYS{12,14,B1}.MHZ.

Adapted from Earthnote's `phase13_pilot.py` (single-station SC) to inter-station
ZZ pairs at 8 Hz native sample rate, science band 1-3 Hz. Hand-rolled CCF: bare
ObsPy + numpy + scipy, no NoisePy dependency in the package itself.

Per-pair outputs under data/ccf/<pair_tag>/:
    cc_30min.npy          shape (N_segs, n_lags)
    cc_30min_times.npy    fractional-day-of-year for each segment
    cc_daily.npy          shape (N_days, n_lags)  per-lag median across day
    cc_dates.npy          datetime64[D] aligned with cc_daily
    cc_ref.npy            long-term linear stack reference
    lags.npy              lag axis in seconds, shape (n_lags,)

Usage:
    python hys_ccf.py                            # 2022-01-01 .. today, all 3 pairs
    python hys_ccf.py --start 2022-01-01 --end 2022-12-31
    python hys_ccf.py --pairs HYS12-HYS14 HYS14-HYSB1
    python hys_ccf.py --workers 8
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from obspy import read, read_inventory

warnings.filterwarnings("ignore")

# ---- Paths ----
DATA_ROOT = Path("/data/wsd02/maleen_data/OOI-Data")
CHRONOS_ROOT = Path("/home/seismic/chronos")
OUT_ROOT = CHRONOS_ROOT / "data" / "ccf"

NETWORK = "OO"
CHANNEL = "MHZ"
DEFAULT_PAIRS = ("HYS12-HYS14", "HYS14-HYSB1", "HYS12-HYSB1")

# ---- Acquisition / preprocessing (8 Hz native) ----
TARGET_FS = 8.0
HP_FREQ = 0.4
GAP_TAPER_S = 100.0
# pre_filt is clamped to Nyquist (~4 Hz) per-trace at runtime.
PRE_FILT = (0.05, 0.1, 3.6, 3.95)

# ---- CC / whitening ----
FMIN, FMAX = 1.0, 3.0          # science band
WHITEN_FMIN, WHITEN_FMAX = 0.5, 3.8
SEG_TAPER_S = 20.0
CC_LEN = 1800                  # 30-min windows
CC_STEP = 450                  # 7.5-min step => 75% overlap
MAXLAG = 60.0                  # seconds

LOG = logging.getLogger("hys_ccf")


# ============================== Preprocessing ==============================

def taper_gaps(tr, taper_s: float = GAP_TAPER_S) -> None:
    """Cosine-taper edges of zero-filled gaps (mirrors phase13_pilot)."""
    data = tr.data.astype(np.float64, copy=True)
    fs = tr.stats.sampling_rate
    n = len(data)
    if n == 0:
        return
    taper_n = int(round(taper_s * fs))
    is_zero = data == 0.0
    if not is_zero.any():
        tr.data = data
        return
    dz = np.diff(is_zero.astype(np.int8))
    zero_starts = np.where(dz == 1)[0] + 1
    zero_ends = np.where(dz == -1)[0]
    for zs in zero_starts:
        lo, hi = max(0, zs - taper_n), zs
        k = hi - lo
        if k > 0:
            i = np.arange(k)
            data[lo:hi] *= 0.5 * (1.0 - np.cos(np.pi * i / k))
    for ze in zero_ends:
        lo, hi = ze + 1, min(n, ze + 1 + taper_n)
        k = hi - lo
        if k > 0:
            i = np.arange(k)
            data[lo:hi] *= 0.5 * (1.0 + np.cos(np.pi * i / k))
    tr.data = data


def daily_path(sta: str, d: date) -> Path:
    doy = d.timetuple().tm_yday
    return DATA_ROOT / sta / f"{d.year}" / f"{doy:03d}" / f"{sta}.{NETWORK}.{d.year}.{doy:03d}.{CHANNEL}"


def load_day_z(sta: str, d: date, inv) -> np.ndarray | None:
    path = daily_path(sta, d)
    if not path.exists():
        return None
    try:
        st = read(str(path))
    except Exception as ex:
        LOG.debug("read failed %s: %s", path, ex)
        return None
    for tr in st:
        if tr.data.dtype != np.float64:
            tr.data = tr.data.astype(np.float64, copy=False)
    try:
        st.merge(method=1, fill_value=0.0)
    except Exception:
        return None
    sel = st.select(channel=CHANNEL)
    if len(sel) == 0:
        return None
    tr = sel[0].copy()

    taper_gaps(tr, GAP_TAPER_S)
    tr.detrend("demean")
    tr.detrend("linear")
    tr.filter("highpass", freq=HP_FREQ, zerophase=True)

    nyq = tr.stats.sampling_rate / 2.0
    pre_filt = (
        PRE_FILT[0], PRE_FILT[1],
        min(PRE_FILT[2], 0.9 * nyq), min(PRE_FILT[3], 0.98 * nyq),
    )
    try:
        tr.remove_response(inventory=inv, output="VEL", pre_filt=pre_filt)
    except Exception as ex:
        LOG.debug("remove_response failed %s %s: %s", sta, d, ex)
        return None

    cur_fs = tr.stats.sampling_rate
    if abs(cur_fs - TARGET_FS) > 1e-6:
        if cur_fs > TARGET_FS:
            tr.resample(TARGET_FS, no_filter=False)
        else:
            tr.interpolate(sampling_rate=TARGET_FS, method="lanczos", a=20)

    return np.asarray(tr.data, dtype=np.float64)


# =========================== CC core ===========================

def cosine_taper_edges(x, fs, taper_s=SEG_TAPER_S):
    n = len(x)
    taper_n = int(round(taper_s * fs))
    if taper_n <= 0 or 2 * taper_n >= n:
        return x
    taper = np.ones(n)
    idx = np.arange(taper_n)
    fade_in = 0.5 * (1.0 - np.cos(np.pi * idx / taper_n))
    taper[:taper_n] = fade_in
    taper[-taper_n:] = fade_in[::-1]
    return x * taper


def whiten_segment(x, fs, fmin, fmax):
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.abs(X)
    Xn = X / np.where(amp > 0, amp, 1.0)
    lo1, lo2 = fmin * 0.5, fmin
    hi1 = fmax
    hi2 = min(fmax * 1.2, fs / 2.0 - 1e-6)
    w = np.ones_like(freqs)
    w[freqs < lo1] = 0.0
    w[freqs > hi2] = 0.0
    m = (freqs >= lo1) & (freqs < lo2)
    if (lo2 - lo1) > 0:
        w[m] = 0.5 * (1.0 - np.cos(np.pi * (freqs[m] - lo1) / (lo2 - lo1)))
    m = (freqs > hi1) & (freqs <= hi2)
    if (hi2 - hi1) > 0:
        w[m] = 0.5 * (1.0 + np.cos(np.pi * (freqs[m] - hi1) / (hi2 - hi1)))
    return np.fft.irfft(Xn * w, n)


def cc_segment(a, b, fs, maxlag):
    """Cross-correlation of a and b. Convention: positive lag = b lags a."""
    n = len(a)
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    A = np.fft.rfft(a, n=nfft)
    B = np.fft.rfft(b, n=nfft)
    c = np.fft.irfft(np.conj(A) * B, n=nfft)
    half = int(round(maxlag * fs))
    pos = c[: half + 1]
    neg = c[-half:]
    return np.concatenate([neg, pos])  # length 2*half + 1


def process_day(a, b, fs, day_offset_days):
    cc_list, t_list = [], []
    seg_n = int(CC_LEN * fs)
    step_n = int(CC_STEP * fs)
    n = min(len(a), len(b))
    i = 0
    while i + seg_n <= n:
        za = a[i: i + seg_n].copy()
        zb = b[i: i + seg_n].copy()
        if np.std(za) == 0 or np.std(zb) == 0:
            i += step_n
            continue
        za -= za.mean()
        zb -= zb.mean()
        idx = np.arange(seg_n)
        for arr in (za, zb):
            p = np.polyfit(idx, arr, 1)
            arr -= p[0] * idx + p[1]
        za = cosine_taper_edges(za, fs, SEG_TAPER_S)
        zb = cosine_taper_edges(zb, fs, SEG_TAPER_S)
        za = whiten_segment(za, fs, WHITEN_FMIN, WHITEN_FMAX)
        zb = whiten_segment(zb, fs, WHITEN_FMIN, WHITEN_FMAX)
        za = np.sign(za)
        zb = np.sign(zb)
        cc_list.append(cc_segment(za, zb, fs, MAXLAG))
        seg_mid_s = (i + seg_n / 2.0) / fs
        t_list.append(day_offset_days + seg_mid_s / 86400.0)
        i += step_n
    return cc_list, t_list


# =========================== Pair driver ===========================

@dataclass
class Pair:
    sta_a: str
    sta_b: str

    @property
    def tag(self) -> str:
        return f"{self.sta_a}-{self.sta_b}"


def _load_inv(sta: str):
    p = DATA_ROOT / "StationXML" / f"{NETWORK}.{sta}..{CHANNEL}.xml"
    if not p.exists():
        raise FileNotFoundError(p)
    return read_inventory(str(p))


def _process_one_day(args):
    pair, d, day_idx = args
    inv_a = _load_inv(pair.sta_a)
    inv_b = _load_inv(pair.sta_b)
    a = load_day_z(pair.sta_a, d, inv_a)
    if a is None:
        return d, None, None
    b = load_day_z(pair.sta_b, d, inv_b)
    if b is None:
        return d, None, None
    n = min(len(a), len(b))
    cc_list, t_list = process_day(a[:n], b[:n], TARGET_FS, day_idx)
    if not cc_list:
        return d, None, None
    return d, np.asarray(cc_list, dtype=np.float32), np.asarray(t_list, dtype=np.float64)


def run_pair(pair: Pair, start: date, end: date, workers: int) -> None:
    out_dir = OUT_ROOT / pair.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    LOG.info("[%s] %d days: %s .. %s", pair.tag, len(days), start, end)

    # Lag axis is fixed by TARGET_FS / MAXLAG.
    half = int(round(MAXLAG * TARGET_FS))
    n_lags = 2 * half + 1
    lags = np.arange(-half, half + 1) / TARGET_FS

    cc_segments: list[np.ndarray] = []
    seg_times: list[np.ndarray] = []
    daily_stack = np.full((len(days), n_lags), np.nan, dtype=np.float32)

    args_list = [(pair, d, i) for i, d in enumerate(days)]
    if workers <= 1:
        results = (_process_one_day(a) for a in args_list)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        futs = [ex.submit(_process_one_day, a) for a in args_list]
        results = (f.result() for f in as_completed(futs))

    ok = 0
    for d_done, cc_arr, t_arr in results:
        if cc_arr is None:
            LOG.debug("[%s] gap %s", pair.tag, d_done)
            continue
        ok += 1
        cc_segments.append(cc_arr)
        seg_times.append(t_arr)
        # per-lag median across the day for the daily stack
        idx = (d_done - start).days
        daily_stack[idx] = np.median(cc_arr, axis=0).astype(np.float32)

    if ok == 0:
        LOG.warning("[%s] no successful days", pair.tag)
        return

    cc_30min = np.concatenate(cc_segments, axis=0).astype(np.float32)
    cc_30min_times = np.concatenate(seg_times, axis=0).astype(np.float64)
    cc_dates = np.array([str(d) for d in days], dtype="datetime64[D]")

    # Reference = linear stack of all valid daily stacks.
    valid = ~np.isnan(daily_stack).any(axis=1)
    cc_ref = np.nanmean(daily_stack[valid], axis=0).astype(np.float32)

    np.save(out_dir / "cc_30min.npy", cc_30min)
    np.save(out_dir / "cc_30min_times.npy", cc_30min_times)
    np.save(out_dir / "cc_daily.npy", daily_stack)
    np.save(out_dir / "cc_dates.npy", cc_dates)
    np.save(out_dir / "cc_ref.npy", cc_ref)
    np.save(out_dir / "lags.npy", lags.astype(np.float64))
    LOG.info(
        "[%s] wrote %d daily, %d 30-min segments to %s",
        pair.tag, int(valid.sum()), cc_30min.shape[0], out_dir,
    )


# =========================== CLI ===========================

def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def parse_pair(s: str) -> Pair:
    a, b = s.split("-", 1)
    return Pair(a.strip(), b.strip())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    p.add_argument("--start", type=parse_date, default=date(2022, 1, 1))
    p.add_argument("--end", type=parse_date, default=date.today())
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for pair_str in args.pairs:
        run_pair(parse_pair(pair_str), args.start, args.end, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
