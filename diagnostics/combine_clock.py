#!/usr/bin/env python
"""Build the canonical daily HYS14 clock-error estimate.

HYS12-HYS14 (highband) is the **primary** source of Δt_HYS14: stations are
near-co-located, the daily reference is high-SNR (RMS ~12.8), and coverage
is good (1331/1582 days). HYS14-HYSB1_lowband is used **only as a
cross-check** — it has lower SNR and a long deployment-gap stretch — so it
should not contaminate the primary estimate via averaging.

Sign convention (HYS14 is shared between the two pairs):
    dt_from_HYS12_HYS14 = -shifts_HYS12_HYS14    (HYS14 is the B-side)
    dt_from_HYS14_HYSB1 = +shifts_HYS14_HYSB1    (HYS14 is the A-side)

Δt_HYS14 > 0 means HYS14 reports times that are "late" relative to true
UTC; the correction subtracts Δt from HYS14's timestamps.

Inputs (from `realign_ccf.py`):
    data/ccf_realigned/HYS12-HYS14/shifts.npy + cc_dates.npy
    data/ccf_realigned/HYS14-HYSB1_lowband/shifts.npy + cc_dates.npy

Outputs (under data/clock_estimate/HYS14/):
    delta_t.npy              primary daily Δt_HYS14, seconds, NaN where unknown
    dates.npy                master date axis (datetime64[D])
    dt_from_HYS12_HYS14.npy  per-day estimate from primary pair (sign-flipped)
    dt_from_HYS14_HYSB1.npy  per-day estimate from cross-check pair (validation only)
    residual.npy             primary - cross-check where both valid; NaN otherwise
    plot.png                 3-panel diagnostic
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

CHRONOS_ROOT = Path("/home/seismic/chronos")
REALIGN_ROOT = CHRONOS_ROOT / "data" / "ccf_realigned"
OUT_DIR = CHRONOS_ROOT / "data" / "clock_estimate" / "HYS14"

PAIR_A_TAG = "HYS12-HYS14"
PAIR_B_TAG = "HYS14-HYSB1_lowband"
SIGN_A = -1.0  # HYS14 is B-side: dt_HYS14 = -shifts
SIGN_B = +1.0  # HYS14 is A-side: dt_HYS14 = +shifts

LOG = logging.getLogger("combine_clock")


def load_pair(tag: str, sign: float) -> tuple[np.ndarray, np.ndarray]:
    d = REALIGN_ROOT / tag
    shifts = np.load(d / "shifts.npy")
    dates = np.load(d / "cc_dates.npy")
    return sign * shifts, dates


def align_on_master(
    estimates: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Align all pair estimates on the union of date axes."""
    all_dates = sorted(set().union(*(set(d.tolist()) for _, d in estimates)))
    master = np.array(all_dates, dtype="datetime64[D]")
    aligned = []
    for arr, dates in estimates:
        out = np.full(len(master), np.nan, dtype=np.float64)
        idx = np.searchsorted(master, dates)
        out[idx] = arr
        aligned.append(out)
    return master, aligned


def combine(arr_a: np.ndarray, arr_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Primary = pair A only. Pair B is a cross-check, never blended in.

    Returns (delta_t, residual) where delta_t is just arr_a (NaNs preserved)
    and residual is arr_a - arr_b on days both have valid measurements.
    """
    out = arr_a.copy()
    valid_a, valid_b = ~np.isnan(arr_a), ~np.isnan(arr_b)
    both = valid_a & valid_b
    residual = np.full_like(arr_a, np.nan)
    residual[both] = arr_a[both] - arr_b[both]
    return out, residual


def run() -> None:
    a, dates_a = load_pair(PAIR_A_TAG, SIGN_A)
    b, dates_b = load_pair(PAIR_B_TAG, SIGN_B)
    master, (a_m, b_m) = align_on_master([(a, dates_a), (b, dates_b)])

    combined, residual = combine(a_m, b_m)

    valid_a, valid_b = ~np.isnan(a_m), ~np.isnan(b_m)
    both = valid_a & valid_b
    res_rms = float(np.sqrt(np.nanmean(residual[both] ** 2))) if both.any() else float("nan")

    LOG.info(
        "primary=%s: %d valid; cross-check=%s: %d valid; both: %d; output: %d",
        PAIR_A_TAG, int(valid_a.sum()), PAIR_B_TAG, int(valid_b.sum()),
        int(both.sum()), int((~np.isnan(combined)).sum()),
    )
    LOG.info("residual RMS (primary - cross-check where both valid): %.3f s", res_rms)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "delta_t.npy", combined)
    np.save(OUT_DIR / "dates.npy", master)
    np.save(OUT_DIR / "dt_from_HYS12_HYS14.npy", a_m)
    np.save(OUT_DIR / "dt_from_HYS14_HYSB1.npy", b_m)
    np.save(OUT_DIR / "residual.npy", residual)
    LOG.info("wrote outputs to %s", OUT_DIR)

    _plot(master, a_m, b_m, combined, residual, res_rms)


def _plot(dates, a, b, combined, residual, res_rms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    days = np.arange(len(dates))

    ax = axes[0]
    va, vb = ~np.isnan(a), ~np.isnan(b)
    ax.plot(days[va], a[va], ".", ms=2.0, color="C0", label=f"{PAIR_A_TAG} primary")
    ax.plot(days[vb], b[vb], ".", ms=2.0, color="C3", label=f"{PAIR_B_TAG} cross-check")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel(r"$\Delta t_{HYS14}$ (s)")
    ax.set_title("Primary estimate and cross-check (both sign-flipped to HYS14 convention)")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)

    ax = axes[1]
    vc = ~np.isnan(combined)
    ax.plot(days[vc], combined[vc], ".", ms=2.0, color="C2")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel(r"$\Delta t_{HYS14}$ (s)")
    ax.set_title(f"Canonical Δt: primary ({PAIR_A_TAG}) only")

    ax = axes[2]
    vr = ~np.isnan(residual)
    ax.plot(days[vr], residual[vr], ".", ms=2.0, color="C4")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel("primary - cross-check (s)")
    ax.set_xlabel("Date")
    ax.set_title(f"Validation residual where both pairs valid  (RMS = {res_rms:.3f} s)")

    n = len(dates)
    tick_idx = np.linspace(0, n - 1, min(10, n)).astype(int)
    for ax in axes:
        ax.set_xticks(tick_idx)
    axes[-1].set_xticklabels([str(dates[i]) for i in tick_idx], rotation=30, ha="right")

    fig.tight_layout()
    out = OUT_DIR / "plot.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    LOG.info("plot -> %s", out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
