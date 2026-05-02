#!/usr/bin/env python
"""Combine the two pair-shift estimates into a single HYS14 clock-error series.

For each day:
    dt_from_HYS12_HYS14 = -shifts_HYS12_HYS14    (HYS14 is the B-side)
    dt_from_HYS14_HYSB1 = +shifts_HYS14_HYSB1    (HYS14 is the A-side)

These two are independent estimates of the same quantity Δt_HYS14, the
HYS14 clock error (positive = HYS14 reports times that are "late"
relative to true UTC; the fix subtracts Δt from HYS14's timestamps).

Combine: mean where both pairs are valid, the single valid value where
only one pair has data, NaN where neither. The disagreement between the
two pair estimates (where both are valid) is the empirical error bar.

Inputs (from `realign_ccf.py`):
    data/ccf_realigned/HYS12-HYS14/shifts.npy + cc_dates.npy
    data/ccf_realigned/HYS14-HYSB1_lowband/shifts.npy + cc_dates.npy

Outputs (under data/clock_estimate/HYS14/):
    delta_t.npy              combined HYS14 clock error per day, seconds
    dates.npy                master date axis (datetime64[D])
    dt_from_HYS12_HYS14.npy  per-day estimate from pair 1 (sign-flipped)
    dt_from_HYS14_HYSB1.npy  per-day estimate from pair 2
    residual.npy             pair1 - pair2 where both valid, NaN otherwise
    plot.png                 4-panel diagnostic
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
    valid_a = ~np.isnan(arr_a)
    valid_b = ~np.isnan(arr_b)
    both = valid_a & valid_b
    only_a = valid_a & ~valid_b
    only_b = valid_b & ~valid_a

    out = np.full_like(arr_a, np.nan)
    out[both] = 0.5 * (arr_a[both] + arr_b[both])
    out[only_a] = arr_a[only_a]
    out[only_b] = arr_b[only_b]

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
        "pair A=%s: %d valid; pair B=%s: %d valid; both: %d; combined: %d",
        PAIR_A_TAG, int(valid_a.sum()), PAIR_B_TAG, int(valid_b.sum()),
        int(both.sum()), int((~np.isnan(combined)).sum()),
    )
    LOG.info("residual RMS (A-B where both valid): %.3f s", res_rms)

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
    ax.plot(days[va], a[va], ".", ms=2.0, color="C0", label=f"{PAIR_A_TAG} (sign {SIGN_A:+.0f})")
    ax.plot(days[vb], b[vb], ".", ms=2.0, color="C3", label=f"{PAIR_B_TAG} (sign {SIGN_B:+.0f})")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel(r"$\Delta t_{HYS14}$ (s)")
    ax.set_title("Two independent estimates of HYS14 clock error")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)

    ax = axes[1]
    vc = ~np.isnan(combined)
    ax.plot(days[vc], combined[vc], ".", ms=2.0, color="C2")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel(r"$\Delta t_{HYS14}$ (s)")
    ax.set_title("Combined estimate (mean where both valid, single where one valid)")

    ax = axes[2]
    vr = ~np.isnan(residual)
    ax.plot(days[vr], residual[vr], ".", ms=2.0, color="C4")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel("A - B (s)")
    ax.set_xlabel("Date")
    ax.set_title(f"Residual where both pairs valid  (RMS = {res_rms:.3f} s)")

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
