#!/usr/bin/env python
"""Build the canonical hourly Δt for a target station from one or two pairs.

Pipeline:
    1. Load each pair's hourly peak-lag track + master hour times.
    2. Anchor each track to the late-window median (the post-fix stable lag).
    3. Convert peak-lag → shift = anchor - peak; sign-flip per pair so that
       Δt_TARGET > 0 means TARGET timestamps are "late" relative to true UTC.
    4. Align both pair estimates on a master hourly time axis.
    5. Pick primary = pair A only. Compute residual against pair B (cross-check).

Sign convention. The target station appears in each pair's tag as either the
A-side (first in PAIR-A=`A-B`) or the B-side. Sign is +1 if target is the
A-side, -1 if the B-side, so that the resulting Δt is in the canonical
"clock-late-relative-to-UTC" sense. This script auto-derives the signs from
the target station name and the pair tags.

Inputs (per pair):
    data/peak_lag_hourly/<pair>/peak_lag_hourly_global.npy
    data/peak_lag_hourly/<pair>/hour_times.npy
    data/ccf/<pair>/lags.npy   (for clipping the anchor window)

Outputs (under data/clock_estimate/<target>/):
    delta_t_hourly.npy                       primary hourly Δt_target (s)
    hour_times.npy                           master hourly axis (datetime64[h])
    dt_hourly_from_<primary-pair-tag>.npy    sign-flipped primary
    dt_hourly_from_<cross-pair-tag>.npy      sign-flipped cross-check (if any)
    residual_hourly.npy                      primary - cross-check, NaN otherwise
    plot_hourly.png                          diagnostic (3 panels with cross,
                                              1 panel otherwise)

Trigger / segment-break detection happens in the next pipeline step
(`chronos.scripts.filter_and_triggers`), not here.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

DEFAULT_CHRONOS_ROOT = Path("/home/seismic/chronos")

LOG = logging.getLogger("combine_clock")


def sign_for_target(pair_tag: str, target: str) -> float:
    """+1 if target is the A-side of `A-B[_tag]`, -1 if B-side."""
    a, b = pair_tag.split("_", 1)[0].split("-", 1)
    if a == target:
        return +1.0
    if b == target:
        return -1.0
    raise ValueError(f"target {target!r} not in pair {pair_tag!r}")


def load_pair(root: Path, tag: str) -> tuple[np.ndarray, np.ndarray, float]:
    d = root / "peak_lag_hourly" / tag
    peak = np.load(d / "peak_lag_hourly_global.npy")
    times = np.load(d / "hour_times.npy")
    lags = np.load(root / "ccf" / tag / "lags.npy")
    extent = float(lags[-1])
    return peak, times, extent


def anchor_lag(peak: np.ndarray, lag_extent: float, ref_window: int) -> float:
    """Median peak lag over the most recent valid hours, excluding picks
    clipped at the maxlag boundary."""
    valid = ~np.isnan(peak) & (np.abs(peak) < 0.99 * lag_extent)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        raise RuntimeError("no valid hourly peak picks for anchoring")
    tail = idx[-min(ref_window, len(idx)):]
    return float(np.median(peak[tail]))


def shifts_from_peak(
    peak: np.ndarray, lag_extent: float, ref_window: int, max_shift: float,
) -> tuple[np.ndarray, float]:
    anchor = anchor_lag(peak, lag_extent, ref_window)
    shifts = anchor - peak
    bad = np.isnan(peak) | (np.abs(peak) > 0.99 * lag_extent) | (np.abs(shifts) > max_shift)
    shifts[bad] = np.nan
    return shifts, anchor


def align_on_master(
    estimates: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[np.ndarray]]:
    all_times = sorted(set().union(*(set(t.tolist()) for _, t in estimates)))
    master = np.array(all_times, dtype="datetime64[h]")
    aligned = []
    for arr, times in estimates:
        out = np.full(len(master), np.nan, dtype=np.float64)
        idx = np.searchsorted(master, times)
        out[idx] = arr
        aligned.append(out)
    return master, aligned


def run(
    target: str, primary_pair: str, cross_pair: str | None,
    chronos_root: Path,
    ref_window: int, max_shift: float,
) -> None:
    data_root = chronos_root / "data"
    sign_a = sign_for_target(primary_pair, target)
    LOG.info("target=%s  primary=%s (sign %+0.0f)", target, primary_pair, sign_a)
    peak_a, times_a, ext_a = load_pair(data_root, primary_pair)
    shifts_a, anc_a = shifts_from_peak(peak_a, ext_a, ref_window, max_shift)
    LOG.info("[%s] anchor = %+0.3f s", primary_pair, anc_a)
    dt_a = sign_a * shifts_a

    estimates: list[tuple[np.ndarray, np.ndarray]] = [(dt_a, times_a)]

    sign_b: float | None = None
    if cross_pair:
        sign_b = sign_for_target(cross_pair, target)
        LOG.info("cross-check=%s (sign %+0.0f)", cross_pair, sign_b)
        peak_b, times_b, ext_b = load_pair(data_root, cross_pair)
        shifts_b, anc_b = shifts_from_peak(peak_b, ext_b, ref_window, max_shift)
        LOG.info("[%s] anchor = %+0.3f s", cross_pair, anc_b)
        dt_b = sign_b * shifts_b
        estimates.append((dt_b, times_b))

    master, aligned = align_on_master(estimates)
    a_m = aligned[0]
    delta_t = a_m.copy()

    if cross_pair:
        b_m = aligned[1]
        valid_a, valid_b = ~np.isnan(a_m), ~np.isnan(b_m)
        both = valid_a & valid_b
        residual = np.full_like(a_m, np.nan)
        residual[both] = a_m[both] - b_m[both]
        res_rms = (float(np.sqrt(np.nanmean(residual[both] ** 2)))
                   if both.any() else float("nan"))
        LOG.info(
            "primary=%s: %d valid; cross=%s: %d valid; both: %d",
            primary_pair, int(valid_a.sum()), cross_pair, int(valid_b.sum()),
            int(both.sum()),
        )
        LOG.info("residual RMS (primary - cross): %.3f s", res_rms)
    else:
        b_m = None
        residual = None
        res_rms = float("nan")

    out_dir = data_root / "clock_estimate" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "delta_t_hourly.npy", delta_t)
    np.save(out_dir / "hour_times.npy", master)
    np.save(out_dir / f"dt_hourly_from_{primary_pair}.npy", a_m)
    if cross_pair and b_m is not None:
        np.save(out_dir / f"dt_hourly_from_{cross_pair}.npy", b_m)
        np.save(out_dir / "residual_hourly.npy", residual)
    LOG.info("wrote outputs to %s", out_dir)

    _plot(out_dir, target, primary_pair, cross_pair,
          master, a_m, b_m, delta_t, residual, res_rms)


def _plot(out_dir, target, primary_pair, cross_pair,
          times, a, b, combined, residual, res_rms):
    """Diagnostic plot for the combine stage.

    Trigger / segment-break detection happens in the next step
    (filter_and_triggers); we don't draw triggers here, only the raw
    primary, the canonical Δt, and the cross-check residual (if any).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 3 if cross_pair else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    x = np.arange(len(times))
    n = len(times)
    tick_idx = np.linspace(0, n - 1, min(10, n)).astype(int)
    tick_labels = [str(times[i].astype("datetime64[D]")) for i in tick_idx]

    if cross_pair:
        ax = axes[0]
        va, vb = ~np.isnan(a), ~np.isnan(b)
        ax.plot(x[va], a[va], ".", ms=1.5, color="C0", label=f"{primary_pair} primary")
        ax.plot(x[vb], b[vb], ".", ms=1.5, color="C3", label=f"{cross_pair} cross-check")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel(rf"$\Delta t_{{{target}}}$ (s)")
        ax.set_title("Primary and cross-check estimates (both sign-flipped to target's clock)")
        ax.legend(loc="upper right", fontsize=8, markerscale=3)
        i_main = 1
    else:
        i_main = 0

    ax = axes[i_main]
    vc = ~np.isnan(combined)
    ax.plot(x[vc], combined[vc], ".", ms=1.5, color="C2")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel(rf"$\Delta t_{{{target}}}$ (s)")
    ax.set_title(f"Canonical hourly Δt: primary ({primary_pair}) only")

    if cross_pair:
        ax = axes[i_main + 1]
        vr = ~np.isnan(residual)
        ax.plot(x[vr], residual[vr], ".", ms=1.5, color="C4")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel("primary - cross (s)")
        ax.set_xlabel("Date")
        ax.set_title(f"Validation residual  (RMS = {res_rms:.3f} s)")

    for ax in axes:
        ax.set_xticks(tick_idx)
    axes[-1].set_xticklabels(tick_labels, rotation=30, ha="right")

    fig.tight_layout()
    out = out_dir / "plot_hourly.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    LOG.info("plot -> %s", out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True,
                   help="Station whose clock is being measured (e.g. HYS14).")
    p.add_argument("--primary-pair", required=True,
                   help="Primary pair tag, e.g. HYS12-HYS14.")
    p.add_argument("--cross-pair", default=None,
                   help="Optional cross-check pair tag, e.g. HYS14-HYSB1_lowband.")
    p.add_argument("--chronos-root", default=str(DEFAULT_CHRONOS_ROOT))
    p.add_argument("--ref-window", type=int, default=24 * 30,
                   help="Trailing valid-hour window for anchor median (default 30 days).")
    p.add_argument("--max-shift", type=float, default=50.0,
                   help="Drop hours whose required shift exceeds this magnitude (s).")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(
        args.target, args.primary_pair, args.cross_pair,
        Path(args.chronos_root),
        args.ref_window, args.max_shift,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
