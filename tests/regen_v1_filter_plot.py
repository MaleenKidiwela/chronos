"""Regenerate the v1 (pre-segment-modeling) filter_and_triggers.png from
the preserved raw cleaned Δt and trigger CSV. Writes to
data/results_archive/ with a versioned name; does NOT touch the
production filter_and_triggers.png.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/seismic/chronos/src")
from chronos.scripts.filter_and_triggers import (
    plot_filtered_and_triggers, compute_trigger_periods, SAMPLES_PER_DAY,
    DEFAULT_TRIGGER_THRESHOLD,
)

CLOCK = Path("/home/seismic/chronos/data/clock_estimate/HYS14")
ARCH = Path("/home/seismic/chronos/data/results_archive")

dt_raw = np.load(CLOCK / "delta_t_hourly_filtered_raw.npy")
periods, valid_t, valid_dt, diff_ignore_nans, _ = compute_trigger_periods(
    dt_raw, threshold=DEFAULT_TRIGGER_THRESHOLD, samples_per_day=SAMPLES_PER_DAY
)
out = ARCH / "filter_and_triggers_v1_unmodeled.png"
plot_filtered_and_triggers(
    dt_raw, periods, valid_t, diff_ignore_nans,
    threshold=DEFAULT_TRIGGER_THRESHOLD,
    samples_per_day=SAMPLES_PER_DAY,
    outfile=out,
)
print(f"wrote {out}")
