# Issue 1: Hourly Peak-Lag Scatter Increases Everywhere After MiniSEED Correction

## Status

**RESOLVED (2026-05-04).** Root cause was the chronfix `_resample` step
linearly interpolating Δt(t) between hourly anchors that carried
±½-sample picker-quantum zigzag, injecting sub-second false time-warp
inside every CCF window. The picker was not at fault; the corrected
CCF itself was degraded.

Fixed by (a) per-segment robust smoothing of Δt at the chronos
clock-model stage so chronfix consumes a shape-preserving smooth
function instead of a picker-quantum staircase, and (b) a leading-NaN
boundary snap in chronfix's `_resample` so float-precision overshoots
no longer drop the entire output. See **Resolution** at the bottom for
the v1 / v2 / v3 outlier-rate comparison and final figures.

## Observation

Two plots, same diagnostic, same picker:

- `data/peak_lag_hourly/HYS12-HYS14/plot.png` — uncorrected: smooth drift
  ramps, clean step-resyncs, tight near-zero band post-2025-07-12, very
  few outliers.
- `data/peak_lag_hourly/HYS12-HYS14_corrected/plot.png` — after chronfix:
  bulk still on zero, but **dense ±60 s outliers throughout the entire
  timeline**, including periods that were clean monotonic ramps in the
  uncorrected plot.

Sub-100 ms hourly Δt jitter cannot directly explain ±60 s peak-lag
outliers, so the previous narrative ("stable-period jitter applied as
correction") was insufficient.

## Diagnostic Trail

Witness picked: **2023-10-11 hour 13 UTC**, mid-drift period.
Uncorrected lag = +11.25 s (riding the drift ramp); corrected lag jumped
to a far noise lobe.

### Step 1 — Sample-grid phase

`tests/test_witness_day.py` and direct ObsPy inspection showed:

- HYS12 raw t0: integer-second, integer-sample.
- HYS14 corrected t0: 23:59:49.250 (= midnight − 10.75 s). Because the
  picker quantizes Δt to 0.125 s = exactly 1 sample at fs = 8 Hz,
  `apparent_start − Δt` always lands on the same integer-sample grid.
- **Sub-sample phase mismatch is structurally zero.** Suspect #1
  eliminated.

### Step 2 — Multi-Trace daily files

For the witness day, `n_traces = 1` — no merge corruption. Suspect #2
not applicable here (still worth checking at trigger boundaries; but the
witness day is mid-drift, no trigger inside the day, so this cannot
explain its outlier).

### Step 3 — Hourly CCF inspection (`tests/test_witness_day.py`)

For hour 13 of 2023-10-11:

```
uncorrected: max|cc| = 482  picked at +10.375 s (clean tall peak)
corrected  : max|cc| = 208  picked at −49.875 s (broad weak peak buried
                            in noise lobes; true peak at ~0 is only 39 %
                            of global max)
```

Peak amplitude cut to ~43 % of uncorrected. Picker is fine; the corrected
CCF itself is much weaker.

### Step 4 — Localize the damage to the resample (`tests/test_resample_vs_intshift.py`)

Three variants, identical CCF processing, same hour:

| variant | peak amplitude | pick | retained |
|---|---|---|---|
| A — raw uncorrected | 536 | +10.75 s | 100 % |
| B — integer-sample shift only (constant Δt for the whole day, no interp, no zigzag) | 492 | 0.00 s | 92 % |
| C — chronfix linear-interp `_resample` (production) | 330 | −0.25 s | 62 % |

B applies the same physical correction as C but with **constant** Δt
across the day and **no fractional interpolation**. It recovers ~92 % of
the uncorrected peak amplitude and lands the peak cleanly on zero. C
loses ~38 % vs B for one reason only: chronfix is linearly interpolating
Δt(t) between hourly anchors that already carry ±0.5 s picker-quantum
zigzag.

## Why the Zigzag Is Fatal

Cleaned hourly Δt around the witness:

```
12:00 → 10.875
13:00 → 11.375    (+0.5 s vs 12:00)
14:00 → 10.500    (−0.875 s vs 13:00)
```

True clock drift is ~1 s/day = 0.04 s/hour. The picker rounds to a
0.125 s quantum, so the cleaned hourly series jumps in steps of 1–2
quanta every few hours — the slow physical drift is sampled by a
staircase. The staircase amplitude (~0.5 s) is harmless as long as Δt is
treated piecewise-constant.

Chronfix's `_resample` linearly interpolates the staircase, so within a
30-min CCF window Δt(t) varies by a fraction of a second non-physically.
Each output sample's apparent-time target zigzags accordingly. HYS12 is
clean. The resulting *relative* timing jitter between the two stations
inside one window scrambles phase at 1–3 Hz, which is exactly the band
where the ballistic peak energy lives.

The earlier intuition ("Δt magnitude is tens of seconds during drift, so
sub-second jitter is invisible") was wrong: it's not the **magnitude** of
Δt that affects CCF coherence, it's the **local variation of Δt within
the CCF window**. The picker quantum guarantees that variation is
hundreds of milliseconds even during smooth drift. That ruins CCF
coherence at 1–3 Hz.

## Fix (station-agnostic, both packages stay standalone)

### Primary — segment-smoothing of Δt at the clock-model stage

In `chronos/scripts/filter_and_triggers.py`, after triggers are detected,
model each inter-trigger segment of Δt:

- If segment slope is below a small threshold, replace with a robust
  median.
- Otherwise, fit a robust linear trend (or low-order spline if drift is
  non-linear).
- Mode is chosen automatically per segment from the data — no
  station-specific dates, no `--stable-start` flag, no `constant-zero`
  mode. Triggers already define segments.

Write the modeled series as `delta_t_hourly_clean.npy` (the file chronfix
consumes). Keep the unmodeled cleaned series as
`delta_t_hourly_filtered_raw.npy` for QC. Chronfix's contract is
unchanged.

### Secondary — `shift_only` when slope is small

When the modeled Δt over a segment varies by less than ½ sample,
`apply_correction` can use chronfix's existing `shift_only` method
instead of `_resample`. This avoids any interpolation kernel at all.
Variant B in step 4 above is essentially this: 92 % of the original CCF
amplitude recovered with zero interpolation. This is a free safety net
on top of the primary fix.

### Rejected from the previous draft

- `--stable-start 2025-07-12` CLI flag — bakes HYS14 episode knowledge
  into chronos.
- `constant-zero` mode — would re-introduce a step at the preceding
  trigger if that segment carried a real constant offset.
- Investigating sample-grid phase or multi-Trace merge as primary
  causes — both eliminated above for the witness case (worth keeping in
  mind for trigger-boundary days, but not the throughout-scatter
  driver).

### Open quality knobs (not bugs)

- Use a higher-order interpolation kernel (windowed sinc / Lanczos)
  inside `_resample` to reduce in-band attenuation. Helpful but
  secondary to fixing the zigzag.
- Round (don't truncate) when casting back to integer dtype.

## Code Locations

- `chronos/src/chronos/scripts/filter_and_triggers.py` — primary fix
  site for segment smoothing.
- `chronfix/src/chronfix/correct.py::_resample` — secondary fix site
  for shift-when-flat behavior.
- `chronos/tests/test_witness_day.py` — picks the witness hour and
  compares per-hour CCFs across pipelines.
- `chronos/tests/test_resample_vs_intshift.py` — three-variant
  experiment that localized the damage to `_resample`.

## Verification Plan

1. Prototype segment smoothing in `tests/` first; confirm the witness
   hour's peak amplitude recovers to ~B-level (~490) without rerunning
   the whole pipeline.
2. Land the smoothing in `filter_and_triggers.py`.
3. Regenerate corrected MiniSEED and CCFs for HYS14.
4. Compare the corrected hourly peak-lag plot against the uncorrected
   one. Success criterion: outlier rate (|lag − rolling-median| > 5 s)
   drops to within ~2× of the uncorrected rate, including drift
   periods.

## Resolution (2026-05-04)

The fix landed in two places:

1. `chronos/scripts/filter_and_triggers.py` — new `model_segments` step
   replaces each inter-trigger segment of cleaned hourly Δt with a
   robust per-segment model: rolling robust median (24 h window) +
   short moving average (6 h) for drift segments, robust segment
   median for flat segments (slope below 0.05 s/day). Station-agnostic;
   trigger boundaries already define the segments so no hardcoded dates.
2. `chronfix/correct.py::_resample` — apparent-time targets within
   half a sample of either grid boundary now snap onto the grid so
   `np.interp` returns a valid value instead of NaN. Eliminates the
   leading-NaN truncation that was silently dropping ~7,800 corrected
   hours.

### Validation: full-timeline hourly peak-lag outlier rates

| | n | \|x\|>5s | \|x\|>20s | \|x\|>40s | med\|x\| |
|---|---|---|---|---|---|
| uncorrected | 31,563 | 34.76 % | 12.76 % | 3.55 % | 1.875 s |
| v1 jittery (original buggy) | 22,244 | 0.81 % | 0.57 % | 0.30 % | 0.125 s |
| v2 linear-fit per segment | 30,084 | 6.33 % | 0.19 % | 0.08 % | 0.375 s |
| **v3 rolling-median per segment** | **30,084** | **0.30 %** | **0.20 %** | **0.10 %** | **0.125 s** |

Reading the table:

- v1 vs uncorrected: bulk distribution does collapse (med\|x\| 1.875 →
  0.125 s), but ±60 s outliers remain dense throughout the timeline,
  including drift periods that were clean ramps in the uncorrected plot.
  This is the bug.
- v2 vs v1: catastrophic >40 s outliers drop 3.75×; chronfix snap fix
  brings 7,840 additional hours through. But the >5 s rate **rises**
  from 0.81 % to 6.33 % because a global linear fit per segment can't
  track curved drift (e.g. 2022-08→2023-01 ramp, 2023-06→2023-09 ramp);
  it leaves 5–20 s curvature residuals.
- v3 vs v2: rolling-median tracks curvature segment-by-segment, so
  >5 s rate drops 21× (6.33 → 0.30 %) while preserving the >20 s and
  >40 s outlier reductions. Bulk median back at the picker quantum.

### Validation: residuals diagnostic on v3

Per inter-trigger segment, residual = (raw cleaned Δt) − (v3 model):

- Median residual = 0.000 ± 0.052 s on every segment (no systematic
  bias).
- MAD residual ≤ 0.115 s ≤ one picker quantum on every segment.
- p95\|residual\| ≈ 0.375–0.477 s ≈ 3 picker quanta — consistent with
  pure quantization noise of the underlying picker output.
- Big-curvature segments (seg 7: 126 days / 56.7 s drift; seg 1:
  65 days / 50.2 s drift; seg 13, 12, 28) all have MAD ≤ 0.094 s. The
  rolling smoother is tracking the drift curve to within picker noise;
  there is no detectable timing signal left in the residuals.

This invalidates the linear-fit hypothesis (v2) and validates the
shape-preserving rolling smoother (v3).

### Long-term reference CCF amplitude

A second validation lever orthogonal to the per-hour picks: stacking
the corrected daily CCFs into a long-term reference and measuring its
RMS. Reference RMS rises from **12.8 (uncorrected) → 40.6 (v3)**, a
**3.16× improvement**, because the per-day ballistics now align
coherently at lag 0 instead of smearing across the drift range.

### Snapshots and figures

All v1, v2, v3 figures and arrays are archived under
`data/results_archive/` with a per-version README documenting the
table above and pointers to:

- `peak_lag_uncorrected_v1.png` / `peak_lag_corrected_v{1,2,3}_*.png`
- `delta_t_hourly_clean_v{2,3}_*.npy`
- `filter_and_triggers_v{1_unmodeled, 2_linear, 3_rolling}.png`
- `v3_residuals_overview.png`, `v3_residuals_per_segment.png`

The chronfix standalone repo example was refreshed to v3 in commit
`3136d66` (`/home/seismic/chronfix-20260504-3136d66.zip`).
