# chronfix — Design

Companion package to `chronos`. Where chronos *measures* timing errors via
ambient-noise cross-correlation, **chronfix applies the measured corrections
to the actual waveform data**.

This document captures open design questions before any code is written.

---

## Scope

- **Input**: per-day clock-error time series produced by chronos, plus the
  raw MiniSEED data on disk.
- **Output**: corrected MiniSEED (and updated metadata) with HYS14's
  timestamps shifted to true UTC.
- **Near-term target**: HYS14 only, MHZ only — the channels and stations
  the chronos diagnostic was run on.

## Inputs from chronos

| File | Type | Meaning |
|---|---|---|
| `data/clock_estimate/HYS14/delta_t.npy` | `float64`, length N_days | Combined Δt per day (s). Positive = HYS14 timestamps are late vs UTC. NaN where unknown. |
| `data/clock_estimate/HYS14/dates.npy` | `datetime64[D]`, length N_days | Master date axis aligned 1:1 with `delta_t`. |
| `data/clock_estimate/HYS14/dt_from_HYS12_HYS14.npy` | `float64` | Single-pair estimate (sign-flipped). |
| `data/clock_estimate/HYS14/dt_from_HYS14_HYSB1.npy` | `float64` | Single-pair estimate. |
| `data/clock_estimate/HYS14/residual.npy` | `float64` | Pair-A − pair-B where both valid. Empirical error bar. |

The combined `delta_t.npy` covers 1330 days of the 1582-day window. The
remaining ~250 days have NaN Δt (gaps in either CCF pair).

## Open design questions

### 1. Within-day Δt model

Δt is sampled at one value per UTC day, but the true clock drifts
continuously *inside* each day. How should chronfix interpolate?

- **(a) Step.** Constant Δt across each whole UTC day; piecewise-constant
  with discontinuities at day boundaries. Trivial to implement. The
  discontinuities are unphysical at day boundaries during drift episodes.
- **(b) Linear between daily samples.** Treat `delta_t[i]` as the value
  at midday of day `i`; interpolate linearly between consecutive days.
  Continuous Δt(t). Requires sample-domain interpolation (resampling).
- **(c) Segmented piecewise-linear.** Detect resync boundaries from
  day-to-day jumps, fit linear drift inside each segment, apply that.
  Most physically faithful; depends on a segment detector that has not
  been written yet.

**Recommended default:** (a) for v0, since (b)/(c) need more machinery
and the residual error of (a) is order ~drift_rate × 12 h ≈ a few
seconds in the worst drift episodes. Revisit if validation says we need
better.

### 2. How to apply the shift to data

- **(i) Timestamp-only.** Subtract Δt from `tr.stats.starttime` (and any
  per-record startime in the mseed). Output trace data is bit-identical;
  only metadata changes. Output mseed will have non-integer-second
  starttimes during drift.
- **(ii) Resample.** Interpolate the trace onto a regular UTC-second
  grid offset by Δt. Output sits cleanly on the canonical grid; trace
  data is resampled (fidelity loss at high frequency depending on
  interpolator).

**Recommended default:** (i). It's lossless and the right primitive for
v0. Resampling can be a separate `--resample` flag later.

### 3. Days with NaN Δt

About 250 days have no chronos measurement (CCF gaps in either pair).
Options:

- **(A) Skip.** Don't write a corrected output for those days. User must
  treat them as "uncorrected; trust at your peril".
- **(B) Linear interpolate.** Fill from neighbours on either side. Risky
  if NaNs span a resync event — the interpolant crosses a discontinuity.
- **(C) Hold last value.** Carry forward the most recent known Δt. Same
  resync-discontinuity risk as (B), and biases toward the previous
  segment.

**Recommended default:** (A). Surface the gap explicitly rather than
fabricate a value.

### 4. Output layout

Mirror the input tree under a sibling root:

```
/data/wsd02/maleen_data/OOI-Data/{sta}/{yr}/{doy:03d}/{sta}.OO.{yr}.{doy:03d}.{ch}
        ↓
/data/wsd02/maleen_data/OOI-Data-corrected/HYS14/{yr}/{doy:03d}/HYS14.OO.{yr}.{doy:03d}.{ch}
```

Plus a sidecar manifest (CSV or JSON) per output day recording: input
path, applied Δt, model used (step/linear/segmented), source
(combined/single-pair), residual, and tool version.

### 5. Channels to correct

For v0: just `MHZ`, the channel chronos measured on. Other HYS14
channels (BHZ, BHE, BHN, HHZ, HHE, HHN, LHZ, SHZ, ...) can use the
*same* Δt(t) since the clock is per-instrument, not per-channel —
trivial to extend once v0 works.

### 6. Folder layout (`chronos/chronfix/`)

```
chronfix/
├── DESIGN.md                       this document
├── __init__.py                     public API exports (later)
├── io.py                           read clock estimate, read mseed
├── model.py                        Δt(t) interpolation choices
├── correct.py                      apply Δt to a Stream
├── manifest.py                     sidecar metadata
└── scripts/
    └── correct_hys14.py            the v0 driver
```

The "package is chronfix, for now it can be a folder" line means this
lives inside the chronos repo as a sub-package and we ship it later as
a separate pip-installable thing.

---

## Brainstorm — applying Δt(t) to MiniSEED traces

The chronos diagnostic now produces both **daily** (`delta_t.npy`) and
**hourly** (`peak_lag_hourly_global.npy` per pair) clock-error estimates.
With hourly resolution available, the within-day approximation becomes
much more defensible — within drift episodes the hourly track is
effectively linear, so `a + b·t` is a near-perfect fit segment by
segment.

The correction problem decomposes into three orthogonal choices.

### Choice 1: how to turn N samples into a continuous Δt(t)

| Model | What it does | Pros | Cons |
|---|---|---|---|
| Step | Hold each sample's value across that hour/day | Trivial; lossless | Discontinuous at bin boundaries |
| Linear interp | Piecewise-linear between samples | Continuous; smooth | Wrong shape across resync events |
| Segment fit | Detect resyncs, fit linear drift in each segment | Matches physics | Needs a segment detector first |

The hourly plot already shows that segments of Δt(t) are essentially
linear. **Segment-fit is the physically correct model.**

### Choice 2: how to apply Δt(t) to the trace

**(a) Timestamp-only.** For every record header / trace `starttime`,
replace `t_apparent` with `t_apparent − Δt(t_apparent)`. Data samples
untouched.

- ✅ Lossless; bit-identical samples.
- ✅ Trivial to implement.
- ❌ Non-integer-second starttimes; many downstream tools dislike this.
- ❌ A single `starttime` can't capture within-trace drift — the rest
  of the trace is off by `Δt(t_end) − Δt(t_start)`.

Only works if the data is also chunked into intervals where Δt is
approximately constant.

**(b) Resample to a true-UTC grid.** Treat samples as occurring at
`t_true = t_apparent − Δt(t_apparent)`. Resample onto the regular UTC
grid.

- ✅ Continuous, smooth, full-day mseed on canonical UTC grid.
- ✅ Within-trace drift handled correctly.
- ❌ Lossy at high frequency (interpolator-dependent).
- ❌ Effective sample rate is non-uniform during drift — reconstruction
  quality varies.

At 8 Hz (MHZ) and the science bands of interest (≤ 3 Hz), resampling
loss is negligible. At higher rates (HHZ at 200 Hz) it matters.

### Choice 3: output chunking

| Layout | Best fit | Comment |
|---|---|---|
| Per-day | Resample-based correction | Matches input layout. |
| Per-hour | Step-Δt + timestamp-only | Discontinuities at hour boundaries. |
| Per-segment | Segment-fit | Most physical. Non-standard layout. |

### Three viable combinations

| Δt model | Application | Output | When this is right |
|---|---|---|---|
| Step (hourly) | Timestamp-only | Per-hour | Quick & lossless v0; accept hour-boundary discontinuities |
| Linear interp | Resample | Per-day | Continuous, on-UTC-grid; balanced default |
| Segment-fit | Timestamp-only + linear correction inside each chunk | Per-segment | Most physically correct; needs segment detector |

### Cross-cutting concerns

- **Resync discontinuities are real.** The clock genuinely jumps at a
  ship visit / GPS sync. Within-segment correction is well-defined;
  across a segment boundary, **after correction, the true-UTC timeline
  has either a gap or an overlap** depending on the sign of the jump:
  - Δt was decreasing at the jump (clock late, then reset) → corrected
    output has a **gap** in true UTC equal to the jump magnitude. Two
    consecutive recorded samples map to true UTC times tens of seconds
    apart, and that interval genuinely was not recorded under correct
    time labels.
  - Δt was increasing at the jump (clock fast, then reset) → corrected
    output has an **overlap**. The same true-UTC interval appears
    twice in adjacent segment files.

  Both are unavoidable physical consequences of the clock having been
  wrong before the resync. chronfix surfaces them as gaps / overlaps
  between segment output files; downstream tools handle them as they
  would any other gap or duplicate. Don't try to fill, merge, or
  interpolate across.
- **NaN-Δt days** (~250 of 1582). Skipping is safest — interpolating
  across a resync boundary is wrong.
- **Sub-sample residual.** At 8 Hz, one sample is 0.125 s. After
  integer-sample shift the residual is < 0.125 s, below diagnostic
  resolution. Higher-rate channels (HHZ 200 Hz) need sub-sample
  precision.
- **Validation hook.** After correcting, re-run `hys_ccf.py` on the
  corrected HYS14 against original HYS12/HYSB1 and verify the hourly
  peak-lag track flattens to the anchor lag. That is the empirical
  truth-test.

### v0 proposal

**Linear-interpolation + resample + per-segment outputs.** The hourly
peak-lag measurement is itself smoothed over ~1.5 h by the 30-min CC
windows, so a step model overstates how sharp our knowledge of Δt is.
Linear interp of the hourly samples matches the smoothness of the
measurement and the linearity of the underlying physical drift.

Concretely:

1. Detect resync events from the hourly Δt track → segment boundaries.
2. For each segment, treat Δt(t) as the linear interpolation of hourly
   samples within that segment.
3. For each output sample at UTC time `t_utc`, look up
   `t_apparent = t_utc + Δt(t_utc)` and read the trace value via
   linear interpolation of the recorded data.
4. Write one output mseed per segment, on a regular UTC sample grid.
5. Across segments, accept the resulting UTC gap or overlap as a real
   feature; do not interpolate or merge across.

Step + timestamp-only is kept as an implementation option
(`method="shift_only"`) for sanity checks: it is bit-identical-data
lossless and useful for spot-validating the resync detector before
trusting the resampler.

---

## Validation strategy

Once a correction has been applied:

1. Re-run `hys_ccf.py` on the corrected HYS14 against original HYS12 and
   HYSB1.
2. Re-run `peak_lag.py` on the new daily stacks.
3. Expected outcome: peak lag is flat at the anchor lag for all corrected
   days. Residual scatter quantifies remaining error.

This closes the loop: chronos detects, chronfix corrects, chronos
re-runs and confirms.

---

## Out of scope for v0

- Other stations / networks. v0 is HYS14-specific.
- Sub-daily Δt models (require sub-daily measurements).
- Multi-channel correction with different per-channel epochs.
- Backfilling NaN days from external clock-log data (e.g. ship logs).
- Re-encoding mseed with non-default block sizes / encodings.
