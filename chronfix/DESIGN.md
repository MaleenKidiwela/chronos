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
