# HYS14 — Timing Diagnostic

Cross-correlation diagnosis of the HYS14 (OOI Hydrate Ridge OBS) clock issue.
Pairs HYS14 with two neighbouring stations and tracks the lag of the
ballistic-wave peak in the daily cross-correlation function over four years.
This document specifies the methods, presents the results, and states what
they imply for the Chronos timing-correction package.

For project context see `Chronos — Implementation Plan.md`.

---

## 1. Data

| Item | Value |
|---|---|
| Network / stations | `OO.HYS12`, `OO.HYS14`, `OO.HYSB1` |
| Channel | `MHZ` (mid-period vertical, 8 Hz native) |
| Time range | 2022-01-01 → 2026-05-01 (1582 days) |
| Source | EarthScope FDSN (`obspy.clients.fdsn.Client("EARTHSCOPE")`) |
| Local cache | `/data/wsd02/maleen_data/OOI-Data/{sta}/{yr}/{doy:03d}/{sta}.OO.{yr}.{doy:03d}.MHZ` |
| Coverage | HYS12: 1326/1582 days (84%); HYS14: 1326/1582 (84%); HYSB1: 1088/1582 (69%) |

Download script: `scripts/download_hys.py`. StationXML stored under
`/data/wsd02/maleen_data/OOI-Data/StationXML/OO.{sta}..MHZ.xml`.

## 2. Cross-correlation pipeline

Adapted from the single-station preprocessing recipe used in Earthnote's
`phase13_pilot.py`, generalized to inter-station pairs and parameterised over
the science band. No NoisePy dependency; bare ObsPy + NumPy + SciPy.

Implementation: `diagnostics/hys_ccf.py`.

### 2.1 Per-day preprocessing

For each station-day:

1. Read the daily MiniSEED, cast to `float64`, `merge(method=1, fill_value=0.0)`.
2. Cosine-taper around zero-fill regions (100 s on each side of every gap).
3. `detrend("demean")`, `detrend("linear")`.
4. High-pass at `HP_FREQ`; corner chosen below the science band.
5. `remove_response(output="VEL", pre_filt=...)` with corners clamped to
   0.98 × Nyquist of the input trace (some channels have lower native rate).
6. Resample / interpolate to `TARGET_FS = 8.0 Hz`. Downsample uses ObsPy's
   anti-aliased `resample`; upsample uses `interpolate(method="lanczos", a=20)`.

### 2.2 Cross-correlation

Per pair (A, B), per day with both station traces present:

1. Cut 30-min windows with 7.5-min step (75 % overlap) → `cc_len = 1800 s`,
   `cc_step = 450 s`.
2. Per window: demean / detrend / 20 s cosine edge taper / **phase whiten
   with raised-cosine band edges** in the band `(WHITEN_FMIN, WHITEN_FMAX)` /
   one-bit normalisation.
3. Frequency-domain cross-correlation: `irfft(conj(FFT(a)) * FFT(b))`,
   truncated to ±`MAXLAG = 60 s`. Sign convention: positive lag = B lags A.
4. Per day: per-lag median across all windows (robust against transient
   high-amplitude segments). Stored as one row of the daily-stack tensor.
5. Long-term reference: linear (mean) stack of all days with valid daily
   stacks.

Outputs per pair under `data/ccf/<pair_tag>/`:

```
cc_30min.npy        (N_seg, n_lags)     all 30-min CCs concatenated
cc_30min_times.npy  (N_seg,)            fractional day index
cc_daily.npy        (N_days, n_lags)    per-lag median per day
cc_dates.npy        (N_days,) datetime64[D]
cc_ref.npy          (n_lags,)           reference stack
lags.npy            (n_lags,)           lag axis in seconds
```

### 2.3 Two bands

The two pairs need different bands because of their different inter-station
SNR. Parameters used:

| Parameter | HYS12-HYS14 (highband) | HYS14-HYSB1 (`_lowband` tag) |
|---|---|---|
| Science band `(FMIN, FMAX)` | 1.0 – 3.0 Hz | 0.1 – 0.3 Hz |
| Whitening `(WHITEN_FMIN, FMAX)` | 0.5 – 3.8 Hz | 0.05 – 0.4 Hz |
| High-pass `HP_FREQ` | 0.4 Hz | 0.04 Hz |
| `pre_filt` | 0.05 / 0.1 / 3.6 / 3.95 Hz | 0.02 / 0.04 / 3.6 / 3.95 Hz |
| `MAXLAG` | 60 s | 60 s |

The lowband is the **secondary microseism** range. OBS pairs with longer
baselines have most of their coherent ambient noise energy at periods
3–10 s, so dropping the science band raised the reference RMS for
HYS14-HYSB1 from **1.46 → 50.4** (≈ 35× SNR gain) and recovered a clear
ballistic wave packet.

## 3. Peak-lag detection

Implementation: `diagnostics/peak_lag.py`. For each daily CCF row $c(\tau)$,
compute the Hilbert envelope of $c^2$:

$$
e(\tau) = \left| \mathcal{H}\big(c(\tau)^2\big) \right|
$$

and report the lag at the global argmax of $e$. Squaring before
envelope-detection sharpens the peak; the location is the same as
$|\mathcal{H}(c)|$ would give.

Output per pair: `data/peak_lag/<pair_tag>/peak_lag_global.npy`, one
`float64` per day, `NaN` for missing days. `--side {pos,neg}` restricts the
search to one side of zero lag.

A separate diagnostic plotter `diagnostics/ccf_plot.py` writes a
two-panel overview (`ccf_overview.png`) showing the reference stack and
the daily-stack heatmap for any pair.

## 4. Results

### 4.1 HYS12-HYS14 (highband, 1–3 Hz)

| | |
|---|---|
| Reference RMS | 12.8 |
| Ballistic anchor lag | ~0 s (stations near-co-located) |
| Plot | `data/peak_lag/HYS12-HYS14/plot.png` |
| Overview | `data/peak_lag/HYS12-HYS14/ccf_overview.png` |

The peak-lag time series shows multiple **linear drift episodes** punctuated
by **resync events** that snap the lag back near zero. Several episodes
drive the peak to ±60 s, where it walks off the analysis-window edge. From
roughly **May 2025** onward the peak lag is stable at ~0 s with low scatter
— consistent with the clock issue having been resolved.

### 4.2 HYS14-HYSB1 (lowband, 0.1–0.3 Hz)

| | |
|---|---|
| Reference RMS | 50.4 |
| Ballistic anchor lag (post-fix) | ≈ −30 s (acausal side dominates) |
| Plot | `data/peak_lag/HYS14-HYSB1_lowband/plot.png` |
| Overview | `data/peak_lag/HYS14-HYSB1_lowband/ccf_overview.png` |

After 2025-05 the global peak settles tightly at ~−30 s, consistent with a
fixed inter-station travel time. The earlier history shows the same
drift-and-resync pattern as HYS12-HYS14 but with the lag walking around
the −30 s anchor. The most striking single event is the early-2023 ramp
from −30 s up through 0 to +20 s (a ~50 s drift over a few months) before
the next resync brings it back. There is a long unobserved stretch
mid-2023 → late-2024 where HYSB1 was off-air (487 missing days total).

### 4.3 Cross-pair consistency

Both pairs share HYS14, and both show the same temporal pattern of drift
episodes and the same stabilization date in May 2025. The other endpoints
(HYS12 and HYSB1) are different and the two pair plots use different
science bands, so the agreement is not artefactual. The conclusion that
**HYS14 is the source of the timing error** is robust against the choice
of partner station.

## 5. Interpretation

The lag-vs-time pattern is:

- **Linear drift between events** — peak walks at roughly steady rate
  (order ~0.1–0.3 s/day, varies between episodes), suggesting a slow
  oscillator drift rather than discrete jumps.
- **Periodic resync events** — sharp resets back to the anchor lag, likely
  tied to ship visits / power cycles / NTP-style sync events.
- **Stabilization from May 2025** — the drift turns off, consistent with
  either a hardware fix or instrument replacement at that time.

This rules out the "constant offset" hypothesis from the implementation
plan and points to the **drift + segments** model as the right fit.

## 6. Implications for Chronos

The Phase 1 priorities should be reordered around what the diagnostic
shows is actually needed:

1. **Segment detection** — find the resync boundaries automatically from
   day-to-day jumps in the peak-lag time series. Defines the time
   intervals over which a single drift fit is valid.
2. **Per-segment linear drift fit and correction** — within each segment,
   fit `t_apparent − t_true ≈ a + b·(t − t_segment_start)` from the daily
   peak-lag track and apply the inverse to the waveform timestamps.
3. **Constant-offset correction** kept as a building block but is no
   longer the primary use case.
4. **Clock-jump correction** is a degenerate case of segment detection
   (zero-length drift between two anchored values).

The peak-lag time series produced here is the empirical input to (1)–(2):
each segment's slope is the drift rate to invert. Correction validation
is straightforward — apply, recompute the CCF for that segment, and
verify the peak lag flattens.

## 7. Caveats

- `MAXLAG = 60 s` truncates several drift episodes; we cannot tell from
  the present data how much further the lag walked before the next
  resync. Re-running with `--maxlag 200` (or larger) would resolve the
  full drift profile.
- The peak-lag plot reports only the lag of the envelope maximum; it
  does not yet quantify SNR or peak prominence. Days with weak ambient
  noise will have peaks at lag values driven by noise, not the
  ballistic, and currently appear in the plot indistinguishably from
  good days.
- Sub-sample lag accuracy is at the 8 Hz native rate (0.125 s);
  parabolic refinement around the envelope max would push that to
  ~10 ms if needed.
- The lowband HYS14-HYSB1 picks were tested with `--side global`. The
  causal-side `--side pos` picks were bimodal (jumping between two
  arrivals at ~+15 s and ~+37 s) which is why we use the global picker,
  whose argmax falls reliably on the dominant acausal packet at −30 s.

## 8. Reproducing

From `/home/seismic/chronos/`, with conda env `noisepy2` (obspy ≥ 1.5):

```bash
# 1. Data
python scripts/download_hys.py --workers 4

# 2. CCFs
python diagnostics/hys_ccf.py --pairs HYS12-HYS14 --workers 8
python diagnostics/hys_ccf.py --pairs HYS14-HYSB1 --workers 8 \
    --tag lowband --fmin 0.1 --fmax 0.3 \
    --whiten-fmin 0.05 --whiten-fmax 0.4 \
    --hp-freq 0.04 --pre-filt 0.02 0.04 3.6 3.95

# 3. Peak-lag tracks
python diagnostics/peak_lag.py --pair HYS12-HYS14
python diagnostics/peak_lag.py --pair HYS14-HYSB1_lowband

# 4. CCF overviews
python diagnostics/ccf_plot.py --pair HYS12-HYS14
python diagnostics/ccf_plot.py --pair HYS14-HYSB1_lowband
```
