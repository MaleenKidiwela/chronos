# Chronos

Detection of timing errors in seismic data via ambient-noise cross-correlation.
Companion package `chronfix` applies the measured corrections to waveform data.

Near-term goal: diagnose and fix the HYS14 (OOI Hydrate Ridge OBS) timing
error. See `Chronos — Implementation Plan.md` for project background and
`HYS14 — Timing Diagnostic.md` for methods + results of the diagnostic run.

## Layout

```
chronos/
├── README.md
├── Chronos — Implementation Plan.md   project background
├── HYS14 — Timing Diagnostic.md       diagnostic write-up
├── scripts/
│   └── download_hys.py                EarthScope FDSN pull for OO.HYS{12,14,B1}.MHZ
├── diagnostics/
│   ├── hys_ccf.py                     daily inter-station ZZ CCFs
│   ├── peak_lag.py                    per-day ballistic peak lag from envelope of CC^2
│   ├── ccf_plot.py                    reference + 2D-stack overview plot
│   ├── realign_ccf.py                 shift each daily CCF onto an anchor lag
│   └── combine_clock.py               merge two pair-shifts into a single Δt(HYS14)
└── chronfix/
    └── DESIGN.md                      open design questions for the correction package
```

## Pipeline at a glance

1. `scripts/download_hys.py` — pull MHZ from EarthScope.
2. `diagnostics/hys_ccf.py` — daily 30-min ZZ cross-correlations per pair.
3. `diagnostics/peak_lag.py` — track the lag of the ballistic peak per day.
4. `diagnostics/realign_ccf.py` — back out each day's clock-error shift.
5. `diagnostics/combine_clock.py` — fuse the two pair shifts into a single
   per-day HYS14 clock-error time series.
6. `chronfix/` — apply the time series to the raw waveforms (TBD).

Outputs live under `data/` (gitignored — regenerate locally).
