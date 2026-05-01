# Chronos

Detection and correction of timing errors in seismic data. ObsPy-style API.
Standalone, pip-installable. Distinct from any specific data pipeline.

Near-term goal: diagnose and fix the HYS14 (OOI) timing error. See
`Chronos — Implementation Plan.md` for the roadmap.

## Layout

- `src/chronos/` — installable package (Phase 1+).
- `scripts/` — one-off data acquisition (e.g. `download_hys.py`).
- `diagnostics/` — exploratory scripts for HYS clock diagnosis.
