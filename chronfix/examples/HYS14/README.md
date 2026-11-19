# HYS14 correction file

Bundled chronos output for the OOI Hydrate Ridge OBS HYS14 (network OO,
channel MHZ) clock fix, covering 2022-08 → 2026-05.

| File | Type | Meaning |
|---|---|---|
| `delta_t_hourly_clean.npy` | float64 (N,) | Cleaned hourly Δt (s); NaN where masked. Δt > 0 ⇒ HYS14 timestamps are late vs UTC. |
| `hour_times.npy` | datetime64[h] (N,) | UTC hour timestamps aligned with `delta_t_hourly_clean`. |
| `trigger_periods.csv` | — | 33 merged trigger intervals (31 clock discontinuities + 2 short coherent excursions marked undefined). chronfix splits output mseed at each. |
| `delta_t_hourly.csv` | csv | Same Δt as `delta_t_hourly_clean.npy`, columns `time_utc,delta_t_s`. |
| `filter_and_triggers.png` | png | Diagnostic plot from chronos. |

## Reproducing the fix on HYS14 mseed

```bash
python -m chronfix.scripts.apply_correction \
    --correction-dir examples/HYS14 \
    --network OO --station HYS14 --channel MHZ \
    --input-root  /path/to/raw/mseed/root \
    --output-root /path/to/corrected/mseed/root \
    --start 2022-08-13 --workers 8
```

See the project-wide methods write-up at
`docs/HYS14 — Correction Method.md` (in the chronos repo).
