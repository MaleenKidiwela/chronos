"""chronos — measure timing errors in seismic data via ambient-noise CCF.

Pipeline (each step has a corresponding script under chronos.scripts):

    download_data        →  pull MiniSEED + StationXML from FDSN
    compute_ccf          →  daily inter-station ZZ cross-correlations
    compute_peak_lag     →  hourly ballistic peak-lag track
    combine_clock        →  combine pair shifts into canonical Δt(HYS_target)
    filter_and_triggers  →  Hampel outlier filter + |Δdt| trigger detector
                            (produces the correction file consumed by chronfix)

The "correction file" is the directory written under
    data/clock_estimate/<target_station>/
containing:
    delta_t_hourly_clean.npy   cleaned hourly Δt
    hour_times.npy             datetime64[h] master axis
    trigger_periods.csv        merged trigger intervals

Pass that directory to chronfix.scripts.apply_correction.
"""
__version__ = "0.1.0"
