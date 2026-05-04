#!/usr/bin/env python
"""Apply the chronos clock model to HYS14 MHZ MiniSEED files.

For each input day file (or selected range), splits the trace at trigger
boundaries and writes one corrected output per stable segment overlap.
Output mirrors the input layout under a parallel root directory.

Run from `/home/seismic/chronos`:

    python -m chronfix.scripts.correct_hys14
    python -m chronfix.scripts.correct_hys14 --start 2024-01-01 --end 2024-01-31
    python -m chronfix.scripts.correct_hys14 --method shift_only --workers 4
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

from obspy import read

from chronfix.clock_model import ClockModel
from chronfix.correct import correct_stream

INPUT_ROOT = Path("/data/wsd02/maleen_data/OOI-Data")
OUTPUT_ROOT = Path("/data/wsd02/maleen_data/OOI-Data-corrected")
NETWORK = "OO"
STATION = "HYS14"
CHANNEL = "MHZ"

LOG = logging.getLogger("correct_hys14")


def daily_input(d: date) -> Path:
    doy = d.timetuple().tm_yday
    return (INPUT_ROOT / STATION / f"{d.year}" / f"{doy:03d}"
            / f"{STATION}.{NETWORK}.{d.year}.{doy:03d}.{CHANNEL}")


def daily_output_dir(d: date) -> Path:
    doy = d.timetuple().tm_yday
    return OUTPUT_ROOT / STATION / f"{d.year}" / f"{doy:03d}"


def correct_day(d: date, clock_root: str, method: str) -> dict:
    """Worker. Returns a manifest row."""
    in_path = daily_input(d)
    if not in_path.exists():
        return {"date": str(d), "input": str(in_path), "status": "missing", "n_chunks": 0}

    try:
        st = read(str(in_path))
    except Exception as ex:
        return {"date": str(d), "input": str(in_path), "status": f"read_err:{ex}",
                "n_chunks": 0}

    model = ClockModel.from_chronos(clock_root)
    corrected = correct_stream(st, model, method=method)
    if len(corrected) == 0:
        return {"date": str(d), "input": str(in_path),
                "status": "no_overlap_or_dt_unavailable", "n_chunks": 0}

    out_dir = daily_output_dir(d)
    out_dir.mkdir(parents=True, exist_ok=True)
    doy = d.timetuple().tm_yday
    # Single file per input day, containing one record per stable segment
    # overlap. Matches the original layout so hys_ccf.py can read it.
    out_path = out_dir / f"{STATION}.{NETWORK}.{d.year}.{doy:03d}.{CHANNEL}"
    corrected.write(str(out_path), format="MSEED")
    rows = [{
        "date": str(d),
        "input": str(in_path),
        "output": str(out_path),
        "segment": i,
        "utc_start": str(tr.stats.starttime),
        "utc_end": str(tr.stats.endtime),
        "n_samples": int(tr.stats.npts),
        "method": method,
    } for i, tr in enumerate(corrected, start=1)]
    return {"date": str(d), "n_chunks": len(rows), "rows": rows}


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    p.add_argument("--method", choices=["resample", "shift_only"], default="resample")
    p.add_argument("--clock-root",
                   default="/home/seismic/chronos/data/clock_estimate/HYS14")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--manifest",
                   default="/data/wsd02/maleen_data/OOI-Data-corrected/HYS14/manifest.csv")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    days = list(daterange(args.start, args.end))
    LOG.info("processing %d days  method=%s  workers=%d",
             len(days), args.method, args.workers)

    manifest_rows: list[dict] = []
    counts = {"ok": 0, "missing": 0, "no_overlap": 0, "err": 0, "chunks": 0}

    if args.workers <= 1:
        for d in days:
            res = correct_day(d, args.clock_root, args.method)
            _tally(res, counts, manifest_rows)
            LOG.info("[%s] %s n_chunks=%d", d, res.get("status", "ok"), res["n_chunks"])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(correct_day, d, args.clock_root, args.method): d
                    for d in days}
            for fut in as_completed(futs):
                d = futs[fut]
                res = fut.result()
                _tally(res, counts, manifest_rows)
                LOG.info("[%s] %s n_chunks=%d", d, res.get("status", "ok"), res["n_chunks"])

    if manifest_rows:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not manifest_path.exists()
        with open(manifest_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(manifest_rows)
        LOG.info("manifest -> %s (%d rows)", manifest_path, len(manifest_rows))

    LOG.info("done: ok=%d missing=%d no_overlap/no_dt=%d err=%d chunks=%d",
             counts["ok"], counts["missing"], counts["no_overlap"],
             counts["err"], counts["chunks"])
    return 0


def _tally(res: dict, counts: dict, manifest_rows: list[dict]) -> None:
    status = res.get("status")
    if status == "missing":
        counts["missing"] += 1
    elif status == "no_overlap_or_dt_unavailable":
        counts["no_overlap"] += 1
    elif status and status.startswith("read_err"):
        counts["err"] += 1
    else:
        counts["ok"] += 1
        counts["chunks"] += res["n_chunks"]
        manifest_rows.extend(res.get("rows", []))


if __name__ == "__main__":
    sys.exit(main())
