#!/usr/bin/env python
"""Download MHZ (8 Hz) data for OO.HYS{12,14,B1} from EarthScope FDSN.

Saves per-day MiniSEED files into the layout the Julia Plot_correlation.ipynb
notebook reads from:

    /data/wsd02/maleen_data/OOI-Data/{sta}/{yr}/{doy:03d}/{sta}.OO.{yr}.{doy:03d}.MHZ
    /data/wsd02/maleen_data/OOI-Data/StationXML/OO.{sta}..MHZ.xml

Idempotent: skips files that already exist on disk. Logs gaps/errors per day
so a re-run only fetches what is missing.

Usage:
    python download_hys.py                       # 2022-01-01 .. today, all 3 stations
    python download_hys.py --start 2022-01-01 --end 2022-12-31
    python download_hys.py --stations HYS14
    python download_hys.py --workers 4           # parallel days per station
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

NETWORK = "OO"
CHANNEL = "MHZ"
LOCATION = "*"  # OOI uses various location codes; let the server resolve
STATIONS_DEFAULT = ("HYS12", "HYS14", "HYSB1")
DATA_ROOT = Path("/data/wsd02/maleen_data/OOI-Data")

LOG = logging.getLogger("download_hys")


def daily_path(sta: str, d: date) -> Path:
    return (
        DATA_ROOT / sta / f"{d.year}" / f"{d.timetuple().tm_yday:03d}"
        / f"{sta}.{NETWORK}.{d.year}.{d.timetuple().tm_yday:03d}.{CHANNEL}"
    )


def stationxml_path(sta: str) -> Path:
    return DATA_ROOT / "StationXML" / f"{NETWORK}.{sta}..{CHANNEL}.xml"


def fetch_stationxml(client: Client, sta: str) -> None:
    out = stationxml_path(sta)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    inv = client.get_stations(
        network=NETWORK, station=sta, channel=CHANNEL,
        level="response",
    )
    inv.write(str(out), format="STATIONXML")
    LOG.info("StationXML written: %s", out)


def fetch_day(client: Client, sta: str, d: date) -> str:
    """Fetch one day. Returns 'ok', 'skip', 'gap', or 'err:<msg>'."""
    out = daily_path(sta, d)
    if out.exists() and out.stat().st_size > 0:
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = UTCDateTime(d.year, d.month, d.day)
    t1 = t0 + 86400.0
    try:
        st = client.get_waveforms(
            network=NETWORK, station=sta, location=LOCATION,
            channel=CHANNEL, starttime=t0, endtime=t1,
        )
    except FDSNNoDataException:
        return "gap"
    except Exception as ex:  # pragma: no cover
        return f"err:{type(ex).__name__}:{ex}"
    if len(st) == 0:
        return "gap"
    st.write(str(out), format="MSEED")
    return "ok"


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def run(stations, start: date, end: date, workers: int) -> None:
    client = Client("EARTHSCOPE")
    for sta in stations:
        try:
            fetch_stationxml(client, sta)
        except Exception as ex:
            LOG.warning("StationXML fetch failed for %s: %s", sta, ex)

        days = list(daterange(start, end))
        LOG.info("[%s] %d days: %s .. %s", sta, len(days), start, end)
        counts = {"ok": 0, "skip": 0, "gap": 0, "err": 0}

        if workers <= 1:
            for d in days:
                status = fetch_day(client, sta, d)
                _tally(counts, status, sta, d)
        else:
            # One Client per thread is safer (urllib3 sessions aren't shared cleanly)
            local_clients = [Client("EARTHSCOPE") for _ in range(workers)]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(fetch_day, local_clients[i % workers], sta, d): d
                    for i, d in enumerate(days)
                }
                for fut in as_completed(futs):
                    d = futs[fut]
                    _tally(counts, fut.result(), sta, d)

        LOG.info(
            "[%s] done: ok=%d skip=%d gap=%d err=%d",
            sta, counts["ok"], counts["skip"], counts["gap"], counts["err"],
        )


def _tally(counts, status: str, sta: str, d: date) -> None:
    if status == "ok":
        counts["ok"] += 1
    elif status == "skip":
        counts["skip"] += 1
    elif status == "gap":
        counts["gap"] += 1
        LOG.debug("[%s %s] no data", sta, d)
    else:
        counts["err"] += 1
        LOG.warning("[%s %s] %s", sta, d, status)


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stations", nargs="+", default=list(STATIONS_DEFAULT))
    p.add_argument("--start", type=parse_date, default=date(2022, 1, 1))
    p.add_argument("--end", type=parse_date, default=date.today())
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args.stations, args.start, args.end, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
