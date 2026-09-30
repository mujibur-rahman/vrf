"""
datasets.py
===========
Adapter that feeds sc_tasks.py from the raw data actually present in this
project. sc_tasks.COHEN_LOADER_MODULE points here.

WHY THIS FILE EXISTS
--------------------
sc_tasks.load_raw() expects each loader to return a pandas DataFrame of *raw
records* with the columns declared in sc_tasks.SCHEMA. The surge proxy of
Eq. (15) and the demand-density target are both built from records, by binning
and grouping.

The Cohen-project loaders (nyc_taxi_loader.py, geolife_loader.py, ...) do NOT
return that. Each returns an already-collapsed (X, y) pair:
    X : (n, 10) float feature matrix
    y : (n,)    binary FRAUD label
That is the wrong shape (an ndarray tuple, not a DataFrame) and the wrong
semantics (a fraud classification label, not a continuous surge/demand target
grouped by region). Calling them through the adapter raised
    AttributeError: 'tuple' object has no attribute 'columns'
inside load_raw(). This module bridges the gap by reading records directly.

DATA ON DISK
------------
Only NYC is available as a ready-to-read record file: the real TLC FHVHV
parquet. It carries pickup/dropoff timestamps, zone ids, distance and
duration -- everything build_nyc_surge() needs. Geolife / Yelp / Foursquare
exist only as (a) collapsed feature matrices under data/ or (b) unextracted
raw archives in ~/Downloads; wiring those to a record-level density target is
a separate step (see load_geolife / load_yelp / load_foursquare below).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# NYC Taxi -- real records from the TLC FHVHV parquet
# ---------------------------------------------------------------------
NYC_PARQUET = os.environ.get(
    "NYC_PARQUET",
    os.path.expanduser("~/Downloads/fhvhv_tripdata_2025-01.parquet"),
)

# Column window (days from the start of the file) and optional row cap. Kept
# modest by default so a single spatial region accumulates enough 15-minute
# bins to clear sc_tasks.to_federated's min_samples filter, while the frame
# stays small enough to aggregate on a laptop CPU. Override via env for a full
# run, e.g. NYC_DAYS=14.
NYC_DAYS = int(os.environ.get("NYC_DAYS", "7"))
NYC_MAX_ROWS = int(os.environ.get("NYC_MAX_ROWS", "0")) or None  # 0 -> no cap


def load_nyc_taxi(path: str | None = None,
                  n_days: int | None = None,
                  n_samples: int | None = None,
                  seed: int = 42) -> pd.DataFrame:
    """
    Return a raw-record DataFrame with exactly the sc_tasks.SCHEMA['nyc_taxi']
    columns: pickup_datetime, dropoff_datetime, pu_zone, do_zone,
    trip_distance, trip_duration.

    Note on supply (Eq. 15): the FHVHV feed has no per-vehicle identifier.
    `hvfhs_license_num` is a *fleet* licence code (~4 distinct values:
    Uber/Lyft/Via/Juno), not a vehicle id, so it is deliberately NOT passed
    through. With no vehicle column present, build_nyc_surge() falls back to
    the drop-off count proxy and records supply_is_dropoff_proxy=True on the
    frame -- which is the honest state of this data source.
    """
    path = path or NYC_PARQUET
    n_days = NYC_DAYS if n_days is None else n_days
    if n_samples is None:
        n_samples = NYC_MAX_ROWS

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NYC parquet not found at {path!r}. Download the TLC High-Volume "
            f"FHV file (fhvhv_tripdata_YYYY-MM.parquet) from "
            f"https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page and "
            f"set the NYC_PARQUET environment variable to its path."
        )

    cols = ["pickup_datetime", "dropoff_datetime",
            "PULocationID", "DOLocationID", "trip_miles", "trip_time"]
    df = pd.read_parquet(path, columns=cols)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    # Restrict to the first n_days so each zone accumulates dense 15-min bins.
    if n_days:
        start = df["pickup_datetime"].min().normalize()
        df = df[df["pickup_datetime"] < start + pd.Timedelta(days=n_days)]

    # Basic sanity filtering: positive distance/duration, valid zones.
    df = df[(df["trip_miles"] > 0) & (df["trip_time"] > 0)
            & df["PULocationID"].notna() & df["DOLocationID"].notna()]

    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed)

    df = df.rename(columns={
        "PULocationID": "pu_zone",
        "DOLocationID": "do_zone",
        "trip_miles": "trip_distance",
        "trip_time": "trip_duration",
    })
    df["pu_zone"] = df["pu_zone"].astype(int)
    df["do_zone"] = df["do_zone"].astype(int)

    return df[["pickup_datetime", "dropoff_datetime",
               "pu_zone", "do_zone",
               "trip_distance", "trip_duration"]].reset_index(drop=True)


# ---------------------------------------------------------------------
# Foursquare -- TSMC2014 check-ins (record per check-in)
# ---------------------------------------------------------------------
# Each check-in is one activity event with a real timestamp and a venue
# lat/lon, which is exactly what build_density() consumes. Local wall-clock
# time is used (UTC + per-row timezone offset) so hour-of-week is meaningful.
FOURSQUARE_CITY = os.environ.get("FOURSQUARE_CITY", "NYC")   # NYC or TKY
FOURSQUARE_PATH = os.environ.get("FOURSQUARE_PATH", "")


def load_foursquare(path: str | None = None, city: str | None = None,
                    n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
    city = city or FOURSQUARE_CITY
    path = path or FOURSQUARE_PATH or os.path.join(
        "data", "foursquare", f"dataset_TSMC2014_{city}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Foursquare file not found at {path!r}. Expected the TSMC2014 "
            f"check-in file, e.g. data/foursquare/dataset_TSMC2014_NYC.txt."
        )
    cols = ["user_id", "venue_id", "cat_id", "cat",
            "lat", "lon", "tz_offset", "utc"]
    df = pd.read_csv(path, sep="\t", names=cols, encoding="latin-1")

    t_utc = pd.to_datetime(df["utc"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    # Local wall-clock = UTC + timezone offset (minutes), then drop tz info.
    local = t_utc + pd.to_timedelta(df["tz_offset"].astype(float), unit="m")
    df["timestamp"] = local.dt.tz_localize(None)

    df = df.dropna(subset=["timestamp", "lat", "lon", "venue_id"])
    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed)
    return df[["timestamp", "lat", "lon", "venue_id"]].reset_index(drop=True)


# ---------------------------------------------------------------------
# Yelp -- check-ins joined to business coordinates (record per check-in)
# ---------------------------------------------------------------------
YELP_DIR = os.environ.get("YELP_DIR", os.path.join("data", "yelp"))
YELP_MAX = int(os.environ.get("YELP_MAX", "300000"))   # cap on check-in records


def _yelp_business_coords(path: str) -> dict:
    """business_id -> (lat, lon) from the NDJSON business file."""
    import json
    coords = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                b = json.loads(line)
                lat, lon = b.get("latitude"), b.get("longitude")
                if lat is None or lon is None:
                    continue
                lat, lon = float(lat), float(lon)
                if lat == 0.0 and lon == 0.0:
                    continue
                coords[b["business_id"]] = (lat, lon)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                continue
    return coords


def load_yelp(business_path: str | None = None, checkin_path: str | None = None,
              n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
    import json
    business_path = business_path or os.path.join(
        YELP_DIR, "yelp_academic_dataset_business.json")
    checkin_path = checkin_path or os.path.join(
        YELP_DIR, "yelp_academic_dataset_checkin.json")
    for p in (business_path, checkin_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Yelp file not found at {p!r}. Expected the Open Dataset "
                f"business.json and checkin.json under {YELP_DIR}."
            )

    coords = _yelp_business_coords(business_path)
    cap = n_samples or YELP_MAX

    ts, lats, lons, bids = [], [], [], []
    with open(checkin_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = rec.get("business_id")
            if bid not in coords:
                continue
            lat, lon = coords[bid]
            for d in rec.get("date", "").split(","):
                d = d.strip()
                if not d:
                    continue
                ts.append(d); lats.append(lat); lons.append(lon); bids.append(bid)
            if cap and len(ts) >= cap:
                break

    df = pd.DataFrame({"timestamp": pd.to_datetime(ts, errors="coerce"),
                       "lat": lats, "lon": lons, "business_id": bids})
    df = df.dropna(subset=["timestamp", "lat", "lon", "business_id"])
    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Geolife -- GPS trajectory points (record per sampled point)
# ---------------------------------------------------------------------
GEOLIFE_DIR = os.environ.get(
    "GEOLIFE_DIR", os.path.join("data", "geolife", "Data"))
GEOLIFE_STRIDE = int(os.environ.get("GEOLIFE_STRIDE", "40"))   # keep 1/N points
GEOLIFE_MAX = int(os.environ.get("GEOLIFE_MAX", "300000"))     # cap records
GEOLIFE_MAX_USERS = int(os.environ.get("GEOLIFE_MAX_USERS", "0")) or None


def _parse_plt_points(filepath: str, stride: int):
    """Yield (lat, lon, 'YYYY-MM-DD HH:MM:SS') for every `stride`-th point."""
    out = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return out
    for line in lines[6::stride]:               # skip 6-line header
        parts = line.strip().split(",")
        if len(parts) < 7:
            continue
        try:
            lat = float(parts[0]); lon = float(parts[1])
            out.append((lat, lon, f"{parts[5].strip()} {parts[6].strip()}"))
        except (ValueError, IndexError):
            continue
    return out


def load_geolife(data_dir: str | None = None, stride: int | None = None,
                 n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
    data_dir = data_dir or GEOLIFE_DIR
    stride = GEOLIFE_STRIDE if stride is None else stride
    cap = n_samples or GEOLIFE_MAX
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Geolife directory not found at {data_dir!r}. Expected the "
            f"extracted 'Data/<user>/Trajectory/*.plt' tree."
        )

    users = sorted(d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)))
    if GEOLIFE_MAX_USERS:
        users = users[:GEOLIFE_MAX_USERS]

    ts, lats, lons, uids = [], [], [], []
    for user in users:
        traj = os.path.join(data_dir, user, "Trajectory")
        if not os.path.isdir(traj):
            continue
        for plt in os.listdir(traj):
            if not plt.endswith(".plt"):
                continue
            for lat, lon, tstr in _parse_plt_points(os.path.join(traj, plt), stride):
                ts.append(tstr); lats.append(lat); lons.append(lon); uids.append(user)
        if cap and len(ts) >= cap * 2:          # oversample then trim below
            break

    df = pd.DataFrame({"timestamp": pd.to_datetime(ts, errors="coerce"),
                       "lat": lats, "lon": lons, "user_id": uids})
    df = df.dropna(subset=["timestamp", "lat", "lon", "user_id"])
    # Keep the Beijing core so the equal-width grid is not dominated by a few
    # far-flung GPS points (Geolife is overwhelmingly Beijing).
    df = df[(df["lat"].between(39.4, 41.1)) & (df["lon"].between(115.8, 117.1))]
    if cap and len(df) > cap:
        df = df.sample(n=cap, random_state=seed)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    for ds, fn in [("nyc_taxi", lambda: load_nyc_taxi(n_days=2)),
                   ("foursquare", load_foursquare),
                   ("yelp", load_yelp),
                   ("geolife", load_geolife)]:
        try:
            d = fn()
            print(f"{ds:11s} records={len(d):>8d}  cols={list(d.columns)}")
        except Exception as e:
            print(f"{ds:11s} ERROR: {e}")
