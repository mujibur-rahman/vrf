"""
nyc_taxi_loader.py
==================
Real NYC Taxi (TLC FHVHV) data loader for the FL experiment pipeline.

Download data from:
  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Files needed (2024-2025 parquet, High Volume FHV):
  fhvhv_tripdata_2025-01.parquet  (or any month)
  taxi_zone_lookup.csv            (LocationID → lat/lon)
  Download taxi_zone_lookup from the same TLC page.

Usage:
  from nyc_taxi_loader import load_nyc_taxi
  X, y = load_nyc_taxi('path/to/fhvhv_tripdata_2025-01.parquet',
                        'path/to/taxi_zone_lookup.csv',
                        n_samples=100_000)

Falls back to synthetic data if files are not found,
so the pipeline runs even without the real dataset (for testing).
"""

import numpy as np
import os

DATASET_NAME = 'nyc-taxi'

# ── Zone centroid lookup (approximate lat/lon per LocationID) ─────────────────
# Full mapping requires taxi_zone_lookup.csv from TLC.
# These are the most common zones as a fallback approximation.
ZONE_CENTROIDS_APPROX = {
    1:  (40.8994, -73.8720), 4:  (40.7204, -74.0099),
    7:  (40.7011, -74.0165), 13: (40.7745, -73.8729),
    17: (40.7831, -73.9712), 24: (40.7282, -74.0085),
    25: (40.7197, -74.0033), 33: (40.7647, -73.9997),
    36: (40.7282, -73.7957), 37: (40.6943, -73.9896),
    40: (40.7223, -74.0050), 41: (40.7175, -74.0130),
    42: (40.7150, -74.0063), 43: (40.7225, -74.0046),
    45: (40.6514, -73.9496), 48: (40.7580, -73.9855),
    50: (40.7484, -73.9967), 68: (40.7480, -73.9860),
    79: (40.7782, -73.9636), 87: (40.7749, -73.8729),
    90: (40.8533, -73.8834), 100:(40.6651, -73.9435),
    107:(40.7413, -73.9991), 113:(40.7922, -73.9461),
    114:(40.7142, -74.0115), 116:(40.8009, -73.9449),
    119:(40.7971, -73.9379), 120:(40.7958, -73.9392),
    125:(40.7480, -73.9860), 127:(40.7480, -73.9860),
    128:(40.7480, -73.9860), 137:(40.7480, -73.9855),
    140:(40.7561, -73.9872), 141:(40.7556, -73.9870),
    142:(40.7617, -73.9754), 143:(40.7624, -73.9738),
    144:(40.7631, -73.9723), 148:(40.7951, -73.9668),
    151:(40.7193, -74.0029), 152:(40.7179, -74.0050),
    153:(40.7167, -74.0068), 158:(40.7500, -73.9740),
    161:(40.7577, -73.9780), 162:(40.7514, -73.9777),
    163:(40.7500, -73.9770), 164:(40.7490, -73.9795),
    166:(40.7481, -73.9817), 170:(40.7471, -73.9840),
    186:(40.7555, -73.9866), 194:(40.7291, -73.7993),
    202:(40.6454, -73.7820), 209:(40.7628, -73.9807),
    211:(40.7570, -73.9800), 224:(40.7559, -73.9862),
    229:(40.7569, -73.9865), 230:(40.7575, -73.9869),
    231:(40.7580, -73.9867), 232:(40.7587, -73.9864),
    233:(40.7594, -73.9860), 234:(40.7601, -73.9855),
    236:(40.7614, -73.9841), 237:(40.7621, -73.9833),
    238:(40.7631, -73.9823), 239:(40.7641, -73.9812),
    243:(40.7661, -73.9791), 244:(40.7671, -73.9781),
    246:(40.7684, -73.9768), 249:(40.7700, -73.9750),
    255:(40.7730, -73.9714), 256:(40.7740, -73.9703),
    257:(40.7750, -73.9692), 258:(40.7760, -73.9680),
    260:(40.7781, -73.9656), 261:(40.6877, -73.9840),
    262:(40.7580, -73.9855), 263:(40.7583, -73.9850),
}


def _get_zone_latlon(location_id, zone_map):
    """Return (lat, lon) for a LocationID."""
    if zone_map is not None and location_id in zone_map:
        return zone_map[location_id]
    if location_id in ZONE_CENTROIDS_APPROX:
        return ZONE_CENTROIDS_APPROX[location_id]
    # Random NYC coordinate as last resort
    return (
        np.random.uniform(40.60, 40.90),
        np.random.uniform(-74.05, -73.75)
    )


def _load_zone_lookup(csv_path):
    """Load taxi_zone_lookup.csv → dict {LocationID: (lat, lon)}."""
    import csv
    zone_map = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lid = int(row.get('LocationID', 0))
                # TLC lookup has borough/zone/service_zone but NOT lat/lon.
                # Use centroid approximation keyed by LocationID.
                if lid in ZONE_CENTROIDS_APPROX:
                    zone_map[lid] = ZONE_CENTROIDS_APPROX[lid]
        print(f"  Zone lookup: {len(zone_map)} zones loaded from {csv_path}")
    except Exception as e:
        print(f"  Zone lookup warning: {e} — using built-in centroids")
    return zone_map if zone_map else None


def _fraud_label(fare, distance_miles, duration_s, hour):
    """
    Heuristic fraud label for NYC taxi trips.
    Fraudulent if any condition holds:
      1. Very high fare-per-mile (> $8/mile) AND short trip (< 1 mile)
         → price gouging / fare manipulation
      2. Extremely short duration (< 60s) AND non-trivial fare (> $15)
         → phantom trip
      3. Very high fare (> $120) with very short distance (< 0.5 miles)
         → GPS spoofing to inflate surge
    Returns 1 (fraud) or 0 (legitimate).
    """
    fare_per_mile = fare / max(distance_miles, 0.01)
    if fare_per_mile > 8.0 and distance_miles < 1.0:
        return 1
    if duration_s < 60 and fare > 15:
        return 1
    if fare > 120 and distance_miles < 0.5:
        return 1
    return 0


def load_nyc_taxi(parquet_path=None, zone_lookup_path=None,
                  n_samples=100_000, seed=42):
    """
    Load and preprocess NYC TLC FHVHV trip data.

    Parameters
    ----------
    parquet_path    : str  Path to fhvhv_tripdata_*.parquet
                           If None or not found, uses synthetic fallback.
    zone_lookup_path: str  Path to taxi_zone_lookup.csv (optional)
    n_samples       : int  Max samples to load
    seed            : int  Random seed for sampling and fallback

    Returns
    -------
    X : np.ndarray  shape (n, 10)
        Features: [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                   duration_s, distance_miles, fare,
                   hour_of_day, fare_per_mile, duration_per_mile]
    y : np.ndarray  shape (n,)  Binary labels: 0=legitimate, 1=fraudulent
    """
    rng = np.random.RandomState(seed)

    # ── Try loading real data ──────────────────────────────────────────────
    if parquet_path and os.path.exists(parquet_path):
        print(f"\n[nyc-taxi] Loading real TLC data from: {parquet_path}")
        try:
            import pandas as pd

            df = pd.read_parquet(parquet_path)
            print(f"  Raw rows: {len(df):,}")

            # ── Load zone lookup ──────────────────────────────────────────
            zone_map = None
            if zone_lookup_path and os.path.exists(zone_lookup_path):
                zone_map = _load_zone_lookup(zone_lookup_path)

            # ── Column name normalisation ─────────────────────────────────
            # FHVHV 2024-2025 column names
            col_map = {
                'pickup_datetime':    ['pickup_datetime', 'tpep_pickup_datetime'],
                'dropoff_datetime':   ['dropoff_datetime', 'tpep_dropoff_datetime'],
                'PULocationID':       ['PULocationID', 'pulocationid'],
                'DOLocationID':       ['DOLocationID', 'dolocationid'],
                'trip_miles':         ['trip_miles', 'trip_distance'],
                'trip_time':          ['trip_time', 'trip_duration'],
                'base_passenger_fare':['base_passenger_fare', 'fare_amount', 'total_amount'],
            }

            def find_col(df, candidates):
                for c in candidates:
                    if c in df.columns:
                        return c
                return None

            pu_dt_col  = find_col(df, col_map['pickup_datetime'])
            do_dt_col  = find_col(df, col_map['dropoff_datetime'])
            pu_loc_col = find_col(df, col_map['PULocationID'])
            do_loc_col = find_col(df, col_map['DOLocationID'])
            miles_col  = find_col(df, col_map['trip_miles'])
            time_col   = find_col(df, col_map['trip_time'])
            fare_col   = find_col(df, col_map['base_passenger_fare'])

            print(f"  Columns found: PU={pu_loc_col}, DO={do_loc_col}, "
                  f"miles={miles_col}, time={time_col}, fare={fare_col}")

            # ── Drop nulls in key columns ─────────────────────────────────
            key_cols = [c for c in [pu_loc_col, do_loc_col, miles_col,
                                     fare_col] if c]
            df = df.dropna(subset=key_cols)

            # ── Sample ────────────────────────────────────────────────────
            if len(df) > n_samples:
                df = df.sample(n=n_samples, random_state=seed)
            df = df.reset_index(drop=True)
            n = len(df)
            print(f"  Sampled: {n:,} rows")

            # ── Extract features ──────────────────────────────────────────
            # Pickup/dropoff lat-lon from LocationID
            pu_lats, pu_lons = [], []
            do_lats, do_lons = [], []

            pu_ids = df[pu_loc_col].fillna(0).astype(int).values
            do_ids = df[do_loc_col].fillna(0).astype(int).values

            for pid, did in zip(pu_ids, do_ids):
                plat, plon = _get_zone_latlon(pid, zone_map)
                dlat, dlon = _get_zone_latlon(did, zone_map)
                pu_lats.append(plat); pu_lons.append(plon)
                do_lats.append(dlat); do_lons.append(dlon)

            # Duration
            if time_col:
                duration_s = df[time_col].fillna(600).values.astype(float)
            elif pu_dt_col and do_dt_col:
                duration_s = (
                    pd.to_datetime(df[do_dt_col]) -
                    pd.to_datetime(df[pu_dt_col])
                ).dt.total_seconds().fillna(600).values
            else:
                duration_s = rng.uniform(300, 3600, n)

            # Hour of day
            if pu_dt_col:
                hours = pd.to_datetime(df[pu_dt_col]).dt.hour.fillna(12).values
            else:
                hours = rng.randint(0, 24, n).astype(float)

            distance_miles = df[miles_col].fillna(1.0).clip(0.1, 50).values
            fare           = df[fare_col].fillna(10.0).clip(2.5, 200).values

            # Derived features
            fare_per_mile    = fare / np.maximum(distance_miles, 0.1)
            duration_per_mile= duration_s / np.maximum(distance_miles, 0.1)

            X = np.column_stack([
                pu_lats, pu_lons, do_lats, do_lons,
                duration_s, distance_miles, fare, hours,
                fare_per_mile, duration_per_mile
            ])

            # ── Fraud labels ──────────────────────────────────────────────
            y = np.array([
                _fraud_label(fare[i], distance_miles[i], duration_s[i], hours[i])
                for i in range(n)
            ])

            fraud_rate = y.mean()
            print(f"  Features: {X.shape}  |  Fraud rate: {fraud_rate:.1%}")

            if fraud_rate < 0.02:
                print(f"  ⚠ Very low fraud rate ({fraud_rate:.1%}). "
                      f"Consider adjusting _fraud_label() thresholds.")
            if fraud_rate > 0.40:
                print(f"  ⚠ Very high fraud rate ({fraud_rate:.1%}). "
                      f"Heuristic may be too aggressive.")

            return X, y

        except Exception as e:
            print(f"  ✗ Failed to load parquet: {e}")
            print(f"  → Falling back to synthetic NYC Taxi data")

            # ── Synthetic fallback ────────────────────────────────────────────────
            print(f"\n[nyc-taxi] ⚠ Real data not found. Using synthetic NYC Taxi data.")
            print(f"  To use real data:")
            print(f"  1. Download from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page")
            print(f"  2. Pass parquet_path='path/to/fhvhv_tripdata_2025-01.parquet'")

    from fl_base import generate_sc_dataset
    return generate_sc_dataset(n_samples, fraud_rate=0.15, seed=seed)


if __name__ == '__main__':
    # Quick test
    print("Testing NYC Taxi loader...")
    X, y = load_nyc_taxi()
    print(f"X shape: {X.shape}")
    print(f"y distribution: {int(y.sum())} fraud / {int((1-y).sum())} legit")
    print(f"Feature ranges:")
    labels = ['pu_lat','pu_lon','do_lat','do_lon',
              'duration_s','miles','fare','hour',
              'fare/mile','dur/mile']
    for i, lab in enumerate(labels):
        print(f"  {lab:12s}: [{X[:,i].min():.2f}, {X[:,i].max():.2f}]")
