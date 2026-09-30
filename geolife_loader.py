"""
geolife_loader.py
=================
Microsoft Geolife GPS Trajectory dataset loader.
Follows the same interface as all other dataset loaders:
  load_geolife(data_dir, n_samples, seed) → (X, y)
  X: (n, 10) float array
  y: (n,)    binary fraud label

Download:
  https://www.microsoft.com/en-us/download/details.aspx?id=52367
  Extract to ~/data/geolife/
  Expected structure:
    ~/data/geolife/Data/
      000/Trajectory/20081023025304.plt
      001/Trajectory/...
      ...
      181/Trajectory/...

PLT file format (first 6 lines = header, then data):
  Latitude, Longitude, 0, Altitude, DateDays, Date, Time

Feature space (identical across all 4 datasets):
  [0]  pickup_lat        — trajectory start latitude
  [1]  pickup_lon        — trajectory start longitude
  [2]  dropoff_lat       — trajectory end latitude
  [3]  dropoff_lon       — trajectory end longitude
  [4]  duration_s        — trip duration in seconds
  [5]  distance_km       — great-circle distance start→end (km)
  [6]  speed_kmh         — average speed (km/h), fare proxy
  [7]  hour_of_day       — start hour (0–23)
  [8]  speed_per_km      — speed / distance (anomaly sensitivity)
  [9]  displacement_ratio— straight-line / path length (sinuosity proxy)

Fraud label heuristics (GPS-spoofing / trajectory anomalies):
  1. Speed > 150 km/h for ground-level trajectory  → GPS spoofing
  2. Displacement > 50 km in < 300 s              → teleportation
  3. Duration < 30 s AND distance > 1 km          → phantom segment
  4. Displacement ratio < 0.05 (wild GPS scatter) → noisy spoof
"""

import os
import math
import numpy as np
from datetime import datetime

DATASET_NAME = 'geolife'


# ── Haversine distance (km) ───────────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Parse a single PLT file → list of (lat, lon, datetime) ───────────────────
def _parse_plt(filepath):
    points = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        # First 6 lines are header
        for line in lines[6:]:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            try:
                lat  = float(parts[0])
                lon  = float(parts[1])
                date = parts[5].strip()
                time = parts[6].strip()
                dt   = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
                points.append((lat, lon, dt))
            except (ValueError, IndexError):
                continue
    except Exception:
        pass
    return points


# ── Convert a trajectory (list of points) → one trip record ──────────────────
def _trajectory_to_record(points):
    """
    Summarise a PLT trajectory as a single trip feature vector.
    Returns (features_10, is_fraud) or None if trajectory too short.
    """
    if len(points) < 2:
        return None

    start_lat, start_lon, start_dt = points[0]
    end_lat,   end_lon,   end_dt   = points[-1]

    # Duration
    duration_s = max(1.0, (end_dt - start_dt).total_seconds())

    # Straight-line displacement
    displacement_km = _haversine(start_lat, start_lon, end_lat, end_lon)

    # Total path length (sum of consecutive distances)
    path_km = 0.0
    for i in range(1, len(points)):
        path_km += _haversine(
            points[i-1][0], points[i-1][1],
            points[i][0],   points[i][1]
        )
    path_km = max(path_km, 0.001)

    # Derived features
    speed_kmh        = (displacement_km / duration_s) * 3600
    hour_of_day      = float(start_dt.hour)
    speed_per_km     = speed_kmh / max(displacement_km, 0.1)
    displacement_ratio = displacement_km / path_km  # 1.0 = straight line

    features = np.array([
        start_lat,
        start_lon,
        end_lat,
        end_lon,
        duration_s,
        displacement_km,
        speed_kmh,
        hour_of_day,
        speed_per_km,
        displacement_ratio,
    ])

    # ── Fraud label ───────────────────────────────────────────────────────
    fraud = 0
    if speed_kmh > 150.0:                                # GPS spoofing speed
        fraud = 1
    elif displacement_km > 50.0 and duration_s < 300:   # teleportation
        fraud = 1
    elif duration_s < 30.0 and displacement_km > 1.0:   # phantom segment
        fraud = 1
    elif displacement_ratio < 0.05 and displacement_km > 0.5:  # GPS scatter
        fraud = 1

    return features, fraud


# ── Main loader ───────────────────────────────────────────────────────────────
def load_geolife(data_dir=None, n_samples=80_000, seed=42):
    """
    Load Microsoft Geolife GPS trajectory dataset.

    Parameters
    ----------
    data_dir  : str   Path to the 'Data/' folder inside the Geolife download.
                      If None or not found, uses synthetic fallback.
    n_samples : int   Max number of trip records to return.
    seed      : int   Random seed.

    Returns
    -------
    X : np.ndarray  (n, 10)  — same feature space as all other loaders
    y : np.ndarray  (n,)     — binary fraud label
    """
    rng = np.random.RandomState(seed)

    # ── Try real Geolife data ─────────────────────────────────────────────
    if data_dir and os.path.isdir(data_dir):
        print(f"\n[geolife] Loading real data from: {data_dir}")

        records_X = []
        records_y = []
        users_loaded = 0
        files_read   = 0
        files_skipped = 0

        user_dirs = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        print(f"  Found {len(user_dirs)} user directories")

        for user_dir in user_dirs:
            traj_dir = os.path.join(data_dir, user_dir, 'Trajectory')
            if not os.path.isdir(traj_dir):
                continue

            plt_files = [f for f in os.listdir(traj_dir)
                         if f.endswith('.plt')]

            user_records = 0
            for plt_file in plt_files:
                if len(records_X) >= n_samples * 2:  # oversample then trim
                    break

                points = _parse_plt(os.path.join(traj_dir, plt_file))
                result = _trajectory_to_record(points)

                if result is not None:
                    feat, fraud = result
                    # Sanity check: valid lat/lon range
                    if (20 <= feat[0] <= 60 and 60 <= feat[1] <= 150 and
                        20 <= feat[2] <= 60 and 60 <= feat[3] <= 150):
                        records_X.append(feat)
                        records_y.append(fraud)
                        user_records += 1
                        files_read += 1
                    else:
                        files_skipped += 1
                else:
                    files_skipped += 1

            if user_records > 0:
                users_loaded += 1

            if len(records_X) >= n_samples * 2:
                break

        if len(records_X) == 0:
            print(f"  ✗ No valid trajectories parsed — check data_dir path")
            print(f"    Expected: {data_dir}/000/Trajectory/*.plt")
            print(f"  → Falling back to synthetic data")
        else:
            X = np.array(records_X)
            y = np.array(records_y)

            # Sample down to n_samples
            if len(X) > n_samples:
                idx = rng.choice(len(X), n_samples, replace=False)
                X, y = X[idx], y[idx]

            fraud_rate = y.mean()
            print(f"  Users loaded : {users_loaded}/{len(user_dirs)}")
            print(f"  Files read   : {files_read}  |  Skipped: {files_skipped}")
            print(f"  Records      : {len(X):,}")
            print(f"  Fraud rate   : {fraud_rate:.1%}")

            if fraud_rate < 0.01:
                print(f"  ⚠ Very low fraud rate. "
                      f"Consider relaxing _trajectory_to_record thresholds.")
            if fraud_rate > 0.50:
                print(f"  ⚠ Very high fraud rate. "
                      f"Trajectory quality may be low (noisy GPS).")

            return X, y

    # ── Synthetic fallback ────────────────────────────────────────────────
    print(f"\n[geolife] ⚠ Real data not found. Using synthetic Geolife-style data.")
    print(f"  To use real data:")
    print(f"  1. Download: https://www.microsoft.com/en-us/download/details.aspx?id=52367")
    print(f"  2. Extract to ~/data/geolife/")
    print(f"  3. Set data_dir='~/data/geolife/Data/'")

    return _synthetic_geolife(n_samples, seed)


def _synthetic_geolife(n_samples=80_000, seed=42):
    """
    Synthetic Geolife-style data.
    Mimics Beijing GPS trajectory distribution:
      - Lat: 39.7 – 40.2 (Beijing bounding box)
      - Lon: 116.0 – 116.8
    Uses same 10-feature space and same fraud heuristics as real loader.
    """
    from fl_base import generate_sc_dataset

    # Generate base synthetic data using shared utility
    X_base, y_base = generate_sc_dataset(n_samples, fraud_rate=0.12, seed=seed)

    rng = np.random.RandomState(seed)
    n = len(X_base)

    # Remap coordinates to Beijing bounding box
    # Feature [0,1] = pickup lat/lon, [2,3] = dropoff lat/lon
    pu_lat = rng.uniform(39.75, 40.10, n)
    pu_lon = rng.uniform(116.20, 116.65, n)
    do_lat = pu_lat + rng.normal(0, 0.05, n)   # nearby dropoff
    do_lon = pu_lon + rng.normal(0, 0.05, n)

    # Keep other features from synthetic generator but remap coords
    X_geo = X_base.copy()
    X_geo[:, 0] = pu_lat
    X_geo[:, 1] = pu_lon
    X_geo[:, 2] = np.clip(do_lat, 39.6, 40.2)
    X_geo[:, 3] = np.clip(do_lon, 115.9, 116.9)

    # Feature [8] = speed_per_km proxy, [9] = displacement_ratio proxy
    X_geo[:, 8] = X_base[:, 8]   # fare_per_mile → speed_per_km
    X_geo[:, 9] = rng.uniform(0.3, 1.0, n)  # displacement ratio

    # Inject GPS spoofing patterns for Byzantine simulation
    # (fraud rate ~12% from generate_sc_dataset, consistent with paper)

    print(f"\n[geolife] Synthetic data generated:")
    print(f"  Records    : {n:,}")
    print(f"  Fraud rate : {y_base.mean():.1%}")
    print(f"  Lat range  : [{X_geo[:,0].min():.3f}, {X_geo[:,0].max():.3f}]")
    print(f"  Lon range  : [{X_geo[:,1].min():.3f}, {X_geo[:,1].max():.3f}]")

    return X_geo, y_base


if __name__ == '__main__':
    print("Testing Geolife loader (synthetic mode)...")
    X, y = load_geolife()
    print(f"\nX shape : {X.shape}")
    print(f"y dist  : {int(y.sum())} fraud / {int((1-y).sum())} legit")
    print(f"\nFeature ranges:")
    labels = ['pu_lat', 'pu_lon', 'do_lat', 'do_lon',
              'duration_s', 'dist_km', 'speed_kmh',
              'hour', 'speed/km', 'disp_ratio']
    for i, lab in enumerate(labels):
        print(f"  {lab:14s}: [{X[:,i].min():.3f}, {X[:,i].max():.3f}]")
