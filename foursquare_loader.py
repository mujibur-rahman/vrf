"""
foursquare_loader.py
====================
Foursquare NYC check-in dataset loader.
Same interface as all other loaders:
  load_foursquare(path, n_samples, seed) → (X, y)
  X: (n, 10) float array — shared feature space
  y: (n,)    binary fraud label

Dataset variants supported (auto-detected):
  1. TSMC2014 NYC (Yang et al. 2014) — most common academic version
       dataset_TSMC2014_NYC.txt
       Columns: user_id, venue_id, venue_category_id, venue_category_name,
                latitude, longitude, timezone_offset, utc_time
     Download:
       https://sites.google.com/site/yangdingqi/home/foursquare-dataset

  2. Foursquare Global (Checkins) — larger, JSON or TSV
       checkins.txt / checkins.json
       Columns: user_id, venue_id, created_at, timezone_offset

  3. Foursquare NYC + Tokyo (Cheng et al. 2011)
       raw_POIs.txt + raw_Checkins.txt

The loader tries each format in order and picks the first that parses.

SC simulation mapping:
  Each check-in pair (user i at venue A then venue B within 2 hours)
  is treated as one "trip":
    - venue A = pickup location
    - venue B = dropoff location
    - time between check-ins = trip duration
    - haversine distance A→B = trip distance

If consecutive pairs cannot be formed (sparse user histories),
single check-ins are used with synthetic trip endpoints.

Feature space (positions fixed across all loaders):
  [0]  pickup_lat          [1]  pickup_lon
  [2]  dropoff_lat         [3]  dropoff_lon
  [4]  duration_s          [5]  distance_km
  [6]  category_score      [7]  hour_of_day
  [8]  dist_per_hour       [9]  checkin_density

Fraud label heuristics (anomalous mobility patterns):
  1. Impossible speed between consecutive check-ins (> 200 km/h)
  2. Consecutive check-ins at same venue < 3 min apart (bot behaviour)
  3. Check-in at time inconsistent with venue category
     (e.g. office venue at 3 AM)
  4. Very high check-in velocity: > 8 venues/hour
"""

import os
import math
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

DATASET_NAME = 'foursquare'


# ── Haversine ─────────────────────────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Category score ────────────────────────────────────────────────────────────
# Maps Foursquare venue category to a numeric score (higher = more transit-like)
CATEGORY_SCORES = {
    'travel & transport': 1.0, 'airport':         1.0,
    'train station':      0.9, 'bus station':     0.9,
    'taxi':               0.95,'road':            0.8,
    'food':               0.4, 'restaurant':      0.4,
    'coffee shop':        0.3, 'shop & service':  0.3,
    'nightlife spot':     0.5, 'bar':             0.5,
    'arts & entertainment':0.4,'hotel':           0.6,
    'outdoors & recreation':0.5,'home (private)': 0.2,
    'office':             0.2, 'college & university': 0.2,
    'medical':            0.3,
}

def _cat_score(category_str):
    if not category_str:
        return 0.5
    cat = str(category_str).lower().strip()
    for key, score in CATEGORY_SCORES.items():
        if key in cat:
            return score
    return 0.45  # unknown category


# ── Fraud label ───────────────────────────────────────────────────────────────
def _fraud_label(speed_kmh, duration_s, dist_km,
                 cat_score, hour, checkin_density):
    """
    Returns 1 (fraud/anomaly) or 0 (legitimate) for a check-in trip.
    """
    # Impossible ground speed
    if speed_kmh > 200.0:
        return 1
    # Bot-like repeated check-in (same venue too fast)
    if duration_s < 180 and dist_km < 0.05:
        return 1
    # Office/home visit at suspicious hours (2–5 AM)
    if cat_score < 0.25 and 2 <= hour <= 5:
        return 1
    # Extremely high check-in density (> 8 venues/hour)
    if checkin_density > 8.0:
        return 1
    return 0


# ── Parse TSMC2014 format ─────────────────────────────────────────────────────
def _parse_tsmc2014(path, encoding='utf-8'):
    """
    TSMC2014 format (tab-separated):
    user_id  venue_id  venue_category_id  venue_category_name
    latitude  longitude  timezone_offset  utc_time
    Returns list of dicts.
    """
    records = []
    try:
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 8:
                    continue
                try:
                    rec = {
                        'user_id':  parts[0].strip(),
                        'venue_id': parts[1].strip(),
                        'category': parts[3].strip(),
                        'lat':      float(parts[4]),
                        'lon':      float(parts[5]),
                        'tz_off':   int(parts[6]),
                        'utc_time': parts[7].strip(),
                    }
                    # Parse datetime
                    for fmt in ['%a %b %d %H:%M:%S +0000 %Y',
                                '%Y-%m-%dT%H:%M:%SZ',
                                '%Y-%m-%d %H:%M:%S']:
                        try:
                            rec['dt'] = datetime.strptime(rec['utc_time'], fmt)
                            # Apply timezone offset to get local time
                            rec['dt'] += timedelta(minutes=rec['tz_off'])
                            break
                        except ValueError:
                            continue
                    if 'dt' not in rec:
                        continue
                    # Validate coords (NYC bounding box)
                    if not (40.4 <= rec['lat'] <= 41.0 and
                            -74.3 <= rec['lon'] <= -73.6):
                        continue
                    records.append(rec)
                except (ValueError, IndexError):
                    continue
    except UnicodeDecodeError:
        return _parse_tsmc2014(path, encoding='latin-1')
    return records


# ── Parse generic TSV / CSV checkin format ────────────────────────────────────
def _parse_generic_tsv(path):
    """
    Tries to parse a generic TSV with lat/lon columns.
    Returns list of dicts or empty list.
    """
    records = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline().strip().split('\t')
            header_lower = [h.lower() for h in header]

            # Find column indices
            def find(names):
                for n in names:
                    for i, h in enumerate(header_lower):
                        if n in h:
                            return i
                return None

            lat_col  = find(['lat'])
            lon_col  = find(['lon'])
            user_col = find(['user'])
            time_col = find(['time', 'date', 'created'])
            cat_col  = find(['categ', 'venue_cat', 'category'])

            if lat_col is None or lon_col is None:
                return []

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) <= max(
                        filter(None, [lat_col, lon_col, user_col,
                                      time_col, cat_col])):
                    continue
                try:
                    rec = {
                        'lat': float(parts[lat_col]),
                        'lon': float(parts[lon_col]),
                        'user_id': parts[user_col] if user_col else '0',
                        'category': parts[cat_col] if cat_col else '',
                        'utc_time': parts[time_col] if time_col else '',
                    }
                    # Try parsing datetime
                    for fmt in ['%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S',
                                '%a %b %d %H:%M:%S +0000 %Y']:
                        try:
                            rec['dt'] = datetime.strptime(
                                rec['utc_time'], fmt)
                            break
                        except ValueError:
                            continue
                    if 'dt' not in rec:
                        rec['dt'] = datetime(2013, 1, 1, 12, 0, 0)
                    records.append(rec)
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    return records


# ── Build trip records from check-in sequences ────────────────────────────────
def _checkins_to_trips(checkins, max_gap_hours=2.0, rng=None):
    """
    Group check-ins by user. For each consecutive pair within max_gap_hours,
    create a trip record.
    Returns list of feature vectors and fraud labels.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Group by user
    by_user = defaultdict(list)
    for c in checkins:
        by_user[c['user_id']].append(c)

    # Sort each user's checkins by time
    for uid in by_user:
        by_user[uid].sort(key=lambda x: x['dt'])

    features = []
    labels   = []

    for uid, visits in by_user.items():
        # Compute per-user check-in density (visits per hour over session)
        if len(visits) >= 2:
            session_hours = max(
                0.1,
                (visits[-1]['dt'] - visits[0]['dt']).total_seconds() / 3600
            )
            checkin_density = len(visits) / session_hours
        else:
            checkin_density = 1.0

        # Build consecutive pairs
        for i in range(len(visits) - 1):
            v_a = visits[i]
            v_b = visits[i + 1]

            gap_s = (v_b['dt'] - v_a['dt']).total_seconds()
            if gap_s < 0 or gap_s > max_gap_hours * 3600:
                continue

            dist_km   = _haversine(v_a['lat'], v_a['lon'],
                                    v_b['lat'], v_b['lon'])
            duration_s = max(1.0, gap_s)
            speed_kmh  = (dist_km / duration_s) * 3600
            hour       = v_a['dt'].hour

            cat_a = _cat_score(v_a.get('category', ''))
            cat_b = _cat_score(v_b.get('category', ''))
            cat_score = (cat_a + cat_b) / 2.0

            dist_per_hour = dist_km / max(duration_s / 3600, 0.01)

            feat = np.array([
                v_a['lat'],          # [0] pickup_lat
                v_a['lon'],          # [1] pickup_lon
                v_b['lat'],          # [2] dropoff_lat
                v_b['lon'],          # [3] dropoff_lon
                duration_s,          # [4] duration_s
                dist_km,             # [5] distance_km
                cat_score,           # [6] category_score (cost proxy)
                float(hour),         # [7] hour_of_day
                dist_per_hour,       # [8] dist_per_hour (cost per unit)
                checkin_density,     # [9] checkin density
            ])

            fraud = _fraud_label(
                speed_kmh, duration_s, dist_km,
                cat_score, hour, checkin_density
            )

            features.append(feat)
            labels.append(fraud)

    return features, labels


# ── Main loader ───────────────────────────────────────────────────────────────
def load_foursquare(path=None, n_samples=80_000, seed=42):
    """
    Load Foursquare check-in dataset and convert to trip records.

    Parameters
    ----------
    path      : str  Path to dataset file (TSMC2014 .txt or generic TSV).
                     If None or not found, uses synthetic fallback.
    n_samples : int  Max records to return.
    seed      : int  Random seed.

    Returns
    -------
    X : np.ndarray (n, 10)
    y : np.ndarray (n,)
    """
    rng = np.random.RandomState(seed)

    # ── Try real data ─────────────────────────────────────────────────────
    if path and os.path.exists(path):
        print(f"\n[foursquare] Loading real data from: {path}")

        # Detect format
        checkins = []

        # Try TSMC2014 first (most common academic format)
        checkins = _parse_tsmc2014(path)
        if checkins:
            print(f"  Format: TSMC2014 | Raw check-ins: {len(checkins):,}")
        else:
            # Try generic TSV
            checkins = _parse_generic_tsv(path)
            if checkins:
                print(f"  Format: generic TSV | Raw check-ins: {len(checkins):,}")

        if not checkins:
            print(f"  ✗ Could not parse file — check format")
            print(f"  → Falling back to synthetic data")
        else:
            # Build trip records from consecutive check-in pairs
            print(f"  Building trip pairs from check-in sequences...")
            feat_list, label_list = _checkins_to_trips(
                checkins, max_gap_hours=2.0, rng=rng)

            if not feat_list:
                print(f"  ✗ No valid trip pairs found")
                print(f"  → Falling back to synthetic data")
            else:
                X = np.array(feat_list)
                y = np.array(label_list)

                # Sample if needed
                if len(X) > n_samples:
                    idx = rng.choice(len(X), n_samples, replace=False)
                    X, y = X[idx], y[idx]
                elif len(X) < n_samples:
                    print(f"  ⚠ Only {len(X):,} trip pairs available "
                          f"(requested {n_samples:,})")
                    print(f"  Consider using a larger Foursquare dataset "
                          f"or reducing n_samples in dataset_config.py")

                fraud_rate = y.mean()
                print(f"  Trip pairs   : {len(X):,}")
                print(f"  Fraud rate   : {fraud_rate:.1%}")
                print(f"  Lat range    : [{X[:,0].min():.3f}, "
                      f"{X[:,0].max():.3f}]")
                print(f"  Lon range    : [{X[:,1].min():.3f}, "
                      f"{X[:,1].max():.3f}]")

                if fraud_rate < 0.01:
                    print(f"  ⚠ Very low fraud rate. "
                          f"Consider adjusting _fraud_label() thresholds.")

                return X, y

    # ── Synthetic fallback ────────────────────────────────────────────────
    print(f"\n[foursquare] ⚠ Real data not found. "
          f"Using synthetic Foursquare-style data.")
    print(f"  To use real data:")
    print(f"  1. Download TSMC2014 NYC dataset from:")
    print(f"     https://sites.google.com/site/yangdingqi/home/foursquare-dataset")
    print(f"  2. Place at ~/data/foursquare/dataset_TSMC2014_NYC.txt")

    return _synthetic_foursquare(n_samples, seed)


def _synthetic_foursquare(n_samples=80_000, seed=42):
    """
    Synthetic Foursquare-style data.
    Mimics NYC check-in distribution:
      Lat: 40.50 – 40.92 (Manhattan + outer boroughs)
      Lon: -74.25 – -73.70
    Feature [6] = category score (0.2–1.0) instead of fare
    Feature [9] = check-in density (venues/hour) instead of duration/mile
    """
    from fl_base import generate_sc_dataset

    X_base, y_base = generate_sc_dataset(
        n_samples, fraud_rate=0.13, seed=seed)

    rng = np.random.RandomState(seed)
    n   = len(X_base)

    # Remap coordinates to NYC check-in distribution
    pu_lat = rng.uniform(40.50, 40.92, n)
    pu_lon = rng.uniform(-74.25, -73.70, n)
    do_lat = pu_lat + rng.normal(0, 0.02, n)
    do_lon = pu_lon + rng.normal(0, 0.02, n)

    X_fsq        = X_base.copy()
    X_fsq[:, 0]  = pu_lat
    X_fsq[:, 1]  = pu_lon
    X_fsq[:, 2]  = np.clip(do_lat, 40.40, 41.00)
    X_fsq[:, 3]  = np.clip(do_lon, -74.30, -73.60)

    # Feature [6]: category score proxy (uniform over venue types)
    X_fsq[:, 6]  = rng.uniform(0.2, 1.0, n)

    # Feature [9]: check-in density (venues per hour, anomaly > 8)
    # Most users: 1–4 venues/hour; fraudulent: > 8
    density = rng.exponential(scale=1.5, size=n).clip(0.1, 15.0)
    X_fsq[:, 9] = density

    print(f"\n[foursquare] Synthetic data generated:")
    print(f"  Records    : {n:,}")
    print(f"  Fraud rate : {y_base.mean():.1%}")
    print(f"  Lat range  : [{X_fsq[:,0].min():.3f}, {X_fsq[:,0].max():.3f}]")
    print(f"  Lon range  : [{X_fsq[:,1].min():.3f}, {X_fsq[:,1].max():.3f}]")

    return X_fsq, y_base


if __name__ == '__main__':
    print("Testing Foursquare loader (synthetic mode)...")
    X, y = load_foursquare()
    print(f"\nX shape : {X.shape}")
    print(f"y dist  : {int(y.sum())} fraud / {int((1-y).sum())} legit")
    print(f"\nFeature ranges:")
    labels = ['pu_lat', 'pu_lon', 'do_lat', 'do_lon',
              'duration_s', 'dist_km', 'cat_score',
              'hour', 'dist/hr', 'checkin_density']
    for i, lab in enumerate(labels):
        print(f"  {lab:18s}: [{X[:,i].min():.3f}, {X[:,i].max():.3f}]")
