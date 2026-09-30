"""
yelp_loader.py
==============
Yelp Open Dataset loader.
Same interface as all other loaders:
  load_yelp(business_path, checkin_path, n_samples, seed) → (X, y)
  X: (n, 10) float array — shared feature space
  y: (n,)    binary fraud label

Dataset files needed (from Yelp Open Dataset):
  yelp_academic_dataset_business.json   — venue locations, categories, stars
  yelp_academic_dataset_checkin.json    — check-in timestamps per business

Download:
  https://business.yelp.com/data/resources/open-dataset/
  Extract JSON files to ~/data/yelp/

SC simulation mapping:
  Yelp has businesses (venues) and check-in counts per business,
  not individual user trips. We simulate SC trips as follows:

  For each business B with check-ins:
    - Treat B as a pickup/dropoff location
    - Pair B with a nearby business B' (within 5 km, same city)
      to form a synthetic trip: B → B'
    - Trip duration: estimated from distance + city average speed (30 km/h)
    - Trip cost proxy: stars * price_level (Yelp-specific signal)
    - Hour: sampled from business operating hours

  This mapping is explicitly documented in the paper as:
  "Yelp business locations and check-in patterns are used to
   simulate ride-hailing demand in urban areas, where high-rated
   venues attract pickup/dropoff activity and anomalous check-in
   patterns proxy for GPS spoofing and surge manipulation."

Feature space (positions fixed across all loaders):
  [0]  pickup_lat          [1]  pickup_lon
  [2]  dropoff_lat         [3]  dropoff_lon
  [4]  duration_s          [5]  distance_km
  [6]  rating_price_proxy  [7]  hour_of_day
  [8]  demand_density      [9]  review_velocity

Fraud label heuristics (Yelp-specific anomalies):
  1. Impossible inter-venue speed > 200 km/h (GPS spoof proxy)
  2. Zero-distance trip with high cost proxy (phantom fare)
  3. Very high demand density relative to review count
     (artificially inflated check-ins → surge manipulation proxy)
  4. Extreme review velocity: > 50 reviews/month for a business
     (bot review patterns → Sybil attack proxy)
"""

import os
import json
import math
import numpy as np
from collections import defaultdict

DATASET_NAME = 'yelp'


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


# ── Category → transit relevance score ───────────────────────────────────────
YELP_CATEGORY_SCORES = {
    'transportation': 1.0, 'taxis':          1.0,
    'airports':       1.0, 'hotels':         0.7,
    'restaurants':    0.5, 'food':           0.4,
    'bars':           0.5, 'nightlife':      0.5,
    'shopping':       0.3, 'health':         0.3,
    'beauty':         0.3, 'automotive':     0.6,
    'gas stations':   0.6, 'parking':        0.7,
    'active life':    0.4, 'arts':           0.4,
    'education':      0.2, 'financial':      0.2,
    'home services':  0.2, 'local services': 0.3,
}

def _category_score(categories_str):
    if not categories_str:
        return 0.4
    cats = str(categories_str).lower()
    best = 0.4
    for key, score in YELP_CATEGORY_SCORES.items():
        if key in cats:
            best = max(best, score)
    return best


# ── Price level → numeric ─────────────────────────────────────────────────────
def _price_level(price_str):
    mapping = {'$': 1.0, '$$': 2.0, '$$$': 3.0, '$$$$': 4.0}
    return mapping.get(str(price_str).strip(), 1.5)


# ── Parse business JSON ───────────────────────────────────────────────────────
def _parse_business(path):
    """
    Read yelp_academic_dataset_business.json.
    Returns list of business dicts with required fields.
    Handles both NDJSON (one JSON per line) and array format.
    """
    businesses = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                # Array format
                data = json.load(f)
                raw_list = data if isinstance(data, list) else []
            else:
                # NDJSON format (one object per line)
                raw_list = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        for b in raw_list:
            try:
                lat = float(b.get('latitude',  0))
                lon = float(b.get('longitude', 0))
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                if lat == 0.0 and lon == 0.0:
                    continue

                businesses.append({
                    'business_id':  b.get('business_id', ''),
                    'name':         b.get('name', ''),
                    'lat':          lat,
                    'lon':          lon,
                    'stars':        float(b.get('stars', 3.0)),
                    'review_count': int(b.get('review_count', 0)),
                    'categories':   b.get('categories', ''),
                    'price':        b.get('attributes', {}).get('RestaurantsPriceRange2', '$')
                                    if b.get('attributes') else '$',
                    'city':         b.get('city', ''),
                    'state':        b.get('state', ''),
                    'is_open':      int(b.get('is_open', 1)),
                })
            except (TypeError, ValueError):
                continue

    except Exception as e:
        print(f"  ✗ Business parse error: {e}")

    return businesses


# ── Parse checkin JSON ────────────────────────────────────────────────────────
def _parse_checkins(path):
    """
    Read yelp_academic_dataset_checkin.json.
    Returns dict: {business_id: total_checkin_count}
    Also returns {business_id: peak_hour} (most common check-in hour).
    """
    checkin_counts = {}
    peak_hours     = {}

    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                raw_list = json.load(f)
            else:
                raw_list = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        for c in raw_list:
            bid = c.get('business_id', '')
            if not bid:
                continue

            # 'date' field: comma-separated timestamp strings
            # e.g. "2016-04-26 19:49:16, 2016-08-30 15:09:55"
            date_str = c.get('date', '')
            timestamps = [d.strip() for d in date_str.split(',') if d.strip()]

            checkin_counts[bid] = len(timestamps)

            # Peak hour from timestamps
            hours = []
            for ts in timestamps[:100]:  # sample first 100 for speed
                try:
                    hours.append(int(ts.split(' ')[1].split(':')[0]))
                except (IndexError, ValueError):
                    pass
            if hours:
                from collections import Counter
                peak_hours[bid] = Counter(hours).most_common(1)[0][0]
            else:
                peak_hours[bid] = 12  # noon default

    except Exception as e:
        print(f"  ✗ Checkin parse error: {e}")

    return checkin_counts, peak_hours


# ── Build spatial index for nearby-business pairing ──────────────────────────
def _find_nearby(businesses, max_dist_km=5.0, n_pairs_per_biz=3, rng=None):
    """
    For each business, find up to n_pairs_per_biz nearby businesses
    within max_dist_km. Returns list of (biz_a, biz_b) pairs.
    Uses grid-based spatial bucketing for efficiency.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Group by city for faster search
    by_city = defaultdict(list)
    for b in businesses:
        by_city[b['city']].append(b)

    pairs = []
    for city, city_bizs in by_city.items():
        if len(city_bizs) < 2:
            continue

        # Sample pairs within city
        rng.shuffle(city_bizs)
        n = len(city_bizs)

        for i in range(min(n, 2000)):  # cap per city
            b_a = city_bizs[i]
            # Find nearby businesses
            candidates = []
            for j in range(n):
                if i == j:
                    continue
                d = _haversine(b_a['lat'], b_a['lon'],
                               city_bizs[j]['lat'], city_bizs[j]['lon'])
                if 0.05 <= d <= max_dist_km:
                    candidates.append((d, city_bizs[j]))

            if not candidates:
                continue

            # Sample up to n_pairs_per_biz
            rng.shuffle(candidates)
            for d, b_b in candidates[:n_pairs_per_biz]:
                pairs.append((b_a, b_b, d))

    rng.shuffle(pairs)
    return pairs


# ── Build feature vector for one trip pair ────────────────────────────────────
def _pair_to_record(b_a, b_b, dist_km,
                    checkin_counts, peak_hours):
    """Convert a (business_a, business_b, distance) pair to feature vector."""

    # Duration: distance / 30 km/h city average
    duration_s = max(60.0, (dist_km / 30.0) * 3600)

    # Cost proxy: stars × price_level (higher = more expensive ride area)
    price_a = _price_level(b_a.get('price', '$'))
    price_b = _price_level(b_b.get('price', '$'))
    rating_price = ((b_a['stars'] + b_b['stars']) / 2) * \
                   ((price_a + price_b) / 2)

    # Hour: peak check-in hour of pickup venue
    hour = float(peak_hours.get(b_a['business_id'], 12))

    # Demand density: check-ins per unit distance
    checkins_a = checkin_counts.get(b_a['business_id'], 1)
    checkins_b = checkin_counts.get(b_b['business_id'], 1)
    demand_density = (checkins_a + checkins_b) / max(dist_km, 0.1)

    # Review velocity: reviews per month (proxy for bot activity)
    # Yelp dataset spans ~10 years → estimate monthly rate
    review_vel_a = b_a['review_count'] / 120.0  # 10 years = 120 months
    review_vel_b = b_b['review_count'] / 120.0
    review_velocity = (review_vel_a + review_vel_b) / 2

    # Speed (for fraud check)
    speed_kmh = (dist_km / duration_s) * 3600

    feat = np.array([
        b_a['lat'],        # [0] pickup_lat
        b_a['lon'],        # [1] pickup_lon
        b_b['lat'],        # [2] dropoff_lat
        b_b['lon'],        # [3] dropoff_lon
        duration_s,        # [4] duration_s
        dist_km,           # [5] distance_km
        rating_price,      # [6] rating × price (cost proxy)
        hour,              # [7] hour_of_day
        demand_density,    # [8] demand density
        review_velocity,   # [9] review velocity
    ])

    # Fraud label
    fraud = 0
    if speed_kmh > 200.0:
        fraud = 1
    elif dist_km < 0.05 and rating_price > 6.0:   # phantom high-cost trip
        fraud = 1
    elif demand_density > 500 and dist_km < 0.2:   # surge manipulation
        fraud = 1
    elif review_velocity > 50.0:                   # Sybil reviews → bot
        fraud = 1

    return feat, fraud


# ── Main loader ───────────────────────────────────────────────────────────────
def load_yelp(business_path=None, checkin_path=None,
              n_samples=80_000, seed=42):
    """
    Load Yelp Open Dataset and convert to SC trip records.

    Parameters
    ----------
    business_path : str  Path to yelp_academic_dataset_business.json
    checkin_path  : str  Path to yelp_academic_dataset_checkin.json
    n_samples     : int  Max records to return.
    seed          : int  Random seed.

    Returns
    -------
    X : np.ndarray (n, 10)
    y : np.ndarray (n,)
    """
    rng = np.random.RandomState(seed)

    biz_found     = business_path and os.path.exists(business_path)
    checkin_found = checkin_path  and os.path.exists(checkin_path)

    if biz_found:
        print(f"\n[yelp] Loading business data from: {business_path}")
        businesses = _parse_business(business_path)
        print(f"  Businesses parsed: {len(businesses):,}")

        if not businesses:
            print(f"  ✗ No businesses parsed — check file format")
            return _synthetic_yelp(n_samples, seed)

        # Load check-in data if available
        checkin_counts = {}
        peak_hours     = {}
        if checkin_found:
            print(f"  Loading check-in data from: {checkin_path}")
            checkin_counts, peak_hours = _parse_checkins(checkin_path)
            print(f"  Businesses with check-ins: {len(checkin_counts):,}")
        else:
            print(f"  ⚠ No check-in file — using review counts as proxy")
            # Fallback: use review_count as check-in proxy
            for b in businesses:
                checkin_counts[b['business_id']] = b['review_count']
                peak_hours[b['business_id']]     = 12

        # Build trip pairs
        print(f"  Building trip pairs from nearby business matches...")
        pairs = _find_nearby(businesses, max_dist_km=5.0,
                              n_pairs_per_biz=3, rng=rng)
        print(f"  Trip pairs found: {len(pairs):,}")

        if not pairs:
            print(f"  ✗ No valid pairs — falling back to synthetic")
            return _synthetic_yelp(n_samples, seed)

        # Convert pairs to feature vectors
        feat_list, label_list = [], []
        for b_a, b_b, dist in pairs:
            feat, fraud = _pair_to_record(
                b_a, b_b, dist, checkin_counts, peak_hours)
            feat_list.append(feat)
            label_list.append(fraud)

        X = np.array(feat_list)
        y = np.array(label_list)

        # Sample to n_samples
        if len(X) > n_samples:
            idx = rng.choice(len(X), n_samples, replace=False)
            X, y = X[idx], y[idx]
        elif len(X) < n_samples:
            print(f"  ⚠ Only {len(X):,} trip pairs available "
                  f"(requested {n_samples:,})")
            print(f"  Increase n_pairs_per_biz or max_dist_km if needed")

        fraud_rate = y.mean()
        print(f"  Final records: {len(X):,}")
        print(f"  Fraud rate   : {fraud_rate:.1%}")
        print(f"  Lat range    : [{X[:,0].min():.3f}, {X[:,0].max():.3f}]")
        print(f"  Lon range    : [{X[:,1].min():.3f}, {X[:,1].max():.3f}]")

        if fraud_rate < 0.01:
            print(f"  ⚠ Very low fraud rate. "
                  f"Adjust _pair_to_record() thresholds.")
        if fraud_rate > 0.50:
            print(f"  ⚠ Very high fraud rate. "
                  f"Review velocity threshold may be too low.")

        return X, y

    # ── Synthetic fallback ────────────────────────────────────────────────
    print(f"\n[yelp] ⚠ Real data not found. "
          f"Using synthetic Yelp-style data.")
    print(f"  To use real data:")
    print(f"  1. Download from: "
          f"https://business.yelp.com/data/resources/open-dataset/")
    print(f"  2. Extract JSON files to ~/data/yelp/")
    print(f"  3. Files needed:")
    print(f"     ~/data/yelp/yelp_academic_dataset_business.json")
    print(f"     ~/data/yelp/yelp_academic_dataset_checkin.json")

    return _synthetic_yelp(n_samples, seed)


def _synthetic_yelp(n_samples=80_000, seed=42):
    """
    Synthetic Yelp-style data.
    Geographic distribution matches Yelp's primary US cities:
      Las Vegas   (36.17, -115.14) — largest Yelp dataset city
      Phoenix     (33.45, -112.07)
      Toronto     (43.65, -79.38)
      Charlotte   (35.23, -80.84)
      Pittsburgh  (40.44, -79.99)
    Sampled proportionally across cities.
    """
    from fl_base import generate_sc_dataset

    X_base, y_base = generate_sc_dataset(
        n_samples, fraud_rate=0.13, seed=seed)

    rng = np.random.RandomState(seed)
    n   = len(X_base)

    # City centroids with weights (Las Vegas is dominant in Yelp dataset)
    cities = [
        (36.17, -115.14, 0.35),  # Las Vegas
        (33.45, -112.07, 0.20),  # Phoenix
        (43.65,  -79.38, 0.15),  # Toronto
        (35.23,  -80.84, 0.15),  # Charlotte
        (40.44,  -79.99, 0.15),  # Pittsburgh
    ]

    # Assign each record to a city
    city_probs = [c[2] for c in cities]
    city_idx   = rng.choice(len(cities), size=n, p=city_probs)

    pu_lat = np.array([cities[i][0] for i in city_idx]) + \
             rng.normal(0, 0.05, n)
    pu_lon = np.array([cities[i][1] for i in city_idx]) + \
             rng.normal(0, 0.05, n)
    do_lat = pu_lat + rng.normal(0, 0.02, n)
    do_lon = pu_lon + rng.normal(0, 0.02, n)

    X_yelp       = X_base.copy()
    X_yelp[:, 0] = pu_lat
    X_yelp[:, 1] = pu_lon
    X_yelp[:, 2] = do_lat
    X_yelp[:, 3] = do_lon

    # Feature [6]: rating × price proxy (1.0–20.0)
    stars = rng.uniform(1.0, 5.0, n)
    price = rng.choice([1, 2, 3, 4], size=n,
                       p=[0.40, 0.35, 0.18, 0.07]).astype(float)
    X_yelp[:, 6] = stars * price

    # Feature [8]: demand density proxy
    X_yelp[:, 8] = rng.exponential(scale=50, size=n).clip(1, 800)

    # Feature [9]: review velocity (reviews/month)
    X_yelp[:, 9] = rng.exponential(scale=8, size=n).clip(0.1, 200)

    print(f"\n[yelp] Synthetic data generated:")
    print(f"  Records    : {n:,}")
    print(f"  Fraud rate : {y_base.mean():.1%}")

    city_names = ['Las Vegas', 'Phoenix', 'Toronto', 'Charlotte', 'Pittsburgh']
    for i, (name, count) in enumerate(zip(
            city_names,
            np.bincount(city_idx, minlength=len(cities)))):
        print(f"  {name:12s}: {count:,} records")

    return X_yelp, y_base


if __name__ == '__main__':
    print("Testing Yelp loader (synthetic mode)...")
    X, y = load_yelp()
    print(f"\nX shape : {X.shape}")
    print(f"y dist  : {int(y.sum())} fraud / {int((1-y).sum())} legit")
    print(f"\nFeature ranges:")
    labels = ['pu_lat', 'pu_lon', 'do_lat', 'do_lon',
              'duration_s', 'dist_km', 'rating×price',
              'hour', 'demand_density', 'review_velocity']
    for i, lab in enumerate(labels):
        print(f"  {lab:18s}: [{X[:,i].min():.3f}, {X[:,i].max():.3f}]")
