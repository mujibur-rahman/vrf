"""
dataset_config.py
=================
Central configuration for all four datasets.
All loaders follow the same interface:
  load_X(path, n_samples, seed) → (X, y)
  X: (n, 10) float array — same feature space across all datasets
  y: (n,)    binary label (0=legitimate, 1=fraudulent)

Feature space (consistent across all 4 datasets):
  [0]  pickup_lat     [1]  pickup_lon
  [2]  dropoff_lat    [3]  dropoff_lon
  [4]  duration_s     [5]  distance (km or miles depending on dataset)
  [6]  cost_proxy     [7]  hour_of_day
  [8]  cost_per_unit  [9]  duration_per_unit

Synthetic fallback: every loader falls back to generate_sc_dataset()
from fl_base.py if the real data file is not found. The feature space
and fraud rate are matched to the real dataset's distribution.

Usage:
  DATASET=nyc-taxi  python3 experiment1_baselines.py
  DATASET=geolife   python3 experiment1_baselines.py
  python3 experiment1_baselines.py --dataset foursquare
  python3 experiment1_baselines.py --dataset yelp
"""

import os
import sys
import numpy as np

# ── Parse dataset name ────────────────────────────────────────────────────────
def _parse_dataset_name():
    for i, arg in enumerate(sys.argv):
        if arg == '--dataset' and i + 1 < len(sys.argv):
            return sys.argv[i + 1].lower().strip()
        if arg.startswith('--dataset='):
            return arg.split('=', 1)[1].lower().strip()
    return os.environ.get('DATASET', 'nyc-taxi').lower().strip()

DATASET_NAME = _parse_dataset_name()


# ── Parse optional real-data override (--data <path> or DATA_FILE env) ────────
def _parse_data_file():
    for i, arg in enumerate(sys.argv):
        if arg == '--data' and i + 1 < len(sys.argv):
            return sys.argv[i + 1].strip()
        if arg.startswith('--data='):
            return arg.split('=', 1)[1].strip()
    return os.environ.get('DATA_FILE', '').strip() or None

DATA_FILE = _parse_data_file()

# ── Dataset registry ──────────────────────────────────────────────────────────
# Edit paths to match your local download locations.
DATASET_REGISTRY = {

    'nyc-taxi': {
        'description':  'NYC TLC High-Volume FHV trips (2025)',
        'real_path':    os.path.expanduser(
                            '~/data/nyc-taxi/fhvhv_tripdata_2025-01.parquet'),
        #'real_path':     os.path.expanduser(
        #                    './data/nyc-taxi/yellow_tripdata_2015-01.csv'),
        'aux_path':     os.path.expanduser(
                            '~/data/nyc-taxi/yellow_tripdata_2015-01.csv'),
                            #'~/data/nyc-taxi/taxi_zone_lookup.csv'),
        'n_samples':    100_000,
        'download_url': 'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page',
        'fraud_rate':   0.15,   # expected synthetic fraud rate
    },

    'geolife': {
        'description':  'Microsoft Geolife GPS Trajectories (Zheng et al. 2008)',
        'real_path':    os.path.expanduser('./data/geolife/Data/'),
        'aux_path':     None,
        'n_samples':    80_000,
        'download_url': ('https://www.microsoft.com/en-us/download/'
                         'details.aspx?id=52367'),
        'fraud_rate':   0.12,
    },

    'foursquare': {
        'description':  'Foursquare NYC check-in dataset (Yang et al. 2014)',
        'real_path':    os.path.expanduser(
                            './data/foursquare/dataset_TSMC2014_NYC.txt'),
        'aux_path':     None,
        'n_samples':    80_000,
        'download_url': ('https://sites.google.com/site/yangdingqi/home/'
                         'foursquare-dataset'),
        'fraud_rate':   0.13,
    },

    'yelp': {
        'description':  'Yelp Open Dataset — business locations and check-ins',
        'real_path':    os.path.expanduser(
                            './data/yelp/yelp_academic_dataset_business.json'),
        'aux_path':     os.path.expanduser(
                            '~/data/yelp/yelp_academic_dataset_checkin.json'),
        'n_samples':    80_000,
        'download_url': 'https://business.yelp.com/data/resources/open-dataset/',
        'fraud_rate':   0.13,
    },
}


# ── Loader dispatch ───────────────────────────────────────────────────────────
def get_dataset(seed=42):
    """
    Load the configured dataset.
    Automatically falls back to synthetic data if real files not found.
    Returns (X, y) in the shared 10-feature space.
    """
    if DATASET_NAME not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{DATASET_NAME}'. "
            f"Choose from: {list(DATASET_REGISTRY.keys())}"
        )

    cfg = DATASET_REGISTRY[DATASET_NAME]

    # ── Real-data override: load a prepped npz/csv (from prep_*.py) ────────────
    # Set via --data <path> or DATA_FILE env; applies to ALL experiments so the
    # full suite runs on real data without editing each script.
    if DATA_FILE:
        import pandas as pd
        if DATA_FILE.endswith('.npz'):
            d = np.load(DATA_FILE); X, y = d['X'].astype(float), d['y'].astype(float)
        else:
            df = pd.read_csv(DATA_FILE)
            if 'label' in df.columns:
                y = df['label'].values.astype(float)
                X = df.drop(columns=['label']).values.astype(float)
            else:
                y = df.iloc[:, -1].values.astype(float)
                X = df.iloc[:, :-1].values.astype(float)
        print(f"\n{'='*60}")
        print(f"  Dataset : {DATASET_NAME}  (real data override)")
        print(f"  Source  : {DATA_FILE}")
        print(f"  Samples : {len(y):,}   fraud_rate: {y.mean():.1%}")
        print(f"{'='*60}")
        return X, y

    print(f"\n{'='*60}")
    print(f"  Dataset : {DATASET_NAME}")
    print(f"  Desc    : {cfg['description']}")
    print(f"  Samples : {cfg['n_samples']:,}")
    print(f"{'='*60}")

    if DATASET_NAME == 'nyc-taxi':
        from nyc_taxi_loader import load_nyc_taxi
        return load_nyc_taxi(
            parquet_path=cfg['real_path'],
            zone_lookup_path=cfg['aux_path'],
            n_samples=cfg['n_samples'],
            seed=seed,
        )

    elif DATASET_NAME == 'geolife':
        from geolife_loader import load_geolife
        return load_geolife(
            data_dir=cfg['real_path'],
            n_samples=cfg['n_samples'],
            seed=seed,
        )

    elif DATASET_NAME == 'foursquare':
        from foursquare_loader import load_foursquare
        return load_foursquare(
            path=cfg['real_path'],
            n_samples=cfg['n_samples'],
            seed=seed,
        )

    elif DATASET_NAME == 'yelp':
        from yelp_loader import load_yelp
        return load_yelp(
            business_path=cfg['real_path'],
            checkin_path=cfg['aux_path'],
            n_samples=cfg['n_samples'],
            seed=seed,
        )


# ── Output path utilities ─────────────────────────────────────────────────────
def results_dir():
    """Dataset-specific results directory. Created on first call."""
    d = os.path.join('results', DATASET_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def tag(filename):
    """
    Prefix filename with dataset name and place in results subdir.
    Example:
      tag('table_baselines.csv')
      → 'results/nyc-taxi/nyc-taxi_table_baselines.csv'
    """
    return os.path.join(results_dir(), f"{DATASET_NAME}_{filename}")


# ── Info utility ──────────────────────────────────────────────────────────────
def print_registry():
    print("\nRegistered datasets:")
    for name, cfg in DATASET_REGISTRY.items():
        print('{cfg}', cfg)
        real = cfg['real_path']
        found = (os.path.exists(real)
                 if not real.endswith('/') else os.path.isdir(real))
        status = "✓ found" if found else "✗ not found (will use synthetic)"
        print(f"  {name:15s}  {status}")
        if not found:
            print(f"  {'':15s}  Download: {cfg['download_url']}")


if __name__ == '__main__':
    print(f"Active dataset : {DATASET_NAME}")
    print(f"Results dir    : {results_dir()}")
    print(f"Sample output  : {tag('table_baselines.csv')}")
    print_registry()
