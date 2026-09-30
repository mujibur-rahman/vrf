"""
sc_tasks.py
Task construction for the surge-pricing revision.

Implements Eq. (15) of the manuscript (NYC Taxi surge proxy) and the
demand-density target used for Yelp and Geolife, plus geographic client
partitioning and temporal train/test splitting.

ADAPTER
-------
Set COHEN_LOADER_MODULE to the module in your Cohen's Kappa project that
exposes the four dataset loaders. The only requirement is that each loader
returns a pandas DataFrame with, at minimum, the columns declared in
SCHEMA below. If your loaders already emit the 10-feature matrix, use
from_feature_matrix() instead and skip label construction.
"""

from __future__ import annotations

import importlib
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------
# ADAPTER CONFIGURATION  -- edit these three lines only
# ---------------------------------------------------------------------
COHEN_LOADER_MODULE = "datasets"          # e.g. "datasets" or "data.loaders"
COHEN_LOADER_FUNCS = {
    "nyc_taxi":   "load_nyc_taxi",
    "geolife":    "load_geolife",
    "yelp":       "load_yelp",
    "foursquare": "load_foursquare",
}
EPSILON = 1e-3                            # eps for the reference-median division

# Supply smoothing for the imbalance ratio of Eq. (15). Supply S is a *count*
# (drop-offs or vehicles over the lookback window), so its natural scale is
# O(1)-O(100). A sub-unit epsilon (1e-3) makes any bin with S=0 blow up to
# ~1000*D, producing a target with a max ~3e4 that swamps MSE training and
# leaves every aggregation rule indistinguishable. SUPPLY_EPS = 1 is
# add-one (Laplace) smoothing at the supply scale: an empty-supply bin becomes
# D/1 rather than D/1e-3. DISCLOSE in Section IV-C as the smoothing constant
# in the imbalance denominator.
SUPPLY_EPS = float(os.environ.get("SC_SUPPLY_EPS", "1.0"))

# Winsorisation of the normalised target. Even after supply smoothing the
# imbalance ratio has a heavy right tail (a few zone-bins with high demand and
# near-zero recent supply). Clipping at these per-frame percentiles removes the
# tail without discarding the surge signal. DISCLOSE the clip in Section IV-C.
WINSOR_LO = float(os.environ.get("SC_WINSOR_LO", "0.005"))
WINSOR_HI = float(os.environ.get("SC_WINSOR_HI", "0.995"))

SCHEMA = {
    "nyc_taxi":   ["pickup_datetime", "dropoff_datetime", "pu_zone", "do_zone",
                   "trip_distance", "trip_duration"],
    "geolife":    ["timestamp", "lat", "lon", "user_id"],
    "yelp":       ["timestamp", "lat", "lon", "business_id"],
    "foursquare": ["timestamp", "lat", "lon", "venue_id"],
}

BIN_MINUTES = 15
SUPPLY_LOOKBACK_BINS = 2
N_LAGS = 4


# ---------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------
@dataclass
class ClientData:
    client_id: int
    region_id: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    is_surge_zone: bool
    X_val: np.ndarray = None      # optional validation split (for HP selection)
    y_val: np.ndarray = None


@dataclass
class FederatedTask:
    name: str
    clients: List[ClientData]
    in_dim: int
    target_kind: str   # "surge_proxy" | "demand_density"

    def region_ids(self) -> List[str]:
        return [c.region_id for c in self.clients]


# ---------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------
def load_raw(dataset: str, **kwargs) -> pd.DataFrame:
    """Call the corresponding loader from the Cohen's Kappa project."""
    mod = importlib.import_module(COHEN_LOADER_MODULE)
    fn = getattr(mod, COHEN_LOADER_FUNCS[dataset])
    df = fn(**kwargs)
    missing = [c for c in SCHEMA[dataset] if c not in df.columns]
    if missing:
        raise KeyError(
            f"{dataset}: loader output is missing {missing}. "
            f"Either rename the columns or edit SCHEMA in sc_tasks.py."
        )
    return df


# ---------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------
def _time_bin(ts: pd.Series, minutes: int = BIN_MINUTES) -> pd.Series:
    return pd.to_datetime(ts).dt.floor(f"{minutes}min")


def _grid_region(lat: pd.Series, lon: pd.Series, n_cells: int = 24) -> pd.Series:
    """Equal-width lat/lon grid, used when a dataset has no zone column."""
    lat_b = pd.cut(lat, bins=n_cells, labels=False)
    lon_b = pd.cut(lon, bins=n_cells, labels=False)
    return lat_b.astype(str) + "_" + lon_b.astype(str)


def _add_calendar(df: pd.DataFrame, bincol: str) -> pd.DataFrame:
    t = pd.to_datetime(df[bincol])
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["hour_of_week"] = df["dow"] * 24 + df["hour"]
    return df


def _winsorize(s: pd.Series, lo: float = WINSOR_LO,
               hi: float = WINSOR_HI) -> pd.Series:
    """Clip a series to its [lo, hi] quantiles."""
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=ql, upper=qh)


def _add_lags(df: pd.DataFrame, region_col: str, target_col: str,
              n_lags: int = N_LAGS) -> pd.DataFrame:
    df = df.sort_values([region_col, "bin"])
    for k in range(1, n_lags + 1):
        df[f"lag_{k}"] = df.groupby(region_col)[target_col].shift(k)
    return df.dropna(subset=[f"lag_{k}" for k in range(1, n_lags + 1)])


# ---------------------------------------------------------------------
# NYC Taxi: surge proxy, Eq. (15)
# ---------------------------------------------------------------------
def build_nyc_surge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demand  D_b : pickups originating in the zone during bin b.
    Supply  S_b : distinct vehicles completing a drop-off in the zone over
                  the SUPPLY_LOOKBACK_BINS preceding bins. Where the loader
                  exposes no vehicle identifier, drop-off count is used and
                  the substitution is recorded on the returned frame.
    Target  y_b : imbalance normalised by the median over matching
                  (zone, hour-of-week) bins.
    """
    df = df.copy()
    df["bin"] = _time_bin(df["pickup_datetime"])
    df["do_bin"] = _time_bin(df["dropoff_datetime"])

    demand = (df.groupby(["pu_zone", "bin"])
                .size().rename("D").reset_index()
                .rename(columns={"pu_zone": "region"}))

    veh_col = next((c for c in ("vehicle_id", "hvfhs_license_num", "medallion")
                    if c in df.columns), None)
    if veh_col is None:
        supply = (df.groupby(["do_zone", "do_bin"])
                    .size().rename("S_raw").reset_index())
        supply_is_proxy = True
    else:
        supply = (df.groupby(["do_zone", "do_bin"])[veh_col]
                    .nunique().rename("S_raw").reset_index())
        supply_is_proxy = False
    supply = supply.rename(columns={"do_zone": "region", "do_bin": "bin"})

    # roll the supply signal forward over the lookback window
    supply = supply.sort_values(["region", "bin"])
    supply["S"] = (supply.groupby("region")["S_raw"]
                         .rolling(SUPPLY_LOOKBACK_BINS, min_periods=1)
                         .sum().reset_index(level=0, drop=True))

    m = demand.merge(supply[["region", "bin", "S"]], on=["region", "bin"],
                     how="left")
    m["S"] = m["S"].fillna(0.0)
    m["imbalance"] = m["D"] / (m["S"] + SUPPLY_EPS)      # Eq. (15), smoothed

    m = _add_calendar(m, "bin")
    ref = (m.groupby(["region", "hour_of_week"])["imbalance"]
             .median().rename("ref").reset_index())
    m = m.merge(ref, on=["region", "hour_of_week"], how="left")
    m["y"] = m["imbalance"] / (m["ref"] + EPSILON)
    m["y"] = _winsorize(m["y"])
    m["intensity"] = m["imbalance"]   # raw surge signal for the surge-zone flag

    # trip-level features aggregated to the bin
    agg = (df.groupby(["pu_zone", "bin"])
             .agg(mean_distance=("trip_distance", "mean"),
                  mean_duration=("trip_duration", "mean"))
             .reset_index().rename(columns={"pu_zone": "region"}))
    m = m.merge(agg, on=["region", "bin"], how="left")

    m = _add_lags(m, "region", "y")
    m.attrs["supply_is_dropoff_proxy"] = supply_is_proxy
    return m


# ---------------------------------------------------------------------
# Yelp / Geolife / Foursquare: demand-density target
# ---------------------------------------------------------------------
def build_density(df: pd.DataFrame, n_cells: int | None = None) -> pd.DataFrame:
    """
    No fare field exists in these datasets, so no surge target is derivable.
    The target is normalised activity density per (region, bin).

    n_cells sets the lat/lon grid resolution (n_cells x n_cells). Datasets that
    span many metros (e.g. Yelp) yield few populated cells at the default 24;
    raise SC_DENSITY_CELLS to split each metro into more client regions.
    """
    if n_cells is None:
        n_cells = int(os.environ.get("SC_DENSITY_CELLS", "24"))
    df = df.copy()
    df["bin"] = _time_bin(df["timestamp"])
    df["region"] = _grid_region(df["lat"], df["lon"], n_cells)

    m = df.groupby(["region", "bin"]).size().rename("count").reset_index()
    m = _add_calendar(m, "bin")

    ref = (m.groupby(["region", "hour_of_week"])["count"]
             .median().rename("ref").reset_index())
    m = m.merge(ref, on=["region", "hour_of_week"], how="left")
    m["y"] = m["count"] / (m["ref"] + EPSILON)
    m["y"] = _winsorize(m["y"])
    m["intensity"] = m["count"]   # raw demand signal for the high-demand flag
    m["mean_distance"] = 0.0
    m["mean_duration"] = 0.0
    m["D"] = m["count"]
    m["S"] = 0.0
    m = _add_lags(m, "region", "y")
    return m


# ---------------------------------------------------------------------
# Feature matrix, partitioning, split
# ---------------------------------------------------------------------
FEATURES = ["D", "S", "mean_distance", "mean_duration",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "lag_1", "lag_2", "lag_3", "lag_4"]


def _standardise(X: np.ndarray, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-8
    return (X - mu) / sd, mu, sd


def to_federated(m: pd.DataFrame, name: str, target_kind: str,
                 min_samples: Optional[int] = None, test_frac: float = 0.2,
                 val_frac: float = 0.0,
                 max_clients: Optional[int] = None) -> FederatedTask:
    """
    Clients correspond to spatial regions. The split is temporal: the last
    test_frac of bins per region is held out, so no client trains on data
    that postdates its own test window. If val_frac>0, an additional temporal
    validation block is carved immediately before the test block (order:
    train | val | test), for honest hyperparameter selection; the test block is
    identical whether or not val_frac is set.
    """
    if min_samples is None:
        min_samples = int(os.environ.get("SC_MIN_SAMPLES", "200"))
    m = m.dropna(subset=FEATURES + ["y"]).copy()
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["y"])

    sizes = m.groupby("region").size()
    keep = sizes[sizes >= min_samples].index.tolist()
    if max_clients:
        keep = sorted(keep, key=lambda r: -sizes[r])[:max_clients]
    m = m[m["region"].isin(keep)]

    Xall = m[FEATURES].to_numpy(np.float32)
    _, mu, sd = _standardise(Xall)

    # Surge / high-demand zone flag, robustly and comparatively. The flag keys
    # off the RAW intensity (imbalance for the surge proxy, activity count for
    # the density target), not the normalised target y: y is normalised per
    # (region, hour-of-week) so its per-zone median is ~1 for every zone and
    # cannot separate zones. A zone is flagged iff its median raw intensity
    # exceeds the cross-zone median of those medians -- a genuine ~50/50 split
    # of busier vs quieter zones. DISCLOSE the surge-zone definition in Sec IV.
    intensity_col = "intensity" if "intensity" in m.columns else "y"
    zone_intensity = m.groupby("region")[intensity_col].mean()
    surge_threshold = float(zone_intensity.median())

    clients: List[ClientData] = []
    for cid, (region, g) in enumerate(m.groupby("region")):
        g = g.sort_values("bin")
        n = len(g)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        c_train = n - n_test - n_val          # train | val | test (temporal)
        c_val = n - n_test
        X = ((g[FEATURES].to_numpy(np.float32) - mu) / sd).astype(np.float32)
        y = g["y"].to_numpy(np.float32).reshape(-1, 1)
        clients.append(ClientData(
            client_id=cid,
            region_id=str(region),
            X_train=X[:c_train], y_train=y[:c_train],
            X_test=X[c_val:],    y_test=y[c_val:],
            X_val=X[c_train:c_val], y_val=y[c_train:c_val],
            is_surge_zone=bool(zone_intensity[region] > surge_threshold),
        ))

    return FederatedTask(name=name, clients=clients,
                         in_dim=len(FEATURES), target_kind=target_kind)


def build_task(dataset: str, val_frac: float = 0.0, **loader_kwargs) -> FederatedTask:
    df = load_raw(dataset, **loader_kwargs)
    if dataset == "nyc_taxi":
        return to_federated(build_nyc_surge(df), dataset, "surge_proxy",
                            val_frac=val_frac)
    return to_federated(build_density(df), dataset, "demand_density",
                        val_frac=val_frac)


def from_feature_matrix(X: np.ndarray, y: np.ndarray, region: np.ndarray,
                        name: str, target_kind: str = "demand_density",
                        test_frac: float = 0.2) -> FederatedTask:
    """Bypass label construction when the Cohen loaders already emit features."""
    clients = []
    for cid, r in enumerate(np.unique(region)):
        idx = np.where(region == r)[0]
        cut = int(len(idx) * (1 - test_frac))
        clients.append(ClientData(
            client_id=cid, region_id=str(r),
            X_train=X[idx[:cut]], y_train=y[idx[:cut]].reshape(-1, 1),
            X_test=X[idx[cut:]],  y_test=y[idx[cut:]].reshape(-1, 1),
            is_surge_zone=bool(np.mean(y[idx]) > 1.0),
        ))
    return FederatedTask(name, clients, X.shape[1], target_kind)
