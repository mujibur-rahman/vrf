# vrf
Robust Federated Client Selection for Fair Surge Pricing in Spatial Crowdsourcing
# VRF-Selected Robust Federated Learning for Surge Pricing

Federated learning pipeline for a spatio-temporal **surge/demand** regression task,
with **VRF-bounded committee selection** (Proposition 1) plus a **trust-weighted,
norm-clipped aggregation** rule, evaluated against data / model / collusion
poisoning and baseline robust aggregators (FedAvg, Trimmed-Mean, Krum, FLTrust,
FoolsGold).


## 1. Requirements

Python 3.14, with: `numpy`, `pandas`, `pyarrow`, `torch` (CPU is fine),
`matplotlib`, `pynacl` (for the selection-proof overhead numbers).

```bash
pip install numpy pandas pyarrow torch matplotlib pynacl
```

## 2. Data setup

The pipeline reads **raw records** (not the collapsed fraud matrices in `data/*.csv`).
`datasets.py` is the adapter that `sc_tasks.py` calls.

| dataset | source file(s) | notes |
|---|---|---|
| `nyc_taxi` | TLC FHVHV parquet | set `NYC_PARQUET` (default `~/Downloads/fhvhv_tripdata_2025-01.parquet`) |
| `foursquare` | `data/foursquare/dataset_TSMC2014_NYC.txt` | check-ins (real timestamps) |
| `yelp` | `data/yelp/yelp_academic_dataset_{business,checkin}.json` | check-ins joined to business coords |
| `geolife` | `data/geolife/Data/<user>/Trajectory/*.plt` | GPS points, thinned by stride |

`nyc_taxi` = surge proxy (Eq. 15); the others = normalized demand-density.


## 3. Repository layout

**Core pipeline**
- `datasets.py` — adapter: raw records → DataFrames for `sc_tasks`.
- `sc_tasks.py` — task construction (surge/density target, region clients, temporal train/val/test split).
- `sc_attacks.py` — data / model / collusion / `model_stealth` (norm-matched) attacks.
- `sc_defenses.py` — FedAvg, Trimmed-Mean, Krum, FLTrust, FoolsGold (norm-clipped), `ours` (TrustWeighted + clip), ablation variants `ours_no_clip`, `ours_no_magnitude`.
- `sc_selection.py` — uniform / VRF (HMAC-SHA256 PRF) / biased selection; Ed25519 proof-cost.
- `sc_metrics.py` — MAE, fairness (per-region MAE std), influence (Eq. 13), trust separation, `gap_over_se`.
- `sc_experiments.py` — serial runner; blocks `overhead|selection|grid|phase`; `run_fl` core.

**Runners / analysis** (all parallel, build task once in parent)
- `sc_run_parallel.py` — parallel + **resumable** grid/selection/phase; `--methods`, `--attacks`.
- `sc_report.py` — per-dataset LaTeX tables.
- `sc_combined_table.py` — one combined comparison table across datasets.
- `sc_plots.py` — combined figures (selection, breakdown, attacks, influence, cross-dataset).
- `sc_hpsweep.py` — validation/test hyperparameter selection (ours + baselines' knobs).
- `sc_ablation.py` — 2×2 selection×aggregation ablation.
- `sc_tune.py` — clip/psi/lam variant sweep on the weak cells.

**Legacy (Cohen fraud project, not on the surge path):** `*_loader.py`, `dataset_config.py`.

## 4. Quickstart (canonical pipeline, parallel)

```bash
# 1. selection-proof overhead (needs pynacl)
python sc_experiments.py overhead

# 2. main blocks, all 4 datasets, 5 seeds (uses all cores; resumable)
python sc_run_parallel.py selection --datasets nyc_taxi foursquare geolife yelp --seeds 5 --jobs 24
python sc_run_parallel.py grid      --datasets nyc_taxi foursquare geolife yelp --seeds 5 --jobs 24 --resume
python sc_run_parallel.py phase     --datasets nyc_taxi foursquare geolife yelp --seeds 5 --jobs 24 --resume

# 3. tables + figures + combined table
for d in nyc_taxi foursquare geolife yelp; do python sc_report.py --dataset $d --out tables_$d.tex; done
python sc_plots.py --dataset nyc_taxi        # emits all combined figures under figures/
python sc_combined_table.py --out table_combined.tex

# 4. hyperparameter sweep and ablation
python sc_hpsweep.py  --days 7 --val-seeds 3 --test-seeds 5 --jobs 24
python sc_ablation.py --seeds 5 --jobs 20
```

Serial equivalent for any block: `python sc_experiments.py <block> --datasets nyc_taxi --seeds 5`.

## 5. Environment knobs

| var | controls | default |
|---|---|---|
| `NYC_PARQUET` | NYC parquet path | `~/Downloads/fhvhv_tripdata_2025-01.parquet` |
| `NYC_DAYS` | NYC time-window (days) | 7 |
| `NYC_MAX_ROWS` | NYC row cap (0 = none) | 0 |
| `FOURSQUARE_CITY` | NYC / TKY | NYC |
| `YELP_MAX` | check-in cap | 300000 |
| `GEOLIFE_STRIDE` / `GEOLIFE_MAX` | point thinning / cap | 40 / 300000 |
| `SC_DENSITY_CELLS` | lat/lon grid resolution | 24 |
| `SC_MIN_SAMPLES` | min time-bins per client region | 200 |
| `SC_SUPPLY_EPS`, `SC_WINSOR_LO/HI` | Eq. 15 smoothing / target clip | 1.0 / 0.005 / 0.995 |
| `SC_RESULTS_DIR` | output folder (**isolate subset runs!**) | `results` |

**Small runs:** the surge task needs ≥`SC_MIN_SAMPLES` bins/region — ~100 records → 0 clients.
Use `NYC_MAX_ROWS`/`NYC_DAYS` (and lower `SC_MIN_SAMPLES` for tiny), always with a
separate `SC_RESULTS_DIR`, e.g.:
```bash
SC_RESULTS_DIR=results_tiny NYC_DAYS=2 NYC_MAX_ROWS=30000 SC_MIN_SAMPLES=50 \
  python sc_run_parallel.py grid --datasets nyc_taxi --seeds 3 --jobs 24
```

## 6. Outputs

- `results/` — full run (grid 312 incl. stealth, phase 528, selection 12, overhead 1). `results_small/`, `results_tiny/` are subset runs.
- `figures/` — `fig_selection_all`, `fig_breakdown_all`, `fig_attacks_all`, `fig_influence_all`, `fig_crossdataset_model` (PDF + PNG).
- `tables_*.tex`, `table_combined*.tex`, `table_ablation.tex`, `ablation_section.tex`.
- `hpsweep_nyc.json`, `ablation_nyc.json`.

## 7. Hyperparameters

`psi=0.7` (angular vs magnitude mix in the deviation score), `lam=0.8` (EMA trust),
`tau=0.2` (committee fraction = α in Prop 1), `clip=True`, 40 rounds, 2 local epochs,
Adam 1e-3. The method is **insensitive to `psi`/`lam`** (val MAE flat), so defaults
are kept; FLTrust needs `root_size` tuned (200→400) to reach parity.

## 8. Key results (honest)

- **Proposition 1:** VRF β̄≈0.20 = uniform ≈ α; biasing coordinator → β=1, MAE 10⁶–10¹¹. Holds on all 4 datasets.
- **Under attack (model/collusion @20%):** ours is best-or-statistically-tied everywhere, significantly beats Krum broadly and FLTrust on NYC model poisoning, and never diverges. FedAvg and FoolsGold diverge under model poisoning.
- **No edge on benign / data poisoning** (geometric score can't see label inflation).
- **Ablation:** components are *necessary and complementary* — without VRF a biased committee explodes MAE for both mean and trust; without trust, mean diverges under model poisoning even at bounded β; only the full method is robust.

## 9. Required paper disclosures (Section IV) — do not omit

1. **VRF** is an HMAC-SHA256 PRF (reproduces the selection distribution, no public verifiability); proof cost is an **Ed25519 stand-in**.
2. **Supply** in Eq. 15 is a **drop-off count proxy** (FHVHV has no vehicle id); check `df.attrs["supply_is_dropoff_proxy"]`.
3. Target uses **add-one supply smoothing** and **[0.5%,99.5%] winsorization**; surge zones = mean raw intensity above cross-zone median.
4. Density datasets (Foursquare/Yelp/Geolife) target = normalized activity count; Yelp is coarse (~14 regions at default grid).
5. **FoolsGold** runs with added **median-norm clipping** (else it diverges under model poisoning); disclose.
6. `phase` forces β directly (uses `ratio=0.5` internally to size the malicious pool — **not** an adversarial ratio).

## 10. Known gotchas

- **Subset runs overwrite `results/`** unless you set `SC_RESULTS_DIR` — always isolate.
- Parallel runners build the task **once in the parent** (a prior bug had every worker re-read the 816 MB parquet → pyarrow crash → silent empty output). Keep it that way.
- Small/tiny tasks are **noisy** (higher MAE, few seeds); use for mechanism/HP insight, **report on `NYC_DAYS=7`**.
- **Stealth attack unresolved risk:** `model_stealth` at 20% is a *weak* attack (mean survives); the meaningful test is a **stealth `phase` (β) sweep**, not yet run — a reviewer will ask.

