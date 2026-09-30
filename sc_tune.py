"""
sc_tune.py  --  honest hyperparameter/ablation sweep for the proposed rule.
Compares old ours (no clip) vs clipped vs clipped+tuned on the weak cells.
Deterministic per seed; parallel across configs.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from concurrent.futures import ProcessPoolExecutor

VARIANTS = {
    "ours_old(noclip,psi.7)":   dict(clip=False, psi=0.7, lam=0.8),
    "clip(psi.7)":              dict(clip=True,  psi=0.7, lam=0.8),
    "clip,psi.9":               dict(clip=True,  psi=0.9, lam=0.8),
    "clip,psi.9,lam.9":         dict(clip=True,  psi=0.9, lam=0.9),
    "clip,psi.95,lam.9":        dict(clip=True,  psi=0.95, lam=0.9),
}
DATASETS = ["nyc_taxi", "foursquare", "geolife"]
ATTACKS = ["model", "collusion"]
SEEDS = 3
_T = {}


def _init(dsk):
    import torch; torch.set_num_threads(1)
    import sc_tasks as T
    for n, kw in dsk.items():
        _T[n] = T.build_task(n, **kw)


def _job(args):
    ds, vname, cfg, attack = args
    import sc_experiments as E
    task = _T[ds]
    maes = [E.run_fl(task, "ours", attack, 0.2, s, cfg=cfg)["mae"]
            for s in range(SEEDS)]
    return (ds, attack, vname, float(np.mean(maes)), float(np.std(maes)))


def main():
    jobs = [(ds, v, cfg, a) for ds in DATASETS for v, cfg in VARIANTS.items()
            for a in ATTACKS]
    dsk = {ds: {} for ds in DATASETS}
    res = {}
    with ProcessPoolExecutor(max_workers=min(24, os.cpu_count()),
                             initializer=_init, initargs=(dsk,)) as ex:
        for ds, attack, vname, mu, sd in ex.map(_job, jobs):
            res[(ds, attack, vname)] = (mu, sd)

    for ds in DATASETS:
        print(f"\n=== {ds} (ours variants, MAE mean+/-std, 3 seeds) ===")
        for a in ATTACKS:
            print(f"  [{a}]")
            for v in VARIANTS:
                mu, sd = res[(ds, a, v)]
                print(f"    {v:24s} {mu:8.3f} +/- {sd:.3f}")


if __name__ == "__main__":
    main()
