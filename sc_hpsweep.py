"""
sc_hpsweep.py
Honest hyperparameter selection for the proposed rule (and fair tuning of the
baselines' knobs), with a strict validation/test separation.

Protocol:
  * build the NYC surge task with a temporal train | val | test split (val_frac);
  * select each method's hyperparameters by VALIDATION MAE, averaged over the
    model + collusion attacks at 20% and `val_seeds` seeds;
  * report TEST MAE at the selected setting (and at the default) over
    `test_seeds` seeds. The test block is never seen during selection.

Tuned knobs: ours -> (psi, lam); trimmed_mean -> trim; fltrust -> root_size.
Krum/FoolsGold/FedAvg have no accuracy hyperparameter here (reported at default).

  python sc_hpsweep.py --days 7 --val-seeds 3 --test-seeds 5 --jobs 24
  python sc_hpsweep.py --days 3 --val-seeds 2 --test-seeds 3 --jobs 24   # pre-screen
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ATTACKS = ["model", "collusion"]
RATIO = 0.20

# candidate grids
OURS_PSI = [0.5, 0.7, 0.85, 0.95]
OURS_LAM = [0.7, 0.85, 0.95]
TRIM = [0.1, 0.2, 0.3]
ROOT = [100, 200, 400]

_TASK = None


def _init(task):
    import torch
    torch.set_num_threads(1)
    global _TASK
    _TASK = task


def _run(job):
    """job = (method, cfg, attack, seed, split) -> mae (inf if non-finite)."""
    import sc_experiments as E
    method, cfg, attack, seed, split = job
    r = E.run_fl(_TASK, method, attack, RATIO, seed, cfg=cfg, eval_split=split)
    m = r["mae"]
    return m if np.isfinite(m) else float("inf")


def _mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-seeds", type=int, default=3)
    ap.add_argument("--test-seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default="hpsweep_nyc.json")
    args = ap.parse_args()

    os.environ["NYC_DAYS"] = str(args.days)
    import sc_tasks as T
    task = T.build_task("nyc_taxi", val_frac=args.val_frac)
    print(f"task: {len(task.clients)} clients, in_dim={task.in_dim}, "
          f"days={args.days}, val_frac={args.val_frac}")

    # candidate configs per method (label -> cfg dict)
    candidates = {
        "ours": {f"psi={p},lam={l}": {"psi": p, "lam": l}
                 for p, l in itertools.product(OURS_PSI, OURS_LAM)},
        "trimmed_mean": {f"trim={t}": {"trim": t} for t in TRIM},
        "fltrust": {f"root={r}": {"root_size": r} for r in ROOT},
    }
    default_label = {"ours": "psi=0.7,lam=0.85", "trimmed_mean": "trim=0.2",
                     "fltrust": "root=200"}

    ex = ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                             initargs=(task,))

    # ---- validation selection ----
    val_jobs, val_key = [], []
    for method, cand in candidates.items():
        for label, cfg in cand.items():
            for a in ATTACKS:
                for s in range(args.val_seeds):
                    val_jobs.append((method, cfg, a, s, "val"))
                    val_key.append((method, label))
    print(f"validation: {len(val_jobs)} runs on {args.jobs} workers ...")
    val_res = list(ex.map(_run, val_jobs))
    agg = {}
    for (method, label), mae in zip(val_key, val_res):
        agg.setdefault((method, label), []).append(mae)
    val_mae = {k: _mean(v) for k, v in agg.items()}

    selected = {}
    for method, cand in candidates.items():
        labels = list(cand)
        best = min(labels, key=lambda L: val_mae[(method, L)])
        selected[method] = best
        print(f"\n[{method}] validation MAE by setting:")
        for L in labels:
            star = "  <-- selected" if L == best else ""
            base = "  (default)" if L == default_label[method] else ""
            print(f"    {L:20s} val={val_mae[(method, L)]:.4f}{base}{star}")

    # ---- test reporting at selected (+ default) settings ----
    report_methods = ["fedavg", "trimmed_mean", "krum", "fltrust", "foolsgold", "ours"]
    def cfg_for(method, label):
        return candidates.get(method, {}).get(label, {})

    test_jobs, test_key = [], []
    for method in report_methods:
        settings = {"selected": selected.get(method, "default")}
        if method in default_label and selected.get(method) != default_label[method]:
            settings["default"] = default_label[method]
        for tag, label in settings.items():
            for a in ATTACKS:
                for s in range(args.test_seeds):
                    test_jobs.append((method, cfg_for(method, label), a, s, "test"))
                    test_key.append((method, tag, a))
    print(f"\ntest: {len(test_jobs)} runs ...")
    test_res = list(ex.map(_run, test_jobs))
    ex.shutdown()

    tagg = {}
    for (method, tag, a), mae in zip(test_key, test_res):
        tagg.setdefault((method, tag, a), []).append(mae)

    print("\n==== TEST MAE (mean over seeds) at selected settings ====")
    print(f"{'method':13s} {'setting':22s} {'model':>10s} {'collusion':>10s}")
    summary = {}
    for method in report_methods:
        for tag in ("selected", "default"):
            if (method, tag, "model") not in tagg:
                continue
            mm = _mean(tagg[(method, tag, "model")])
            cc = _mean(tagg[(method, tag, "collusion")])
            label = selected.get(method, "default") if tag == "selected" else default_label.get(method, "default")
            print(f"{method:13s} {tag+':'+label:22s} {mm:10.4f} {cc:10.4f}")
            summary[f"{method}/{tag}"] = {"setting": label, "model": mm, "collusion": cc}

    out = {"days": args.days, "clients": len(task.clients),
           "selected": selected, "val_mae": {f"{m}|{l}": v for (m, l), v in val_mae.items()},
           "test": summary}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
