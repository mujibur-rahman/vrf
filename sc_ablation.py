"""
sc_ablation.py
Two-factor ablation: selection (VRF-bounded beta vs coordinator-concentrated
beta) x aggregation (mean vs trust-weighted). 20% population adversaries.

  * FedAvg      : mean aggregation, coordinator biases the committee (beta=0.5)
  * Trust only  : trust aggregation, coordinator biases the committee (beta=0.5)
  * VRF only    : mean aggregation, VRF keeps beta = alpha = 0.2
  * Full method : trust aggregation, VRF keeps beta = alpha = 0.2

The selection axis is realised with run_fl's forced_beta (committee adversarial
share), so the VRF benefit = moving beta from the concentrated 0.5 back to the
population rate 0.2. Metrics: MAE, fairness (per-region MAE std), malicious
influence. Reported per attack, mean +/- std over seeds.

  python sc_ablation.py --seeds 5 --jobs 12          # NYC_DAYS via env (default 7)
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RATIO = 0.20            # population adversarial share (alpha)
BETA_VRF = 0.20         # committee share under VRF (= alpha, Proposition 1)
BETA_BIAS = 0.50        # committee share a biasing coordinator can concentrate

# (label, aggregation method, forced committee beta)
VARIANTS = [
    ("FedAvg",      "fedavg", BETA_BIAS),
    ("Trust only",  "ours",   BETA_BIAS),
    ("VRF only",    "fedavg", BETA_VRF),
    ("Full method", "ours",   BETA_VRF),
]
ATTACKS = ["model", "collusion"]
_TASK = None


def _init(task):
    import torch
    torch.set_num_threads(1)
    global _TASK
    _TASK = task


def _run(job):
    import sc_experiments as E
    method, beta, attack, seed = job
    r = E.run_fl(_TASK, method, attack, RATIO, seed, forced_beta=beta)
    return (r["mae"], r["fairness_all"], r["influence"]["malicious"])


def _ms(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return (float(np.mean(xs)), float(np.std(xs, ddof=1) if len(xs) > 1 else 0.0)) \
        if xs else (float("nan"), 0.0)


def _cell(mean, std):
    if not np.isfinite(mean):
        return "--"
    if mean >= 100:
        return rf"${mean:.0e}$"
    return rf"{mean:.3f}\,$\pm$\,{std:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default="table_ablation.tex")
    args = ap.parse_args()

    import sc_tasks as T
    task = T.build_task("nyc_taxi")
    print(f"task: {len(task.clients)} clients (NYC_DAYS={os.environ.get('NYC_DAYS','7')})")

    jobs, key = [], []
    for label, method, beta in VARIANTS:
        for a in ATTACKS:
            for s in range(args.seeds):
                jobs.append((method, beta, a, s)); key.append((label, a))
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                             initargs=(task,)) as ex:
        res = list(ex.map(_run, jobs))

    agg = {}
    for (label, a), (mae, fair, mal) in zip(key, res):
        agg.setdefault((label, a), {"mae": [], "fair": [], "mal": []})
        agg[(label, a)]["mae"].append(mae)
        agg[(label, a)]["fair"].append(fair)
        agg[(label, a)]["mal"].append(mal)

    for a in ATTACKS:
        print(f"\n=== {a} poisoning (20% adversaries; no-VRF beta={BETA_BIAS}, "
              f"VRF beta={BETA_VRF}) ===")
        print(f"{'variant':13s} {'MAE':>16s} {'Fairness':>16s} {'Mal.Inf':>16s}")
        for label, _, _ in VARIANTS:
            d = agg[(label, a)]
            mm, ms = _ms(d["mae"]); fm, fs = _ms(d["fair"]); vm, vs = _ms(d["mal"])
            print(f"{label:13s} {mm:8.3f}+-{ms:.3f} {fm:8.3f}+-{fs:.3f} "
                  f"{vm:8.3f}+-{vs:.3f}")

    # LaTeX table for the model-poisoning setting (primary)
    prim = "model"
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Ablation under 20\% adversarial clients ({prim} poisoning). "
         rf"Selection axis: a biasing coordinator concentrates the committee to "
         rf"$\beta={BETA_BIAS}$ without VRF, while VRF holds $\beta=\alpha={BETA_VRF}$. "
         r"Lower is better.}",
         r"\label{tab:ablation}", r"\scriptsize",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"\textbf{Method Variant} & \textbf{MAE $\downarrow$} & "
         r"\textbf{Fairness $\downarrow$} & \textbf{Mal. Inf. $\downarrow$} \\",
         r"\midrule"]
    # bold best (min) per column among finite means
    means = {label: {c: _ms(agg[(label, prim)][k])[0]
                     for c, k in [("mae", "mae"), ("fair", "fair"), ("mal", "mal")]}
             for label, _, _ in VARIANTS}
    best = {c: min((means[l][c] for l, _, _ in VARIANTS if np.isfinite(means[l][c])),
                   default=np.nan) for c in ["mae", "fair", "mal"]}
    for label, _, _ in VARIANTS:
        d = agg[(label, prim)]
        cells = []
        for c, k in [("mae", "mae"), ("fair", "fair"), ("mal", "mal")]:
            mean, std = _ms(d[k])
            s = _cell(mean, std)
            if np.isfinite(mean) and np.isclose(mean, best[c]):
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        L.append(f"{label} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(args.out, "w").write("\n".join(L))
    print(f"\nwrote {args.out}")
    json.dump({f"{l}|{a}": {kk: _ms(agg[(l, a)][kk]) for kk in ("mae", "fair", "mal")}
               for l, _, _ in VARIANTS for a in ATTACKS},
              open("ablation_nyc.json", "w"), indent=2)


if __name__ == "__main__":
    main()
