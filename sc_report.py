"""
sc_report.py
Reads results/ and emits LaTeX that drops into the labels used in the
revision patch: tab:main_results, tab:fairness, tab:trust_influence,
tab:overhead, tab:ablation, plus the breakdown-point figure data.

Every cell is mean +/- std over the completed seeds. A configuration with
fewer than two seeds is printed as a dash rather than as a bare mean.

  python sc_report.py --out tables.tex
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List

RESULTS_DIR = os.environ.get("SC_RESULTS_DIR", "results")

METHOD_LABEL = {
    "fedavg": "FedAvg", "trimmed_mean": "Trimmed Mean", "krum": "Krum",
    "fltrust": "FLTrust", "foolsgold": "FoolsGold", "ours": r"\textbf{Ours}",
    "ours_no_magnitude": r"Ours ($\psi=1$)",
}
METHOD_ORDER = ["fedavg", "trimmed_mean", "krum", "fltrust", "foolsgold", "ours"]
ATTACKS = ["data", "model", "collusion"]
RATIOS = [10, 20, 40]


def _load(block: str) -> List[List[Dict]]:
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, block, "*.json"))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def _cell(vals: List[float], fmt: str = "{:.3f}") -> str:
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return "--"
    if v.size == 1:
        return fmt.format(v[0]) + r"$^{\dagger}$"
    return (fmt.format(v.mean()) + r"\,$\pm$\," +
            fmt.format(v.std(ddof=1)))


def _index(runs: List[List[Dict]]):
    ix = defaultdict(list)
    for group in runs:
        for r in group:
            ix[(r["dataset"], r["method"], r["attack"], int(round(r["ratio"] * 100)))].append(r)
    return ix


# ---------------------------------------------------------------------
def table_main(ix, dataset: str) -> str:
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
         rf"\caption{{Prediction MAE on {dataset} under three attack families "
         r"and three adversarial ratios. Mean $\pm$ std over completed seeds. "
         r"$\dagger$ marks a single-seed cell.}",
         r"\label{tab:main_results}",
         r"\begin{tabular}{lccccccccc}", r"\toprule",
         r"& \multicolumn{3}{c}{\textbf{Data poisoning}}"
         r"& \multicolumn{3}{c}{\textbf{Model poisoning}}"
         r"& \multicolumn{3}{c}{\textbf{Collusion}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r"\textbf{Method} & 10\% & 20\% & 40\% & 10\% & 20\% & 40\% "
         r"& 10\% & 20\% & 40\% \\", r"\midrule"]
    for m in METHOD_ORDER:
        cells = []
        for a in ATTACKS:
            for rr in RATIOS:
                cells.append(_cell([x["mae"] for x in ix[(dataset, m, a, rr)]]))
        L.append(f"{METHOD_LABEL[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def table_degradation(ix, dataset: str, attack="model", ratio=20) -> str:
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         rf"\caption{{Robustness degradation on {dataset} under {attack} "
         rf"poisoning at {ratio}\% adversarial clients, "
         r"$(\mathrm{MAE}_{\mathrm{attack}}-\mathrm{MAE}_{\mathrm{benign}})"
         r"/\mathrm{MAE}_{\mathrm{benign}}$.}",
         r"\label{tab:degradation}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"\textbf{Method} & \textbf{Benign MAE} & \textbf{Attack MAE} "
         r"& \textbf{Degradation} \\", r"\midrule"]
    for m in METHOD_ORDER:
        ben = [x["mae"] for x in ix[(dataset, m, "none", 0)]]
        att = [x["mae"] for x in ix[(dataset, m, attack, ratio)]]
        deg = ([(a - b) / b for a, b in zip(att, ben)]
               if len(ben) == len(att) and ben else [])
        L.append(f"{METHOD_LABEL[m]} & {_cell(ben)} & {_cell(att)} "
                 f"& {_cell(deg, '{:.2f}')} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table_fairness(ix, dataset: str, attack="model", ratio=20) -> str:
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\caption{Standard deviation of per-region MAE. Lower values "
         r"indicate more consistent regional error. This is a measure of "
         r"error consistency, not of fare fairness.}",
         r"\label{tab:fairness}",
         r"\renewcommand{\arraystretch}{1.1}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"\textbf{Method} & \textbf{All Regions $\downarrow$} "
         r"& \textbf{Surge Zones $\downarrow$} "
         r"& \textbf{Non-Surge Zones $\downarrow$} \\", r"\midrule"]
    for m in METHOD_ORDER:
        rs = ix[(dataset, m, attack, ratio)]
        L.append(f"{METHOD_LABEL[m]} & "
                 f"{_cell([x['fairness_all'] for x in rs])} & "
                 f"{_cell([x['fairness_surge'] for x in rs])} & "
                 f"{_cell([x['fairness_nonsurge'] for x in rs])} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table_influence(ix, dataset: str, attack="model", ratio=20) -> str:
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\caption{Client influence, Eq.~(\ref{eq:influence}), averaged over "
         r"selected rounds and grouped by client type. The uniform share is "
         r"reported for reference.}",
         r"\label{tab:trust_influence}",
         r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"\textbf{Method} & \textbf{Honest $\uparrow$} "
         r"& \textbf{Malicious $\downarrow$} & \textbf{Trust conv. (rounds)} \\",
         r"\midrule"]
    for m in METHOD_ORDER:
        rs = ix[(dataset, m, attack, ratio)]
        conv = [x["trust"]["convergence_round"] for x in rs
                if np.isfinite(x["trust"]["convergence_round"])]
        L.append(f"{METHOD_LABEL[m]} & "
                 f"{_cell([x['influence']['honest'] for x in rs])} & "
                 f"{_cell([x['influence']['malicious'] for x in rs])} & "
                 f"{_cell(conv, '{:.1f}') if conv else '--'} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table_selection(dataset: str) -> str:
    runs = _load("selection")
    by = defaultdict(list)
    for g in runs:
        for r in g:
            if r["dataset"] == dataset:
                by[r["selection"]].append(r)
    label = {"uniform": "Honest uniform sampling",
             "vrf": "VRF self-selection",
             "biased": "Biasing coordinator"}
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\caption{Realised adversarial share $\beta_t$ and resulting MAE "
         r"under the three selection regimes of "
         r"Proposition~\ref{prop:concentration}, at $\alpha=0.20$.}",
         r"\label{tab:selection}",
         r"\begin{tabular}{lcc}", r"\toprule",
         r"\textbf{Selection regime} & $\bar{\beta}_t$ & \textbf{MAE} \\",
         r"\midrule"]
    for k in ["uniform", "vrf", "biased"]:
        rs = by.get(k, [])
        L.append(f"{label[k]} & {_cell([x['beta_mean'] for x in rs])} & "
                 f"{_cell([x['mae'] for x in rs])} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table_overhead(ix, dataset: str, attack="model", ratio=20) -> str:
    try:
        with open(os.path.join(RESULTS_DIR, "overhead", "proof_cost.json")) as f:
            pc = json.load(f)
    except FileNotFoundError:
        pc = {"verify_ms": float("nan"), "proof_bytes": float("nan")}

    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\caption{Measured per-round overhead. Server time is the "
         r"aggregation cost; proof verification is reported separately "
         r"because it applies only to the selection stage.}",
         r"\label{tab:overhead}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcc}", r"\toprule",
         r"\textbf{Method} & \textbf{Server time (ms/round)} "
         r"& \textbf{Proof verify (ms/client)} \\", r"\midrule"]
    for m in METHOD_ORDER:
        rs = ix[(dataset, m, attack, ratio)]
        pv = (f"{pc['verify_ms']:.3f}" if m == "ours"
              and np.isfinite(pc.get("verify_ms", float("nan"))) else "--")
        L.append(f"{METHOD_LABEL[m]} & "
                 f"{_cell([x['server_time_ms_per_round'] for x in rs], '{:.1f}')} "
                 f"& {pv} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def breakdown_points(threshold: float = 0.10) -> str:
    """
    First beta at which MAE exceeds the beta=0 baseline by `threshold`
    relative. Reported as a table and as the data behind the sweep figure.
    """
    runs = _load("phase")
    by = defaultdict(dict)
    for g in runs:
        for r in g:
            key = (r["dataset"], r["method"], r["attack"])
            by[key].setdefault(r["forced_beta"], []).append(r["mae"])

    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         rf"\caption{{Empirical breakdown point: the smallest committee "
         rf"adversarial share $\beta$ at which MAE exceeds its "
         rf"$\beta=0$ value by {int(threshold*100)}\%.}}",
         r"\label{tab:breakdown}",
         r"\begin{tabular}{lcc}", r"\toprule",
         r"\textbf{Method} & \textbf{Model poisoning} & \textbf{Collusion} \\",
         r"\midrule"]
    datasets = sorted({k[0] for k in by})
    for ds in datasets:
        for m in METHOD_ORDER:
            row = []
            for a in ["model", "collusion"]:
                d = by.get((ds, m, a), {})
                if not d:
                    row.append("--"); continue
                bs = sorted(d)
                base = np.mean(d[bs[0]])
                bp = next((b for b in bs
                           if np.mean(d[b]) > base * (1 + threshold)), None)
                row.append(f"{bp:.2f}" if bp is not None else r"$>$0.50")
            L.append(f"{METHOD_LABEL[m]} & " + " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc_taxi")
    ap.add_argument("--out", default="tables.tex")
    args = ap.parse_args()

    ix = _index(_load("grid"))
    parts = [
        table_main(ix, args.dataset),
        table_degradation(ix, args.dataset),
        table_fairness(ix, args.dataset),
        table_influence(ix, args.dataset),
        table_selection(args.dataset),
        table_overhead(ix, args.dataset),
        breakdown_points(),
    ]
    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
