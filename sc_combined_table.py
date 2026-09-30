"""
sc_combined_table.py
Combine the per-dataset grid results into ONE LaTeX comparison table:
rows are methods grouped into a block per dataset, columns are the benign case
and each attack family at 20% adversaries. The best (lowest) MAE in each
column of each dataset block is bold-faced. Mean +/- std over completed seeds;
diverged cells are shown in compact scientific form.

  python sc_combined_table.py --out table_combined.tex
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

RESULTS = os.environ.get("SC_RESULTS_DIR", "results")
DATASETS = [("nyc_taxi", "NYC-Taxi"), ("foursquare", "Foursquare"),
            ("yelp", "Yelp"), ("geolife", "Geolife")]
METHOD_ORDER = ["fedavg", "trimmed_mean", "krum", "fltrust", "foolsgold", "ours"]
METHOD_LABEL = {"fedavg": "FedAvg", "trimmed_mean": "Trimmed Mean",
                "krum": "Krum", "fltrust": "FLTrust", "foolsgold": "FoolsGold",
                "ours": r"\textbf{Ours}"}
# (attack, ratio%, column header)
CONDS = [("none", 0, "Benign"), ("data", 20, r"Data"),
         ("model", 20, r"Model"), ("collusion", 20, r"Collusion")]


def _index():
    ix = defaultdict(list)
    for f in glob.glob(os.path.join(RESULTS, "grid", "*.json")):
        for r in json.load(open(f)):
            ix[(r["dataset"], r["method"], r["attack"],
                int(round(r["ratio"] * 100)))].append(r)
    return ix


def _mean_std(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return None, None
    return float(v.mean()), float(v.std(ddof=1) if v.size > 1 else 0.0)


def _fmt(mean, std, is_best):
    if mean is None:
        return "--"
    if mean < 100:
        s = rf"{mean:.3f}\,$\pm$\,{std:.3f}"
    elif mean < 1e4:
        s = rf"{mean:.0f}"
    else:
        exp = int(np.floor(np.log10(mean)))
        s = rf"${mean/10**exp:.1f}{{\times}}10^{{{exp}}}$"
    return rf"\underline{{{s}}}" if is_best else s


def build(out_path):
    ix = _index()
    ncol = len(CONDS)
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{5pt}",
         r"\caption{Prediction MAE under each attack family at 20\% adversarial "
         r"clients (benign column for reference). Mean $\pm$ std over completed "
         r"seeds; the best (lowest) MAE per column within each dataset is "
         r"\underline{underlined}. Diverged cells are shown in scientific form. "
         r"Lower is better.}",
         r"\label{tab:combined}",
         r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
         r"\textbf{Method} & " +
         " & ".join(rf"\textbf{{{h}}}" for _, _, h in CONDS) + r" \\"]

    present = [(ds, lab) for ds, lab in DATASETS
               if any(ix[(ds, m, a, rt)] for m in METHOD_ORDER for a, rt, _ in CONDS)]
    for ds, ds_label in present:
        # per-column best (min finite mean) within this dataset block
        col_means = []
        for a, rt, _ in CONDS:
            ms = {m: _mean_std([r["mae"] for r in ix[(ds, m, a, rt)]])[0]
                  for m in METHOD_ORDER}
            finite = {m: v for m, v in ms.items() if v is not None}
            col_means.append((ms, min(finite, key=finite.get) if finite else None))

        L.append(r"\midrule")
        L.append(rf"\multicolumn{{{ncol+1}}}{{l}}{{\textit{{{ds_label}}}}} \\")
        for m in METHOD_ORDER:
            cells = []
            for (a, rt, _), (ms, best_m) in zip(CONDS, col_means):
                mean, std = _mean_std([r["mae"] for r in ix[(ds, m, a, rt)]])
                cells.append(_fmt(mean, std, is_best=(m == best_m)))
            L.append(f"{METHOD_LABEL[m]} & " + " & ".join(cells) + r" \\")

    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    tex = "\n".join(L)
    with open(out_path, "w") as f:
        f.write(tex)
    print("wrote", out_path)
    return ix


def preview(ix):
    """Plain-text preview of the same numbers for a sanity check."""
    for ds, ds_label in DATASETS:
        if not any(ix[(ds, m, a, rt)] for m in METHOD_ORDER for a, rt, _ in CONDS):
            continue
        print(f"\n== {ds_label} ==")
        print(f"{'method':13s} " + " ".join(f"{h:>16s}" for _, _, h in CONDS))
        for m in METHOD_ORDER:
            row = []
            for a, rt, _ in CONDS:
                mean, std = _mean_std([r["mae"] for r in ix[(ds, m, a, rt)]])
                row.append("--" if mean is None else
                           (f"{mean:.3f}±{std:.3f}" if mean < 100 else f"{mean:.2e}"))
            print(f"{m:13s} " + " ".join(f"{c:>16s}" for c in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="table_combined.tex")
    args = ap.parse_args()
    ix = build(args.out)
    preview(ix)


if __name__ == "__main__":
    main()
