"""
sc_plots.py
Publication figures for the surge-pricing FL robustness study.
Reads results/ (written by the experiment blocks) and emits vector PDF (for
LaTeX \\includegraphics) plus PNG previews under figures/.

  python sc_plots.py --dataset nyc_taxi

Figures:
  fig_selection      Proposition 1: realised beta and MAE per selection regime
  fig_breakdown      MAE vs committee adversarial share beta (model, collusion)
  fig_attacks        MAE per method under each attack at 20% adversaries
  fig_influence      honest vs malicious client influence per method
  fig_crossdataset   MAE under model poisoning across all datasets (comparison)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})

FIG_DIR = "figures"
RESULTS = os.environ.get("SC_RESULTS_DIR", "results")

METHODS = ["fedavg", "trimmed_mean", "krum", "fltrust", "foolsgold", "ours"]
LABEL = {"fedavg": "FedAvg", "trimmed_mean": "Trimmed Mean", "krum": "Krum",
         "fltrust": "FLTrust", "foolsgold": "FoolsGold", "ours": "Ours"}
# Colour-blind-safe palette; "Ours" is black and drawn thick so it stands out.
COLOR = {"fedavg": "#999999", "trimmed_mean": "#0072B2", "krum": "#009E73",
         "fltrust": "#E69F00", "foolsgold": "#CC79A7", "ours": "#000000"}
MARK = {"fedavg": "o", "trimmed_mean": "s", "krum": "^", "fltrust": "D",
        "foolsgold": "v", "ours": "*"}


# ---------------------------------------------------------------------
def _grid_index():
    ix = defaultdict(list)
    for f in glob.glob(os.path.join(RESULTS, "grid", "*.json")):
        for r in json.load(open(f)):
            ix[(r["dataset"], r["method"], r["attack"],
                int(round(r["ratio"] * 100)))].append(r)
    return ix


def _stat(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), 0.0
    return float(v.mean()), float(v.std(ddof=1) if v.size > 1 else 0.0)


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(FIG_DIR, f"{name}.pdf"))


def _capped_bar(ax, xs, vals, cap, **kw):
    """Bar plot with values above `cap` clipped to the top and annotated."""
    shown = [min(v, cap) if np.isfinite(v) else 0.0 for v in vals]
    bars = ax.bar(xs, shown, **kw)
    for x, v in zip(xs, vals):
        if np.isfinite(v) and v > cap:
            ax.annotate(f"{v:.0e}", (x, cap), ha="center", va="bottom",
                        fontsize=6.5, rotation=90, xytext=(0, 1),
                        textcoords="offset points")
    return bars


# ---------------------------------------------------------------------
def fig_selection(dataset):
    regimes = ["uniform", "vrf", "biased"]
    if not any(os.path.exists(os.path.join(RESULTS, "selection", f"{dataset}_{k}.json"))
               for k in regimes):
        print(f"skip fig_selection ({dataset}): no selection runs"); return
    rlabel = {"uniform": "Uniform\n(honest)", "vrf": "VRF\n(self-select)",
              "biased": "Biased\n(coordinator)"}
    beta_m, beta_s, mae_m = [], [], []
    for k in regimes:
        p = os.path.join(RESULTS, "selection", f"{dataset}_{k}.json")
        if not os.path.exists(p):
            beta_m.append(np.nan); beta_s.append(0); mae_m.append(np.nan); continue
        g = json.load(open(p))
        bm, bs = _stat([x["beta_mean"] for x in g]); mm, _ = _stat([x["mae"] for x in g])
        beta_m.append(bm); beta_s.append(bs); mae_m.append(mm)

    fig, ax1 = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(regimes))
    ax1.bar(x - 0.2, beta_m, 0.4, yerr=beta_s, capsize=3,
            color="#0072B2", label=r"realised $\bar{\beta}$")
    ax1.axhline(0.20, ls="--", c="grey", lw=1, label=r"$\alpha=0.20$")
    ax1.set_ylabel(r"adversarial share $\bar{\beta}_t$")
    ax1.set_ylim(0, 1.05); ax1.set_xticks(x)
    ax1.set_xticklabels([rlabel[k] for k in regimes])
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, mae_m, 0.4, color="#D55E00", alpha=0.85, label="MAE")
    ax2.set_ylabel("MAE (log)"); ax2.set_yscale("log"); ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)
    ax1.set_title(f"Selection regimes ({dataset})")
    _save(fig, f"fig_selection_{dataset}")


def fig_breakdown(dataset):
    by = defaultdict(dict)
    for f in glob.glob(os.path.join(RESULTS, "phase", "*.json")):
        for r in json.load(open(f)):
            if r["dataset"] != dataset:
                continue
            by[(r["method"], r["attack"])].setdefault(r["forced_beta"], []).append(r["mae"])
    if not by:
        print("no phase data for", dataset); return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, attack in zip(axes, ["model", "collusion"]):
        for m in METHODS:
            d = by.get((m, attack), {})
            if not d:
                continue
            bs = sorted(d)
            mu = [np.mean(d[b]) for b in bs]
            ax.plot(bs, mu, marker=MARK[m], color=COLOR[m], label=LABEL[m],
                    lw=2.2 if m == "ours" else 1.3,
                    ms=8 if m == "ours" else 5, zorder=5 if m == "ours" else 3)
        ax.set_yscale("log")
        ax.set_xlabel(r"committee adversarial share $\beta$")
        ax.set_title(f"{attack.capitalize()} poisoning")
    axes[0].set_ylabel("MAE (log)")
    axes[1].legend(loc="upper left", ncol=1, framealpha=0.9)
    fig.suptitle(f"Empirical breakdown point ({dataset})", y=1.02)
    _save(fig, f"fig_breakdown_{dataset}")


def _selection_panel(ax1, dataset):
    regimes = ["uniform", "vrf", "biased"]
    rlabel = {"uniform": "Uniform", "vrf": "VRF", "biased": "Biased"}
    beta_m, beta_s, mae_m = [], [], []
    for k in regimes:
        p = os.path.join(RESULTS, "selection", f"{dataset}_{k}.json")
        if not os.path.exists(p):
            beta_m.append(np.nan); beta_s.append(0); mae_m.append(np.nan); continue
        g = json.load(open(p))
        bm, bs = _stat([x["beta_mean"] for x in g])
        mm, _ = _stat([x["mae"] for x in g])
        beta_m.append(bm); beta_s.append(bs); mae_m.append(mm)
    x = np.arange(len(regimes))
    ax1.bar(x - 0.2, beta_m, 0.4, yerr=beta_s, capsize=3, color="#0072B2")
    ax1.axhline(0.20, ls="--", c="grey", lw=1)
    ax1.set_ylim(0, 1.08); ax1.set_xticks(x)
    ax1.set_xticklabels([rlabel[k] for k in regimes])
    ax1.set_ylabel(r"$\bar{\beta}_t$")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, mae_m, 0.4, color="#D55E00", alpha=0.85)
    ax2.set_yscale("log"); ax2.set_ylabel("MAE (log)"); ax2.grid(False)


def fig_selection_all(datasets, ds_labels=None):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ds_labels = ds_labels or {d: d for d in datasets}
    present = [d for d in datasets
               if os.path.exists(os.path.join(RESULTS, "selection", f"{d}_vrf.json"))]
    if not present:
        print("no selection runs"); return
    nrow = (len(present) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(8.4, 3.1 * nrow), squeeze=False)
    for idx, ds in enumerate(present):
        ax = axes[idx // 2][idx % 2]
        _selection_panel(ax, ds)
        ax.set_title(ds_labels[ds])
    for j in range(len(present), nrow * 2):
        axes[j // 2][j % 2].axis("off")
    handles = [Patch(facecolor="#0072B2", label=r"realised $\bar{\beta}$"),
               Line2D([0], [0], ls="--", c="grey", label=r"$\alpha=0.20$"),
               Patch(facecolor="#D55E00", alpha=0.85, label="MAE (log)")]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.03), framealpha=0.95)
    fig.suptitle("Selection regimes (Proposition 1) across datasets", y=1.06)
    fig.tight_layout()
    _save(fig, "fig_selection_all")


def fig_breakdown_all(datasets, ds_labels=None):
    """One combined breakdown figure: rows = datasets, cols = model/collusion."""
    ds_labels = ds_labels or {d: d for d in datasets}
    by = defaultdict(dict)
    present = []
    for f in glob.glob(os.path.join(RESULTS, "phase", "*.json")):
        for r in json.load(open(f)):
            by[(r["dataset"], r["method"], r["attack"])].setdefault(
                r["forced_beta"], []).append(r["mae"])
    present = [d for d in datasets
               if any((d, m, "model") in by for m in METHODS)]
    if not present:
        print("no phase data"); return

    attacks = ["model", "collusion"]
    nrow, ncol = len(present), len(attacks)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.3 * nrow),
                             sharex="col", sharey="row", squeeze=False)
    handles = labels = None
    for i, ds in enumerate(present):
        for j, attack in enumerate(attacks):
            ax = axes[i][j]
            for m in METHODS:
                d = by.get((ds, m, attack), {})
                if not d:
                    continue
                bs = sorted(d)
                mu = [np.mean(d[b]) for b in bs]
                ax.plot(bs, mu, marker=MARK[m], color=COLOR[m], label=LABEL[m],
                        lw=2.2 if m == "ours" else 1.2,
                        ms=7 if m == "ours" else 4,
                        zorder=5 if m == "ours" else 3)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"{attack.capitalize()} poisoning")
            if i == nrow - 1:
                ax.set_xlabel(r"committee adversarial share $\beta$")
            if j == 0:
                ax.set_ylabel(f"{ds_labels[ds]}\nMAE (log)")
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.95)
    fig.suptitle("Empirical breakdown point across datasets", y=1.05)
    fig.tight_layout()
    _save(fig, "fig_breakdown_all")


def fig_attacks(dataset):
    ix = _grid_index()
    cols = [("none", 0, "None"), ("data", 20, "Data"),
            ("model", 20, "Model"), ("collusion", 20, "Collusion")]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(cols)); w = 0.13
    # Cap just above the robust-method range so their differences are visible;
    # diverging methods are drawn to the cap and labelled with their true value.
    allv = [_stat([r["mae"] for r in ix[(dataset, m, a, rt)]])[0]
            for m in METHODS for a, rt, _ in cols]
    robust_max = max([v for v in allv if np.isfinite(v) and v < 10], default=1.0)
    cap = robust_max * 2.2
    for i, m in enumerate(METHODS):
        means = [_stat([r["mae"] for r in ix[(dataset, m, a, rt)]])[0] for a, rt, _ in cols]
        errs = [_stat([r["mae"] for r in ix[(dataset, m, a, rt)]])[1] for a, rt, _ in cols]
        xs = x + (i - len(METHODS)/2 + 0.5) * w
        shown = [min(v, cap) if np.isfinite(v) else 0 for v in means]
        se = [e if (np.isfinite(v) and v <= cap) else 0 for v, e in zip(means, errs)]
        ax.bar(xs, shown, w, yerr=se, capsize=2, color=COLOR[m], label=LABEL[m],
               edgecolor="black" if m == "ours" else "none", linewidth=1.0)
        for xi, v in zip(xs, means):
            if np.isfinite(v) and v > cap:
                ax.annotate(f"{v:.0e}", (xi, cap), ha="center", va="bottom",
                            fontsize=6, rotation=90, xytext=(0, 1),
                            textcoords="offset points")
    ax.axhline(cap, ls=":", c="grey", lw=0.7)
    ax.set_yscale("log"); ax.set_ylim(top=cap * 3.0)
    ax.set_xticks(x); ax.set_xticklabels([c[2] for c in cols])
    ax.set_xlabel("attack (20% adversaries)"); ax.set_ylabel("MAE (log)")
    ax.set_title(f"Robustness to attacks ({dataset}) -- bars above dotted line diverge",
                 pad=10)
    ax.legend(ncol=3, loc="upper left", framealpha=0.95)
    _save(fig, f"fig_attacks_{dataset}")


def _attacks_panel(ax, ix, dataset, cols):
    x = np.arange(len(cols)); w = 0.13
    allv = [_stat([r["mae"] for r in ix[(dataset, m, a, rt)]])[0]
            for m in METHODS for a, rt, _ in cols]
    robust_max = max([v for v in allv if np.isfinite(v) and v < 10], default=1.0)
    cap = robust_max * 2.2
    for i, m in enumerate(METHODS):
        means = [_stat([r["mae"] for r in ix[(dataset, m, a, rt)]])[0] for a, rt, _ in cols]
        errs = [_stat([r["mae"] for r in ix[(dataset, m, a, rt)]])[1] for a, rt, _ in cols]
        xs = x + (i - len(METHODS)/2 + 0.5) * w
        shown = [min(v, cap) if np.isfinite(v) else 0 for v in means]
        se = [e if (np.isfinite(v) and v <= cap) else 0 for v, e in zip(means, errs)]
        ax.bar(xs, shown, w, yerr=se, capsize=1.5, color=COLOR[m],
               edgecolor="black" if m == "ours" else "none", linewidth=0.8)
        for xi, v in zip(xs, means):
            if np.isfinite(v) and v > cap:
                ax.annotate(f"{v:.0e}", (xi, cap), ha="center", va="bottom",
                            fontsize=5.5, rotation=90, xytext=(0, 1),
                            textcoords="offset points")
    ax.axhline(cap, ls=":", c="grey", lw=0.7)
    ax.set_yscale("log"); ax.set_ylim(top=cap * 3.0)
    ax.set_xticks(x); ax.set_xticklabels([c[2] for c in cols], fontsize=8)
    ax.set_ylabel("MAE (log)")


def fig_attacks_all(datasets, ds_labels=None):
    from matplotlib.patches import Patch
    ds_labels = ds_labels or {d: d for d in datasets}
    ix = _grid_index()
    cols = [("none", 0, "None"), ("data", 20, "Data"),
            ("model", 20, "Model"), ("collusion", 20, "Collusion")]
    present = [d for d in datasets
               if any(ix[(d, m, a, rt)] for m in METHODS for a, rt, _ in cols)]
    nrow = (len(present) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(9.0, 3.0 * nrow), squeeze=False)
    for idx, ds in enumerate(present):
        ax = axes[idx // 2][idx % 2]
        _attacks_panel(ax, ix, ds, cols)
        ax.set_title(ds_labels[ds])
        if idx // 2 == nrow - 1:
            ax.set_xlabel("attack (20% adversaries)")
    for j in range(len(present), nrow * 2):
        axes[j // 2][j % 2].axis("off")
    handles = [Patch(facecolor=COLOR[m], edgecolor="black" if m == "ours" else "none",
                     label=LABEL[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.95)
    fig.suptitle("MAE per attack (20% adversaries) across datasets — "
                 "bars above dotted line diverge", y=1.05)
    fig.tight_layout()
    _save(fig, "fig_attacks_all")


def fig_influence_all(datasets, ds_labels=None, attack="model", ratio=20):
    from matplotlib.patches import Patch
    ds_labels = ds_labels or {d: d for d in datasets}
    ix = _grid_index()
    present = [d for d in datasets
               if any(ix[(d, m, attack, ratio)] for m in METHODS)]
    nrow = (len(present) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(9.0, 3.0 * nrow), squeeze=False)
    for idx, ds in enumerate(present):
        ax = axes[idx // 2][idx % 2]
        hon = [_stat([r["influence"]["honest"] for r in ix[(ds, m, attack, ratio)]])[0]
               for m in METHODS]
        mal = [_stat([r["influence"]["malicious"] for r in ix[(ds, m, attack, ratio)]])[0]
               for m in METHODS]
        xx = np.arange(len(METHODS)); w = 0.38
        ax.bar(xx - w/2, hon, w, color="#009E73", label="honest")
        ax.bar(xx + w/2, mal, w, color="#D55E00", label="malicious")
        ax.set_xticks(xx)
        ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel("mean influence")
        ax.set_title(ds_labels[ds])
    for j in range(len(present), nrow * 2):
        axes[j // 2][j % 2].axis("off")
    handles = [Patch(facecolor="#009E73", label="honest"),
               Patch(facecolor="#D55E00", label="malicious")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.95)
    fig.suptitle(f"Client influence, {attack} @ {ratio}% across datasets", y=1.05)
    fig.tight_layout()
    _save(fig, "fig_influence_all")


def fig_influence(dataset, attack="model", ratio=20):
    ix = _grid_index()
    hon, mal = [], []
    for m in METHODS:
        rs = ix[(dataset, m, attack, ratio)]
        hon.append(_stat([r["influence"]["honest"] for r in rs])[0])
        mal.append(_stat([r["influence"]["malicious"] for r in rs])[0])
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = np.arange(len(METHODS)); w = 0.38
    ax.bar(x - w/2, hon, w, color="#009E73", label="honest")
    ax.bar(x + w/2, mal, w, color="#D55E00", label="malicious")
    ax.axhline(np.nanmean(hon + mal), ls=":", c="grey", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=25, ha="right")
    ax.set_ylabel("mean client influence")
    ax.set_title(f"Client influence, {attack} @ {ratio}% ({dataset})")
    ax.legend(framealpha=0.9)
    _save(fig, f"fig_influence_{dataset}")


def fig_crossdataset(datasets, attack="model", ratio=20):
    ix = _grid_index()
    present = [d for d in datasets
               if any(ix[(d, m, attack, ratio)] for m in METHODS)]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(present)); w = 0.13
    allv = [_stat([r["mae"] for r in ix[(d, m, attack, ratio)]])[0]
            for d in present for m in METHODS]
    robust = [v for v in allv if np.isfinite(v) and v < 10]
    cap = max(robust, default=1.0) * 2.2
    bottom = min([v for v in allv if np.isfinite(v) and v > 0], default=1e-3) / 2
    for i, m in enumerate(METHODS):
        means = [_stat([r["mae"] for r in ix[(d, m, attack, ratio)]])[0] for d in present]
        xs = x + (i - len(METHODS)/2 + 0.5) * w
        shown = [min(v, cap) if np.isfinite(v) else 0 for v in means]
        ax.bar(xs, shown, w, color=COLOR[m], label=LABEL[m],
               edgecolor="black" if m == "ours" else "none", linewidth=1.0)
        for xi, v in zip(xs, means):
            if np.isfinite(v) and v > cap:
                ax.annotate(f"{v:.0e}", (xi, cap), ha="center", va="bottom",
                            fontsize=6, rotation=90, xytext=(0, 1),
                            textcoords="offset points")
    ax.axhline(cap, ls=":", c="grey", lw=0.7)
    ax.set_yscale("log"); ax.set_ylim(bottom, cap * 4.0)
    ax.set_xticks(x); ax.set_xticklabels(present)
    ax.set_ylabel("MAE (log)")
    ax.set_title(f"Cross-dataset robustness ({attack} poisoning @ {ratio}%) "
                 "-- bars above dotted line diverge", pad=10)
    ax.legend(ncol=3, loc="upper left", framealpha=0.95)
    _save(fig, "fig_crossdataset_model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc_taxi")
    ap.add_argument("--all-datasets", nargs="+",
                    default=["nyc_taxi", "foursquare", "geolife", "yelp"])
    args = ap.parse_args()
    labels = {"nyc_taxi": "NYC-Taxi", "foursquare": "Foursquare",
              "yelp": "Yelp", "geolife": "Geolife"}
    labels = {d: labels.get(d, d) for d in args.all_datasets}
    fig_selection_all(args.all_datasets, labels)     # combined
    fig_breakdown_all(args.all_datasets, labels)     # combined
    fig_attacks_all(args.all_datasets, labels)       # combined
    fig_influence_all(args.all_datasets, labels)     # combined
    fig_crossdataset(args.all_datasets)


if __name__ == "__main__":
    main()
