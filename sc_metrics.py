"""
sc_metrics.py
Every metric declared in Section IV-E, including the two that were declared
in the submitted version but never reported.

Influence uses a single definition, Eq. (13), so the value appearing in the
main results table and the value in the honest/malicious table are the same
quantity computed over different client groups.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Sequence


# ---------------------------------------------------------------------
# Prediction error
# ---------------------------------------------------------------------
def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def per_region_mae(preds: Dict[str, np.ndarray],
                   trues: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {r: mae(preds[r], trues[r]) for r in preds}


def global_mae(preds: Dict[str, np.ndarray],
               trues: Dict[str, np.ndarray]) -> float:
    p = np.concatenate([preds[r].ravel() for r in preds])
    t = np.concatenate([trues[r].ravel() for r in trues])
    return mae(p, t)


# ---------------------------------------------------------------------
# Fairness (regional error consistency)
# ---------------------------------------------------------------------
def fairness_score(region_mae: Dict[str, float],
                   subset: Sequence[str] | None = None) -> float:
    """
    Standard deviation of per-region MAE. This measures consistency of
    predictive error across regions. It is not a measure of fare fairness,
    and the manuscript should not describe it as one.
    """
    if subset is None:
        keys = list(region_mae)
    else:
        # An explicitly empty subset (e.g. there are no non-surge zones) must
        # report NaN, not silently fall back to the full set of regions.
        keys = list(subset)
    vals = [region_mae[r] for r in keys if r in region_mae]
    return float(np.std(vals)) if vals else float("nan")


# ---------------------------------------------------------------------
# Robustness degradation (declared in IV-E, previously unreported)
# ---------------------------------------------------------------------
def robustness_degradation(mae_attack: float, mae_benign: float) -> float:
    return float((mae_attack - mae_benign) / (mae_benign + 1e-12))


# ---------------------------------------------------------------------
# Client influence, Eq. (13)
# ---------------------------------------------------------------------
def round_influence(weights: Dict[int, float],
                    update_norms: Dict[int, float]) -> Dict[int, float]:
    denom = sum(weights[i] * update_norms[i] for i in weights) + 1e-12
    return {i: (weights[i] * update_norms[i]) / denom for i in weights}


class InfluenceTracker:
    """Averages Eq. (13) over the rounds in which each client was selected."""

    def __init__(self):
        self.sums: Dict[int, float] = {}
        self.counts: Dict[int, int] = {}

    def update(self, infl: Dict[int, float]) -> None:
        for i, v in infl.items():
            self.sums[i] = self.sums.get(i, 0.0) + v
            self.counts[i] = self.counts.get(i, 0) + 1

    def per_client(self) -> Dict[int, float]:
        return {i: self.sums[i] / self.counts[i] for i in self.sums}

    def group_means(self, malicious: Sequence[int]) -> Dict[str, float]:
        pc = self.per_client()
        mal = set(malicious)
        h = [v for i, v in pc.items() if i not in mal]
        m = [v for i, v in pc.items() if i in mal]
        return {"honest": float(np.mean(h)) if h else float("nan"),
                "malicious": float(np.mean(m)) if m else float("nan")}


# ---------------------------------------------------------------------
# Trust convergence
# ---------------------------------------------------------------------
def trust_separation(history: List[np.ndarray],
                     malicious: Sequence[int]) -> Dict[str, float]:
    """
    Separation is the gap between mean honest trust and mean malicious trust.
    Convergence round is the first round after which the gap stays within 5%
    of its final value, reported instead of the qualitative "Stable" label.
    """
    if not history:
        return {"final_gap": float("nan"), "convergence_round": float("nan")}

    H = np.stack(history)
    mal = np.zeros(H.shape[1], dtype=bool)
    mal[list(malicious)] = True

    gap = H[:, ~mal].mean(1) - H[:, mal].mean(1) if mal.any() else H.mean(1) * 0
    final = gap[-1]
    tol = 0.05 * abs(final) + 1e-9
    conv = len(gap)
    for t in range(len(gap)):
        if np.all(np.abs(gap[t:] - final) <= tol):
            conv = t + 1
            break
    return {"final_gap": float(final), "convergence_round": float(conv)}


# ---------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------
def beta_stats(betas: Sequence[float]) -> Dict[str, float]:
    b = np.asarray(betas, dtype=float)
    return {"beta_mean": float(b.mean()), "beta_std": float(b.std()),
            "beta_max": float(b.max()) if b.size else float("nan")}


# ---------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------
def mean_std(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1) if v.size > 1 else 0.0),
            "n": int(v.size)}


def gap_over_se(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Separation criterion used elsewhere in the portfolio: the difference in
    means divided by the standard error of that difference. Values below 2
    should not be reported as an improvement.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size) + 1e-12
    return float(abs(a.mean() - b.mean()) / se)
