"""
sc_attacks.py
The three attack families named in Section IV-B.

data poisoning   : the client's local target is inflated in high-demand bins,
                   so the learned demand-to-surge mapping over-reports scarcity
model poisoning  : the submitted update is sign-flipped and scaled
collusion        : malicious clients agree on a common direction before
                   submitting, concentrated on a target region

Each returns a modified update or dataset; nothing else in the pipeline is
aware of which clients are adversarial, so the defences see only submissions.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Sequence, Tuple

Update = List[np.ndarray]


# ---------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------
def assign_malicious(n_clients: int, ratio: float,
                     rng: np.random.Generator,
                     target_regions: Sequence[int] | None = None) -> List[int]:
    """
    Malicious clients are drawn uniformly unless target_regions is given, in
    which case they are drawn preferentially from those regions to model a
    geographically concentrated adversary.
    """
    k = int(round(ratio * n_clients))
    if k == 0:
        return []
    if target_regions:
        pool = list(target_regions)
        if len(pool) >= k:
            return sorted(rng.choice(pool, size=k, replace=False).tolist())
        rest = [i for i in range(n_clients) if i not in set(pool)]
        extra = rng.choice(rest, size=k - len(pool), replace=False).tolist()
        return sorted(pool + extra)
    return sorted(rng.choice(n_clients, size=k, replace=False).tolist())


# ---------------------------------------------------------------------
# Data poisoning
# ---------------------------------------------------------------------
def poison_data(y: np.ndarray, strength: float = 2.0,
                quantile: float = 0.6) -> np.ndarray:
    """
    Scarcity inflation. Targets above the given quantile are multiplied by
    `strength`, which biases the mapping from demand-supply features to the
    surge proxy upward without producing obviously out-of-range labels.
    """
    y = y.copy()
    thr = np.quantile(y, quantile)
    mask = (y >= thr).ravel()
    y[mask] = y[mask] * strength
    return y


# ---------------------------------------------------------------------
# Model poisoning
# ---------------------------------------------------------------------
def poison_model(update: Update, scale: float = 5.0,
                 flip: bool = True) -> Update:
    sign = -1.0 if flip else 1.0
    return [sign * scale * u for u in update]


def poison_model_stealth(update: Update, reference_norm: float,
                         scale: float = 1.0) -> Update:
    """
    Norm-matched sign flip. Direction is reversed but magnitude is held at
    the honest median, so the magnitude term m_i^t of Eq. (11) does not fire
    and only the angular term a_i^t responds. Used for the detection-floor
    measurement in the limitations subsection.
    """
    n = np.sqrt(sum(float(np.sum(u ** 2)) for u in update)) + 1e-12
    k = scale * reference_norm / n
    return [-k * u for u in update]


# ---------------------------------------------------------------------
# Collusion
# ---------------------------------------------------------------------
def collude(updates: Dict[int, Update], malicious: Sequence[int],
            scale: float = 3.0) -> Dict[int, Update]:
    """
    Colluding clients replace their submissions with a shared scaled mean of
    their own updates. Agreement raises their apparent consensus with each
    other, which is the case a median reference direction is meant to resist
    and which fails once they hold the majority of the committee.
    """
    present = [i for i in malicious if i in updates]
    if len(present) < 2:
        return updates
    stacked = [np.mean([updates[i][l] for i in present], axis=0)
               for l in range(len(updates[present[0]]))]
    common = [-scale * s for s in stacked]
    for i in present:
        updates[i] = [c.copy() for c in common]
    return updates


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------
def apply_update_attack(attack: str, updates: Dict[int, Update],
                        malicious: Sequence[int],
                        median_norm: float | None = None) -> Dict[int, Update]:
    if attack == "data":
        return updates                      # applied at the data stage
    if attack == "model":
        for i in malicious:
            if i in updates:
                updates[i] = poison_model(updates[i])
        return updates
    if attack == "model_stealth":
        for i in malicious:
            if i in updates:
                updates[i] = poison_model_stealth(updates[i], median_norm or 1.0)
        return updates
    if attack == "collusion":
        return collude(updates, malicious)
    if attack == "none":
        return updates
    raise ValueError(f"unknown attack: {attack}")
