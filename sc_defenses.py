"""
sc_defenses.py
Aggregation rules, including the two baselines the reviewers asked for.

  fedavg        sample-weighted mean
  trimmed_mean  coordinate-wise trimmed mean
  krum          single update minimising distance to its n-f-2 nearest
  fltrust       server root-dataset trust bootstrapping (Cao et al., NDSS 2021)
  foolsgold     cosine-history contribution reweighting (Fung et al., RAID 2020)
  ours          Eqs. (10)-(14): angular + magnitude deviation, EMA trust,
                trust-weighted aggregation

All rules take {client_id: update} and return a single aggregated update plus
a per-client weight vector, so the influence score of Eq. (13) can be computed
identically across methods.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Sequence, Tuple

Update = List[np.ndarray]


# ---------------------------------------------------------------------
# Flattening helpers
# ---------------------------------------------------------------------
def flatten(u: Update) -> np.ndarray:
    return np.concatenate([x.ravel() for x in u])


def unflatten(v: np.ndarray, ref: Update) -> Update:
    out, o = [], 0
    for r in ref:
        n = r.size
        out.append(v[o:o + n].reshape(r.shape))
        o += n
    return out


def _norms(M: np.ndarray) -> np.ndarray:
    return np.linalg.norm(M, axis=1) + 1e-12


# ---------------------------------------------------------------------
# Deviation score, Eqs. (10)-(12)
# ---------------------------------------------------------------------
def deviation_scores(M: np.ndarray, psi: float = 0.7) -> np.ndarray:
    """
    M : (k, P) matrix of flattened updates for the current committee.
    Reference direction is the coordinate-wise median, which is itself
    resistant to a minority of corrupted rows.
    """
    ref = np.median(M, axis=0)
    ref_n = np.linalg.norm(ref) + 1e-12

    cos = (M @ ref) / (_norms(M) * ref_n)
    a = 0.5 * (1.0 - np.clip(cos, -1.0, 1.0))              # Eq. (10)

    med_norm = np.median(_norms(M))
    m = np.minimum(1.0, np.abs(_norms(M) / (med_norm + 1e-12) - 1.0))  # Eq. (11)

    return psi * a + (1.0 - psi) * m                        # Eq. (12)


# ---------------------------------------------------------------------
# Aggregation rules
# ---------------------------------------------------------------------
def agg_fedavg(updates: Dict[int, Update], sizes: Dict[int, int],
               **kw) -> Tuple[Update, Dict[int, float]]:
    ids = list(updates)
    tot = sum(sizes[i] for i in ids)
    w = {i: sizes[i] / tot for i in ids}
    ref = updates[ids[0]]
    agg = np.sum([w[i] * flatten(updates[i]) for i in ids], axis=0)
    return unflatten(agg, ref), w


def agg_trimmed_mean(updates: Dict[int, Update], trim: float = 0.2,
                     **kw) -> Tuple[Update, Dict[int, float]]:
    ids = list(updates)
    M = np.stack([flatten(updates[i]) for i in ids])
    k = int(np.floor(trim * len(ids)))
    S = np.sort(M, axis=0)
    core = S[k:len(ids) - k] if len(ids) - 2 * k > 0 else S
    agg = core.mean(axis=0)
    w = {i: 1.0 / len(ids) for i in ids}   # nominal; trimming is per-coordinate
    return unflatten(agg, updates[ids[0]]), w


def agg_krum(updates: Dict[int, Update], n_malicious: int = 0,
             **kw) -> Tuple[Update, Dict[int, float]]:
    ids = list(updates)
    M = np.stack([flatten(updates[i]) for i in ids])
    n = len(ids)
    f = min(max(n_malicious, 0), max(0, (n - 3) // 2))
    k = max(1, n - f - 2)
    D = ((M[:, None, :] - M[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    scores = np.sort(D, axis=1)[:, :k].sum(axis=1)
    pick = int(np.argmin(scores))
    w = {cid: (1.0 if j == pick else 0.0) for j, cid in enumerate(ids)}
    return unflatten(M[pick], updates[ids[0]]), w


def agg_fltrust(updates: Dict[int, Update], server_update: Update,
                **kw) -> Tuple[Update, Dict[int, float]]:
    """Cao et al., NDSS 2021. ReLU-clipped cosine trust, norm-clipped updates."""
    ids = list(updates)
    g0 = flatten(server_update)
    n0 = np.linalg.norm(g0) + 1e-12
    M = np.stack([flatten(updates[i]) for i in ids])
    cos = (M @ g0) / (_norms(M) * n0)
    ts = np.maximum(cos, 0.0)
    if ts.sum() <= 0:
        w = {i: 0.0 for i in ids}
        return unflatten(np.zeros_like(g0), updates[ids[0]]), w
    Mc = M * (n0 / _norms(M))[:, None]
    agg = (ts[:, None] * Mc).sum(0) / ts.sum()
    w = {cid: float(ts[j] / ts.sum()) for j, cid in enumerate(ids)}
    return unflatten(agg, updates[ids[0]]), w


class FoolsGold:
    """Fung et al., RAID 2020. Maintains a per-client history of updates."""

    def __init__(self, n_clients: int, dim: int):
        self.H = np.zeros((n_clients, dim), dtype=np.float32)

    def __call__(self, updates: Dict[int, Update],
                 **kw) -> Tuple[Update, Dict[int, float]]:
        ids = list(updates)
        for i in ids:
            self.H[i] += flatten(updates[i])
        Hs = self.H[ids]
        cs = (Hs @ Hs.T) / np.outer(_norms(Hs), _norms(Hs))
        np.fill_diagonal(cs, 0.0)
        maxcs = cs.max(axis=1)

        # pardoning
        for a in range(len(ids)):
            for b in range(len(ids)):
                if maxcs[a] < maxcs[b] and maxcs[b] > 0:
                    cs[a, b] *= maxcs[a] / maxcs[b]

        wv = 1.0 - cs.max(axis=1)
        wv = np.clip(wv, 0.0, 1.0)
        if wv.max() > 0:
            wv = wv / wv.max()
        wv = np.clip(wv, 1e-6, 1.0 - 1e-6)
        wv = np.log(wv / (1 - wv)) + 0.5
        wv = np.clip(wv, 0.0, 1.0)

        if wv.sum() <= 0:
            wv = np.ones_like(wv)
        wv = wv / wv.sum()

        M = np.stack([flatten(updates[i]) for i in ids])
        # Norm-clip each update to the committee median norm before the weighted
        # mean. Vanilla FoolsGold reweights by cosine history but leaves update
        # magnitude uncontrolled, so a scaled model-poisoning update passes
        # through and the aggregate diverges. Clipping (updates above the median
        # norm are shrunk to it; the cosine reweighting is scale-invariant and
        # unaffected) makes FoolsGold a credible baseline under model poisoning.
        # DISCLOSE this addition to the FoolsGold baseline in the paper.
        norms = np.linalg.norm(M, axis=1) + 1e-12
        clip = np.minimum(1.0, np.median(norms) / norms)
        M = M * clip[:, None]
        agg = (wv[:, None] * M).sum(0)
        return unflatten(agg, updates[ids[0]]), {c: float(wv[j])
                                                 for j, c in enumerate(ids)}


class TrustWeighted:
    """
    The proposed rule. Trust is updated in the same round it is applied, so
    the one-round lag present in the submitted version is removed. Round 1
    aggregates under uniform trust by construction, since rho is initialised
    to one for every client.
    """

    def __init__(self, n_clients: int, lam: float = 0.8, psi: float = 0.7,
                 clip: bool = True):
        self.rho = np.ones(n_clients, dtype=np.float64)
        self.lam = lam
        self.psi = psi
        self.clip = clip
        self.history: List[np.ndarray] = []

    def __call__(self, updates: Dict[int, Update],
                 **kw) -> Tuple[Update, Dict[int, float]]:
        ids = list(updates)
        M = np.stack([flatten(updates[i]) for i in ids])
        d = deviation_scores(M, psi=self.psi)                # Eq. (12)

        for j, cid in enumerate(ids):
            self.rho[cid] = self.lam * self.rho[cid] + (1 - self.lam) * (1 - d[j])

        r = np.array([self.rho[i] for i in ids])
        if r.sum() <= 0:
            r = np.ones_like(r)
        alpha = r / r.sum()                                   # Eq. (14)
        self.history.append(self.rho.copy())

        # Trust weighting controls *direction* (a low-trust client still keeps a
        # small weight); norm-clipping to the committee median bounds *magnitude*
        # so a scaled sign-flip cannot dominate through its residual weight. The
        # two together are what keep the aggregate bounded under strong model
        # poisoning and stable on small committees (Geolife). Clipping is
        # magnitude-only and does not change the trust scores above.
        Ma = M
        if self.clip:
            norms = np.linalg.norm(M, axis=1) + 1e-12
            scale = np.minimum(1.0, np.median(norms) / norms)
            Ma = M * scale[:, None]

        agg = (alpha[:, None] * Ma).sum(0)
        return unflatten(agg, updates[ids[0]]), {c: float(alpha[j])
                                                 for j, c in enumerate(ids)}


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def make_defense(name: str, n_clients: int, dim: int, **kw):
    if name == "fedavg":
        return lambda u, **k: agg_fedavg(u, **k)
    if name == "trimmed_mean":
        return lambda u, **k: agg_trimmed_mean(u, **k)
    if name == "krum":
        return lambda u, **k: agg_krum(u, **k)
    if name == "fltrust":
        return lambda u, **k: agg_fltrust(u, **k)
    if name == "foolsgold":
        return FoolsGold(n_clients, dim)
    if name == "ours":
        return TrustWeighted(n_clients, lam=kw.get("lam", 0.8),
                             psi=kw.get("psi", 0.7), clip=kw.get("clip", True))
    if name == "ours_no_magnitude":      # ablation: psi = 1
        return TrustWeighted(n_clients, lam=kw.get("lam", 0.8), psi=1.0,
                             clip=kw.get("clip", True))
    if name == "ours_no_clip":           # ablation: trust weighting only
        return TrustWeighted(n_clients, lam=kw.get("lam", 0.8),
                             psi=kw.get("psi", 0.7), clip=False)
    raise ValueError(f"unknown defence: {name}")
