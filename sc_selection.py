"""
sc_selection.py
The three selection regimes of Proposition 1.

  uniform  : honest coordinator, sampling without replacement   -> E[beta] = alpha
  vrf      : client self-selection, seed bound to H(w^{t-1}||t) -> E[beta] = alpha
  biased   : coordinator maximises beta                         -> beta = min(1, alpha/tau)

The VRF here is an HMAC-SHA256 construction. It reproduces the selection
*distribution* of a real VRF, which is what Proposition 1 concerns, but it is
a PRF and provides no public verifiability. Overhead figures reported in the
paper must come from measure_proof_cost(), which uses Ed25519 signatures as
a stand-in for real proof generation and verification cost. State this
substitution in the implementation-details subsection.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import numpy as np
from typing import Dict, List, Sequence

try:
    from nacl.signing import SigningKey
    _HAVE_NACL = True
except ImportError:  # pragma: no cover
    _HAVE_NACL = False


# ---------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------
def bind_seed(global_weights: Sequence[np.ndarray], round_t: int) -> bytes:
    """
    s_t = H(w^{t-1} || t).

    A coordinator-chosen seed permits grinding: the server resamples s_t
    until the eligible set is favourable. Binding the seed to the previous
    global model means a favourable committee also requires a favourable
    model, which the coordinator does not control unilaterally.
    """
    h = hashlib.sha256()
    for arr in global_weights:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    h.update(int(round_t).to_bytes(8, "big"))
    return h.digest()


# ---------------------------------------------------------------------
# Key material
# ---------------------------------------------------------------------
def make_keys(n_clients: int, rng: np.random.Generator) -> Dict[int, bytes]:
    return {i: rng.bytes(32) for i in range(n_clients)}


def vrf_eval(sk: bytes, seed: bytes) -> float:
    """Returns r_i^t in [0,1)."""
    digest = hmac.new(sk, seed, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


# ---------------------------------------------------------------------
# Selection regimes
# ---------------------------------------------------------------------
def select_uniform(n_clients: int, tau: float,
                   rng: np.random.Generator) -> List[int]:
    k = max(1, int(round(tau * n_clients)))
    return sorted(rng.choice(n_clients, size=k, replace=False).tolist())


def select_vrf(keys: Dict[int, bytes], seed: bytes, tau: float) -> List[int]:
    """
    Eligibility is r_i^t <= tau, evaluated independently per client. The
    committee size is therefore Binomial(N, tau) rather than fixed; this is
    the standard behaviour of threshold self-selection and is why the
    realised size is reported alongside beta.
    """
    return sorted(i for i, sk in keys.items() if vrf_eval(sk, seed) <= tau)


def select_biased(n_clients: int, malicious: Sequence[int], tau: float,
                  rng: np.random.Generator) -> List[int]:
    """
    Coordinator fills the committee from the malicious pool first, then pads
    with honest clients. Realises beta = min(1, alpha/tau) of Proposition 1(iii).
    """
    k = max(1, int(round(tau * n_clients)))
    mal = list(malicious)
    rng.shuffle(mal)
    chosen = mal[:k]
    if len(chosen) < k:
        honest = [i for i in range(n_clients) if i not in set(malicious)]
        rng.shuffle(honest)
        chosen += honest[: k - len(chosen)]
    return sorted(chosen)


def select(regime: str, n_clients: int, keys: Dict[int, bytes],
           malicious: Sequence[int], tau: float, seed: bytes,
           rng: np.random.Generator) -> List[int]:
    if regime == "uniform":
        return select_uniform(n_clients, tau, rng)
    if regime == "vrf":
        s = select_vrf(keys, seed, tau)
        return s if s else select_uniform(n_clients, tau, rng)
    if regime == "biased":
        return select_biased(n_clients, malicious, tau, rng)
    raise ValueError(f"unknown selection regime: {regime}")


def realised_beta(selected: Sequence[int], malicious: Sequence[int]) -> float:
    if len(selected) == 0:
        return 0.0
    mal = set(malicious)
    return sum(1 for i in selected if i in mal) / len(selected)


# ---------------------------------------------------------------------
# Overhead measurement (Table: tab:overhead)
# ---------------------------------------------------------------------
def measure_proof_cost(n_trials: int = 200) -> Dict[str, float]:
    """
    Ed25519 sign/verify as a cost stand-in for VRF prove/verify. Returns
    milliseconds per operation and proof size in bytes. Without PyNaCl the
    entry is returned as unavailable rather than estimated.
    """
    if not _HAVE_NACL:
        return {"prove_ms": float("nan"), "verify_ms": float("nan"),
                "proof_bytes": float("nan"), "available": 0.0}

    sk = SigningKey.generate()
    vk = sk.verify_key
    msg = b"x" * 32

    t0 = time.perf_counter()
    for _ in range(n_trials):
        sig = sk.sign(msg)
    t1 = time.perf_counter()
    for _ in range(n_trials):
        vk.verify(sig)
    t2 = time.perf_counter()

    return {"prove_ms": 1e3 * (t1 - t0) / n_trials,
            "verify_ms": 1e3 * (t2 - t1) / n_trials,
            "proof_bytes": float(len(sig.signature)),
            "available": 1.0}
