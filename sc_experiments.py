"""
sc_experiments.py
Runs the four experiment blocks the revised manuscript needs.

  grid       main results: methods x attacks x ratios x datasets x seeds
  selection  Proposition 1: uniform vs vrf vs biased coordinator
  phase      breakdown-point sweep in beta, per method per attack
  overhead   measured per-round cost

Usage
-----
  python sc_experiments.py grid      --datasets nyc_taxi yelp geolife --seeds 5
  python sc_experiments.py selection --datasets nyc_taxi --seeds 5
  python sc_experiments.py phase     --datasets nyc_taxi --seeds 5
  python sc_experiments.py overhead

Results are written as JSON under results/<block>/, one file per
configuration, then consumed by sc_report.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, List, Sequence

import sc_tasks as T
import sc_attacks as A
import sc_defenses as D
import sc_metrics as Me
import sc_selection as Sel

RESULTS_DIR = os.environ.get("SC_RESULTS_DIR", "results")


# ---------------------------------------------------------------------
# Model: matches FLNeuralNet from the Cohen's Kappa project
# ---------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def get_weights(m: nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in m.parameters()]


def set_weights(m: nn.Module, w: Sequence[np.ndarray]) -> None:
    with torch.no_grad():
        for p, v in zip(m.parameters(), w):
            p.copy_(torch.tensor(v, dtype=p.dtype))


def local_train(model: nn.Module, X: np.ndarray, y: np.ndarray,
                epochs: int, lr: float, batch: int,
                device: str) -> List[np.ndarray]:
    """Returns the update Delta w = w_local - w_global."""
    w0 = get_weights(model)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    w1 = get_weights(model)
    return [b - a for a, b in zip(w0, w1)]


@torch.no_grad()
def evaluate(model: nn.Module, task: T.FederatedTask, device: str,
             split: str = "test"):
    model.eval()
    preds, trues = {}, {}
    for c in task.clients:
        X = c.X_val if split == "val" else c.X_test
        y = c.y_val if split == "val" else c.y_test
        if X is None or len(X) == 0:
            continue
        p = model(torch.tensor(X, device=device)).cpu().numpy()
        preds[c.region_id] = p
        trues[c.region_id] = y
    return preds, trues


# ---------------------------------------------------------------------
# Core federated run
# ---------------------------------------------------------------------
DEFAULTS = dict(rounds=40, tau=0.2, local_epochs=2, lr=1e-3, batch=64,
                server_lr=1.0, lam=0.8, psi=0.7, clip=True, trim=0.2,
                root_size=200)


def run_fl(task: T.FederatedTask, method: str, attack: str, ratio: float,
           seed: int, selection: str = "vrf", cfg: Dict | None = None,
           forced_beta: float | None = None, eval_split: str = "test") -> Dict:
    cfg = {**DEFAULTS, **(cfg or {})}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n = len(task.clients)
    model = MLP(task.in_dim).to(device)
    dim = sum(p.numel() for p in model.parameters())

    malicious = A.assign_malicious(n, ratio, rng)
    mal_set = set(malicious)
    keys = Sel.make_keys(n, rng)
    defense = D.make_defense(method, n, dim, lam=cfg["lam"], psi=cfg["psi"],
                             clip=cfg["clip"])
    infl = Me.InfluenceTracker()
    betas: List[float] = []

    # local data, with data poisoning applied once at the data stage
    data = {}
    for c in task.clients:
        y = c.y_train
        if attack == "data" and c.client_id in mal_set:
            y = A.poison_data(y)
        data[c.client_id] = (c.X_train, y)

    # FLTrust root set: held out from the server's own reference sample
    root = None
    if method == "fltrust":
        Xs = np.concatenate([c.X_train for c in task.clients])
        ys = np.concatenate([c.y_train for c in task.clients])
        idx = rng.choice(len(Xs), size=min(cfg["root_size"], len(Xs)),
                         replace=False)
        root = (Xs[idx], ys[idx])

    t_server = 0.0
    for t in range(1, cfg["rounds"] + 1):
        gw = get_weights(model)
        seed_t = Sel.bind_seed(gw, t)

        if forced_beta is not None:
            k = max(2, int(round(cfg["tau"] * n)))
            n_mal = min(len(malicious), int(round(forced_beta * k)))
            pick = list(rng.choice(malicious, size=n_mal, replace=False)) if n_mal else []
            honest = [i for i in range(n) if i not in mal_set]
            pick += list(rng.choice(honest, size=k - n_mal, replace=False))
            selected = sorted(int(i) for i in pick)
        else:
            selected = Sel.select(selection, n, keys, malicious,
                                  cfg["tau"], seed_t, rng)

        betas.append(Sel.realised_beta(selected, malicious))

        updates: Dict[int, List[np.ndarray]] = {}
        sizes: Dict[int, int] = {}
        for cid in selected:
            X, y = data[cid]
            if len(X) < 2:
                continue
            local = MLP(task.in_dim).to(device)
            set_weights(local, gw)
            updates[cid] = local_train(local, X, y, cfg["local_epochs"],
                                       cfg["lr"], cfg["batch"], device)
            sizes[cid] = len(X)
        if not updates:
            continue

        med_norm = float(np.median([np.linalg.norm(D.flatten(u))
                                    for u in updates.values()]))
        updates = A.apply_update_attack(attack, updates, malicious, med_norm)

        kwargs = dict(sizes=sizes)
        if method == "krum":
            kwargs["n_malicious"] = max(1, int(round(ratio * len(updates))))
        if method == "trimmed_mean":
            kwargs["trim"] = cfg["trim"]
        if method == "fltrust":
            sm = MLP(task.in_dim).to(device)
            set_weights(sm, gw)
            kwargs["server_update"] = local_train(sm, root[0], root[1],
                                                  cfg["local_epochs"],
                                                  cfg["lr"], cfg["batch"], device)

        t0 = time.perf_counter()
        agg, weights = defense(updates, **kwargs)
        t_server += time.perf_counter() - t0

        norms = {i: float(np.linalg.norm(D.flatten(u))) for i, u in updates.items()}
        infl.update(Me.round_influence(weights, norms))

        set_weights(model, [w + cfg["server_lr"] * a for w, a in zip(gw, agg)])

    preds, trues = evaluate(model, task, device, split=eval_split)
    rm = Me.per_region_mae(preds, trues)
    surge = [c.region_id for c in task.clients if c.is_surge_zone]
    nonsurge = [c.region_id for c in task.clients if not c.is_surge_zone]

    hist = getattr(defense, "history", [])
    out = {
        "dataset": task.name, "target_kind": task.target_kind,
        "method": method, "attack": attack, "ratio": ratio, "seed": seed,
        "selection": selection, "forced_beta": forced_beta,
        "mae": Me.global_mae(preds, trues),
        "fairness_all": Me.fairness_score(rm),
        "fairness_surge": Me.fairness_score(rm, [r for r in surge if r in rm]),
        "fairness_nonsurge": Me.fairness_score(rm, [r for r in nonsurge if r in rm]),
        "influence": infl.group_means(malicious),
        "trust": Me.trust_separation(hist, malicious),
        "server_time_ms_per_round": 1e3 * t_server / cfg["rounds"],
        "n_clients": n, "n_malicious": len(malicious),
        **Me.beta_stats(betas),
    }
    return out


# ---------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------
METHODS = ["fedavg", "trimmed_mean", "krum", "fltrust", "foolsgold", "ours"]
ATTACKS = ["data", "model", "collusion"]
RATIOS = [0.10, 0.20, 0.40]


def _save(block: str, tag: str, obj) -> None:
    d = os.path.join(RESULTS_DIR, block)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{tag}.json"), "w") as f:
        json.dump(obj, f, indent=2)


def _load_tasks(names: Sequence[str], loader_kwargs: Dict) -> Dict[str, T.FederatedTask]:
    return {n: T.build_task(n, **loader_kwargs.get(n, {})) for n in names}


def block_grid(tasks, seeds: int) -> None:
    for name, task in tasks.items():
        for method in METHODS:
            for attack in ATTACKS + ["none"]:
                ratios = [0.0] if attack == "none" else RATIOS
                for ratio in ratios:
                    runs = [run_fl(task, method, attack, ratio, s)
                            for s in range(seeds)]
                    _save("grid", f"{name}_{method}_{attack}_{int(ratio*100)}",
                          runs)
                    print(f"[grid] {name} {method} {attack} {ratio:.2f} "
                          f"mae={np.mean([r['mae'] for r in runs]):.4f}")


def block_selection(tasks, seeds: int, ratio: float = 0.20) -> None:
    """
    Proposition 1. Expect uniform and vrf to coincide within seed variance;
    that agreement is the predicted result and should be reported as such.
    """
    for name, task in tasks.items():
        for regime in ["uniform", "vrf", "biased"]:
            runs = [run_fl(task, "ours", "model", ratio, s, selection=regime)
                    for s in range(seeds)]
            _save("selection", f"{name}_{regime}", runs)
            print(f"[selection] {name} {regime} "
                  f"beta={np.mean([r['beta_mean'] for r in runs]):.3f} "
                  f"mae={np.mean([r['mae'] for r in runs]):.4f}")


def block_phase(tasks, seeds: int,
                betas=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                       0.35, 0.40, 0.45, 0.50)) -> None:
    """
    Breakdown-point sweep. beta is forced directly rather than induced by a
    ratio, so the committee composition is controlled rather than sampled.
    """
    for name, task in tasks.items():
        for method in METHODS:
            for attack in ["model", "collusion"]:
                for b in betas:
                    runs = [run_fl(task, method, attack, ratio=0.5, seed=s,
                                   forced_beta=b) for s in range(seeds)]
                    _save("phase",
                          f"{name}_{method}_{attack}_b{int(b*100):03d}", runs)
                    print(f"[phase] {name} {method} {attack} beta={b:.2f} "
                          f"mae={np.mean([r['mae'] for r in runs]):.4f}")


def block_overhead() -> None:
    _save("overhead", "proof_cost", Sel.measure_proof_cost())
    print("[overhead]", Sel.measure_proof_cost())


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("block", choices=["grid", "selection", "phase", "overhead"])
    ap.add_argument("--datasets", nargs="+", default=["nyc_taxi"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--loader-kwargs", type=str, default="{}",
                    help="JSON dict, e.g. '{\"nyc_taxi\": {\"path\": \"...\"}}'")
    args = ap.parse_args()

    if args.block == "overhead":
        block_overhead()
        return

    tasks = _load_tasks(args.datasets, json.loads(args.loader_kwargs))
    for n, t in tasks.items():
        print(f"{n}: {len(t.clients)} clients, in_dim={t.in_dim}, "
              f"target={t.target_kind}")

    if args.block == "grid":
        block_grid(tasks, args.seeds)
    elif args.block == "selection":
        block_selection(tasks, args.seeds)
    elif args.block == "phase":
        block_phase(tasks, args.seeds)


if __name__ == "__main__":
    main()
