"""
sc_run_parallel.py
Parallel, resumable driver for the grid / selection / phase blocks.

Same experiments and same output files as sc_experiments.py, but each
configuration runs in its own process so all CPU cores are used. run_fl is
deterministic per seed, so results are identical to the serial blocks -- this
only changes scheduling, not the numbers.

  python sc_run_parallel.py phase     --datasets nyc_taxi --seeds 5 --jobs 24 --resume
  python sc_run_parallel.py grid      --datasets nyc_taxi foursquare geolife yelp --seeds 5 --jobs 24 --resume
  python sc_run_parallel.py selection --datasets nyc_taxi --seeds 5 --jobs 8  --resume

--resume skips any configuration whose JSON already exists and parses, so a
killed run can be restarted and it picks up where it stopped. Existing files
(e.g. a completed grid) are never overwritten under --resume.
"""

from __future__ import annotations

# Pin per-process math threads BEFORE numpy/torch load: one process per core,
# one thread per process, avoids 28*20 thread oversubscription.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

RESULTS_DIR = os.environ.get("SC_RESULTS_DIR", "results")
_TASKS: dict = {}          # per-worker cache: dataset name -> FederatedTask


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------
def _init(tasks: dict):
    """Runs once per worker: single-thread torch, receive the prebuilt tasks.

    Tasks are built ONCE in the parent and passed in (pickled to each worker),
    rather than each worker re-reading the multi-GB parquet -- 24 concurrent
    full-parquet reads exhaust memory and crash pyarrow's to-pandas step.
    """
    import torch
    torch.set_num_threads(1)
    global _TASKS
    _TASKS = tasks


def _tag(block: str, job: dict) -> str:
    name = job["dataset"]
    if block == "grid":
        return f"{name}_{job['method']}_{job['attack']}_{int(job['ratio']*100)}"
    if block == "phase":
        return f"{name}_{job['method']}_{job['attack']}_b{int(job['beta']*100):03d}"
    if block == "selection":
        return f"{name}_{job['regime']}"
    raise ValueError(block)


def _path(block: str, tag: str) -> str:
    return os.path.join(RESULTS_DIR, block, f"{tag}.json")


def _run_job(args) -> str:
    block, job, seeds = args
    import sc_experiments as E
    task = _TASKS[job["dataset"]]
    runs = []
    for s in range(seeds):
        if block == "grid":
            r = E.run_fl(task, job["method"], job["attack"], job["ratio"], s)
        elif block == "phase":
            r = E.run_fl(task, job["method"], job["attack"], ratio=0.5, seed=s,
                         forced_beta=job["beta"])
        elif block == "selection":
            r = E.run_fl(task, "ours", "model", job["ratio"], s,
                         selection=job["regime"])
        runs.append(r)

    tag = _tag(block, job)
    d = os.path.join(RESULTS_DIR, block)
    os.makedirs(d, exist_ok=True)
    with open(_path(block, tag), "w") as f:
        json.dump(runs, f, indent=2)
    mae = sum(r["mae"] for r in runs) / len(runs)
    return f"{block} {tag} mae={mae:.4f}"


# ---------------------------------------------------------------------
# Job enumeration
# ---------------------------------------------------------------------
METHODS = ["fedavg", "trimmed_mean", "krum", "fltrust", "foolsgold", "ours"]
ATTACKS = ["data", "model", "collusion"]
RATIOS = [0.10, 0.20, 0.40]
PHASE_BETAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
               0.35, 0.40, 0.45, 0.50)
PHASE_ATTACKS = ["model", "collusion"]


def _jobs(block: str, datasets, methods=None, attacks=None):
    methods = methods or METHODS
    grid_attacks = attacks or (ATTACKS + ["none"])
    out = []
    for name in datasets:
        if block == "grid":
            for m in methods:
                for a in grid_attacks:
                    for rr in ([0.0] if a == "none" else RATIOS):
                        out.append({"dataset": name, "method": m,
                                    "attack": a, "ratio": rr})
        elif block == "phase":
            for m in methods:
                for a in PHASE_ATTACKS:
                    for b in PHASE_BETAS:
                        out.append({"dataset": name, "method": m,
                                    "attack": a, "beta": b})
        elif block == "selection":
            for regime in ["uniform", "vrf", "biased"]:
                out.append({"dataset": name, "regime": regime, "ratio": 0.20})
    return out


def _already_done(block: str, job: dict, seeds: int) -> bool:
    p = _path(block, _tag(block, job))
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        return isinstance(data, list) and len(data) >= seeds
    except (json.JSONDecodeError, OSError):
        return False   # truncated/partial -> redo


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("block", choices=["grid", "selection", "phase"])
    ap.add_argument("--datasets", nargs="+", default=["nyc_taxi"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restrict to these methods (default: all)")
    ap.add_argument("--attacks", nargs="+", default=None,
                    help="grid only: restrict to these attacks (e.g. model_stealth)")
    ap.add_argument("--loader-kwargs", type=str, default="{}",
                    help="JSON, e.g. '{\"nyc_taxi\": {\"n_days\": 14}}'")
    args = ap.parse_args()

    loader_kwargs = json.loads(args.loader_kwargs)
    dataset_kwargs = {n: loader_kwargs.get(n, {}) for n in args.datasets}

    jobs = _jobs(args.block, args.datasets, args.methods, args.attacks)
    if args.resume:
        pending = [j for j in jobs if not _already_done(args.block, j, args.seeds)]
    else:
        pending = jobs
    skipped = len(jobs) - len(pending)

    print(f"[{args.block}] {len(jobs)} configs, {skipped} already done, "
          f"{len(pending)} to run on {args.jobs} workers "
          f"(seeds={args.seeds})")
    if not pending:
        print("nothing to do."); return

    # Build each needed task ONCE in the parent (one parquet read), then hand
    # the prebuilt tasks to the workers -- avoids 24 concurrent parquet reads.
    import sc_tasks as T
    needed = sorted({j["dataset"] for j in pending})
    tasks = {n: T.build_task(n, **dataset_kwargs.get(n, {})) for n in needed}
    for n in needed:
        print(f"  built {n}: {len(tasks[n].clients)} clients")

    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs,
                             initializer=_init,
                             initargs=(tasks,)) as ex:
        futs = {ex.submit(_run_job, (args.block, j, args.seeds)): j
                for j in pending}
        for fut in as_completed(futs):
            done += 1
            msg = fut.result()
            el = time.perf_counter() - t0
            eta = el / done * (len(pending) - done)
            print(f"[{done}/{len(pending)}] {msg}  "
                  f"(elapsed {el/60:.1f}m, eta {eta/60:.1f}m)")

    print(f"[{args.block}] complete in {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
