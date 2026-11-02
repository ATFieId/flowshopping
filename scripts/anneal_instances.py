"""Simulated quantum annealing of every instance in a folder.

    python scripts/anneal_instances.py                         # instances/ -> figures/anneal/
    python scripts/anneal_instances.py my_folder -o out --T 5 20 100

For each instance ``<name>.json`` the script builds ``H_P = H_pen + H_obj`` (report
Equation 2), diagonalises ``H(G) = H_P - G sum sigma^x`` (Equation 1) along G, and
runs the anneal ``G(t) = G_max (1 - t / T)`` by exact time evolution. It writes
into ``<out>/<name>/``:

    gantt.png / .pdf    standard vs no-wait Gantt chart of the optimal sequence
    anneal.png          full spectrum (n = 3) or lowest levels (n = 4), and success probability
    qubo_heatmap.png    QUBO matrix of H_P
    summary.json        P, optimum, minimum gap, final success probabilities

If the folder holds no instances, the five DEFAULT_INSTANCES are generated into it.
Run time on a PC: seconds for n = 3, a few minutes for n = 4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import generator, nowait, plotting, quantum, qubo, spectrum  # noqa: E402

# Five instances from the generator (unique optimum, processing times in [1, 9]).
DEFAULT_INSTANCES = [
    dict(n=3, m=2, seed=11),
    dict(n=3, m=3, seed=12),
    dict(n=3, m=4, seed=13),
    dict(n=4, m=3, seed=14),
    dict(n=4, m=4, seed=15),
]


def default_P(spec: spectrum.Spectrum) -> float:
    """Smallest integer above the sufficient bound ``H*_obj / 2`` (Proposition 5)."""
    return float(np.floor(spec.sufficient_P()) + 1)


def merge_sweeps(a: dict, b: dict) -> dict:
    """Union of two sweeps of the same Hamiltonian, sorted by decreasing G."""
    g = np.concatenate([a["gammas"], b["gammas"]])
    order = np.argsort(-g, kind="stable")
    return {"gammas": g[order],
            "levels": np.concatenate([a["levels"], b["levels"]])[order],
            "weights": {k: np.concatenate([a["weights"][k], b["weights"][k]])[order] for k in a["weights"]}}


def run_instance(inst, out: str, P: float | None, gamma_factor: float, Ts, dt: float,
                 levels: int, points: int | None) -> dict:
    t0 = time.perf_counter()
    os.makedirs(out, exist_ok=True)
    n, N = inst.n, inst.n ** 2
    spec = spectrum.enumerate_spectrum(inst)
    cmax, opt_seqs = nowait.optimum(inst)
    P = P if P is not None else default_P(spec)
    H = qubo.problem_qubo(inst, P)
    tf = quantum.TransverseFieldHamiltonian(H)

    track = {"optimal schedule": [quantum.basis_index(qubo.encode(s)) for s in opt_seqs],
             "any feasible schedule": np.flatnonzero(spec.feasible).tolist()}
    g_max = gamma_factor * P
    dense = 2 ** N <= quantum.DENSE_LIMIT
    u = np.linspace(1.0, 0.0, points or (161 if dense else 61))
    gammas = g_max * u ** 2                          # denser at small G, where the crossings are
    sweep = tf.sweep(gammas, k=None if dense else levels, track=track)
    upper = min(len(opt_seqs), sweep["levels"].shape[1] - 1)
    gc, gap = tf.refine_min_gap(sweep["gammas"], sweep["levels"], upper=upper)
    # resolve the avoided crossing in the plot: fine grid around the minimum gap
    width = 0.05 * max(gc, 0.2)
    fine = np.linspace(min(gc + width, g_max), max(gc - width, 0.0), 21)
    sweep = merge_sweeps(sweep, tf.sweep(fine, k=sweep["levels"].shape[1], track=track))
    runs = [tf.anneal(g_max, T, dt=dt, track=track) for T in Ts]

    title = f"{inst.name}:  n = {n}, m = {inst.m}, {N} qubits, P = {P:g}"
    names = inst.job_names
    seq = opt_seqs[0]
    plotting.plot_gantt_comparison(inst, seq, os.path.join(out, "gantt"))
    plotting.plot_anneal(sweep, runs, low=levels, degeneracy=len(opt_seqs), gap=(gc, gap), title=title,
                         path=os.path.join(out, "anneal.png"))
    H.plot_heatmap(n=n, title=f"QUBO matrix of H_P, {inst.name}, P = {P:g}",
                   path=os.path.join(out, "qubo_heatmap.png"))
    plt.close("all")

    summary = {
        "name": inst.name, "n": n, "m": inst.m, "qubits": N, "p": inst.p.tolist(),
        "P": P, "gamma_max": g_max,
        "optimal_makespan": cmax, "optimal_sequences": [[names[j] for j in s] for s in opt_seqs],
        "h_star": spec.h_star, "threshold_ground_P": spec.threshold_ground(),
        "min_gap": gap, "min_gap_gamma": gc,
        "anneal": [{"T": r["T"], **{f"P({k})": v for k, v in r["final"].items()}} for r in runs],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", nargs="?", default=os.path.join(ROOT, "instances"),
                    help="folder with instance JSON files (default: instances/)")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "figures", "anneal"), help="output folder")
    ap.add_argument("--P", type=float, default=None, help="penalty weight (default: floor(H*/2) + 1)")
    ap.add_argument("--gamma-max", type=float, default=2.0, help="initial field, in units of P (default 2)")
    ap.add_argument("--T", type=float, nargs="+", default=[2.0, 10.0, 50.0], help="anneal times")
    ap.add_argument("--dt", type=float, default=0.01, help="time step of the evolution")
    ap.add_argument("--levels", type=int, default=6, help="levels in the low-eigenspace panel")
    ap.add_argument("--points", type=int, default=None, help="G points (default 161 for n=3, 61 for n=4)")
    args = ap.parse_args()

    if not os.path.isdir(args.folder) or not any(f.endswith(".json") for f in os.listdir(args.folder)):
        print(f"no instances in {args.folder}: generating the {len(DEFAULT_INSTANCES)} default instances")
        generator.generate_folder(args.folder, DEFAULT_INSTANCES)

    instances = generator.load_folder(args.folder)
    print(f"{len(instances)} instances from {os.path.abspath(args.folder)}")
    for inst in instances:
        if inst.n > generator.MAX_JOBS:
            print(f"  skip {inst.name}: n = {inst.n} is too large for exact diagonalisation")
            continue
        s = run_instance(inst, os.path.join(args.out, inst.name), args.P, args.gamma_max, args.T,
                         args.dt, args.levels, args.points)
        finals = ", ".join(f"T={a['T']:g}: {a['P(optimal schedule)']:.2f}" for a in s["anneal"])
        print(f"  {s['name']:<14} Cmax* = {s['optimal_makespan']:>3}  P = {s['P']:g}  "
              f"min gap {s['min_gap']:.3f} at G = {s['min_gap_gamma']:.2f}  "
              f"P(opt) {finals}  [{s['seconds']} s]")
    print("figures written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
