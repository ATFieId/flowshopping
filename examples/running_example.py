"""Running example of the report (n = m = 3), with every main object kept in one place.

Use it in one of three ways:

    python examples/running_example.py          # print the walk-through, save figures
    python -i examples/running_example.py       # same, then explore `ex` interactively
    >>> from examples.running_example import build
    >>> ex = build()                            # no printing, no figures

All objects are attributes of ``ex`` (a :class:`RunningExample`). Examples:

    ex.inst.p                 processing times, shape (n, m)
    ex.w                      waste matrix on the bottleneck
    ex.x_opt, ex.X_opt        optimal state as a vector and as a matrix
    ex.H_P.energy(ex.x_opt)   energy of a state
    ex.spec.lowest(P=1)       lowest levels for another penalty weight
    ex.sweep["levels"]        low spectrum of H(Gamma)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import example_instance, nowait, perturbation as pt, quantum, qubo, schedules, spectrum  # noqa: E402
from src.instance import Instance  # noqa: E402


@dataclass
class RunningExample:
    # ---------------------------------------------------------- instance (Sec. 3)
    inst: Instance
    b: int                                   # reference machine (0-based; bottleneck)
    P: float                                 # penalty weight
    # ---------------------------------------------- no-wait quantities (Sec. 3.2-3.5)
    D: np.ndarray                            # delay matrix, Eq. (15)
    w: np.ndarray                            # waste matrix on b, Eq. (18)
    waste_causes: dict                       # (j, l) -> starvation / blocking terms, Eq. (22)
    clean: np.ndarray                        # clean transitions, Definition 9
    makespans: list                          # all (Cmax, sequence), sorted
    seq_opt: tuple                           # one optimal sequence
    cmax_opt: int
    decomposition: dict                      # machine b -> parts of Eq. (19) for seq_opt
    lower_bounds: np.ndarray                 # Eq. (21)
    std_schedule: schedules.Schedule         # standard flow shop, same sequence
    nw_schedule: schedules.Schedule          # no-wait flow shop
    # ------------------------------------------------------------ QUBO (Sec. 3.6)
    H_obj: qubo.QUBO                         # Eq. (26)
    H_pen: qubo.QUBO                         # Eq. (31)
    H_P: qubo.QUBO                           # Eq. (2)
    feasible_states: dict                    # sequence -> state vector
    x_opt: np.ndarray
    X_opt: np.ndarray
    # ------------------------------------------------ spectrum (Sec. 3.7-3.8, 3.10)
    spec: spectrum.Spectrum
    b_comparison: list                       # Table 7
    # ------------------------------------------------- perturbation (Sec. 2, 3.9)
    kappas: dict                             # sequence -> kappa_z, Eq. (3)
    x_star: np.ndarray
    x_L: np.ndarray
    crossing: dict                           # Eqs. (4), (7), (11)-(13)
    swap_profile: dict                       # 16 / 8 flip orders
    swap_coupling: dict                      # Eq. (34) and the exact path sum
    # ------------------------------------------------------------ exact quantum
    sweep: dict = field(default_factory=dict)
    gaps: dict = field(default_factory=dict)


def build(P: float = 3.0, gammas: np.ndarray | None = None) -> RunningExample:
    inst = example_instance()
    b = inst.bottleneck
    n = inst.n

    # -------------------------------------------------- no-wait quantities
    makespans = nowait.all_makespans(inst)
    cmax_opt, opt_seqs = nowait.optimum(inst)
    seq_opt = opt_seqs[0]
    waste_causes = {(j, l): nowait.waste_terms(inst, b, j, l)
                    for j in range(n) for l in range(n) if j != l}

    # -------------------------------------------------------------- QUBO
    H_obj = qubo.objective_qubo(inst, b)
    H_pen = qubo.penalty_qubo(n, P)
    H_P = H_pen + H_obj
    feasible_states = {s: qubo.encode(s) for _, s in makespans}
    x_opt = feasible_states[seq_opt]

    # ---------------------------------------------------------- spectrum
    spec = spectrum.enumerate_spectrum(inst, b)
    b_comparison = []
    for bb in range(inst.m):
        s = spectrum.enumerate_spectrum(inst, bb)
        b_comparison.append({
            "b": bb, "load": int(inst.loads[bb]), "h_star": s.h_star,
            "nonzero_job_pairs": qubo.coupler_count(qubo.objective_qubo(inst, bb)) // (n - 1),
            "sum_waste": int(nowait.waste_matrix(inst, bb).sum()),
            "P_sufficient": s.sufficient_P(), "P_ground": s.threshold_ground(),
            "P_all_feasible": s.threshold_all_feasible()})

    # ------------------------------------------------------ perturbation
    kappas = {s: pt.kappa(H_P, x) for s, x in feasible_states.items()}
    # Sharp minimum: the optimum. Decoy: the next feasible level, at Hamming distance 4.
    x_star = x_opt
    seq_L = next(s for c, s in makespans if c > cmax_opt)
    x_L = feasible_states[seq_L]
    crossing = {
        "seq_star": seq_opt, "seq_L": seq_L,
        "delta": H_P.energy(x_L) - H_P.energy(x_star),
        "kappa_star": kappas[seq_opt], "kappa_L": kappas[seq_L],
        "gamma_c": pt.crossing_gamma(H_P, x_star, x_L),
        "approx": pt.crossing_gamma_approx(H_P, x_star, x_L, n, P),
        "gap": pt.gap_estimate(H_P, x_star, x_L),
    }
    crossing["gamma_c_over_2P"] = crossing["gamma_c"] / (2 * P) if crossing["gamma_c"] else None

    x1, x2 = feasible_states[opt_seqs[0]], feasible_states[opt_seqs[1]]   # the two optima: one swap
    swap_profile = dict(pt.path_penalty_profile(x1, x2, n))
    swap_coupling = {"pair": (opt_seqs[0], opt_seqs[1]), "hamming": qubo.hamming(x1, x2),
                     "leading_order_per_gamma4": pt.swap_coupling_leading_order(P),
                     "exact_path_sum_per_gamma4": pt.path_sum(H_P, x1, x2)}

    ex = RunningExample(
        inst=inst, b=b, P=P, D=nowait.delay_matrix(inst), w=nowait.waste_matrix(inst, b),
        waste_causes=waste_causes, clean=nowait.clean_transitions(inst, b), makespans=makespans,
        seq_opt=seq_opt, cmax_opt=cmax_opt,
        decomposition={bb: nowait.decomposition(inst, seq_opt, bb) for bb in range(inst.m)},
        lower_bounds=nowait.machine_lower_bounds(inst),
        std_schedule=schedules.standard_schedule(inst, seq_opt),
        nw_schedule=schedules.nowait_schedule(inst, seq_opt),
        H_obj=H_obj, H_pen=H_pen, H_P=H_P, feasible_states=feasible_states,
        x_opt=x_opt, X_opt=qubo.as_matrix(x_opt, n), spec=spec, b_comparison=b_comparison,
        kappas=kappas, x_star=x_star, x_L=x_L, crossing=crossing,
        swap_profile=swap_profile, swap_coupling=swap_coupling)

    # ------------------------------------------------------ exact quantum
    if gammas is not None:
        tf = quantum.TransverseFieldHamiltonian(H_P)
        idx = lambda s: quantum.basis_index(feasible_states[s])          # noqa: E731
        track = {"optimal schedules": [idx(s) for s in opt_seqs],
                 f"decoy {inst.job_names[seq_L[0]]}{inst.job_names[seq_L[1]]}{inst.job_names[seq_L[2]]}": [idx(seq_L)],
                 "all feasible": [idx(s) for s in feasible_states]}
        ex.sweep = tf.sweep(gammas, k=4, track=track)
        g = len(opt_seqs)                                               # optimal level is g-fold degenerate
        ex.gaps = {"E1-E0 (inside the optimal pair)": quantum.min_gap(gammas, ex.sweep["levels"], 1),
                   f"E{g}-E0 (first level outside the optimal pair)":
                       quantum.min_gap(gammas, ex.sweep["levels"], g)}
    return ex


# ============================================================== walk-through
def _h(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def report(ex: RunningExample) -> None:
    inst, nm = ex.inst, ex.inst.job_names
    fmt = lambda s: " ".join(nm[j] for j in s)                           # noqa: E731

    _h("Section 3 - Instance")
    print("Processing times:", inst.describe())
    print("Machine loads Lambda_i:", dict(zip(inst.machine_names, inst.loads.tolist())))
    print(f"Bottleneck: {inst.machine_names[ex.b]}  (code index b = {ex.b})")
    print("Head on b:", dict(zip(nm, inst.head(ex.b).tolist())), "  Tail on b:", dict(zip(nm, inst.tail(ex.b).tolist())))

    _h("Section 3.2 - Delay matrix D(j, l)  (rows j, columns l)")
    print(ex.D)
    _h("Sections 3.3-3.5 - Waste on the bottleneck w_b(j, l) and its cause")
    print(ex.w)
    for (j, l), t in ex.waste_causes.items():
        cause = "clean" if t["cause"] is None else f"{t['cause'][0]} from M{t['cause'][1] + 1}"
        print(f"  {nm[j]} -> {nm[l]}: waste {t['waste']}  ({cause})")

    _h("No-wait makespan of every sequence")
    for c, s in ex.makespans:
        print(f"  {fmt(s)}   Cmax = {c}")
    print(f"Machine lower bounds (Eq. 21): {ex.lower_bounds.tolist()}")
    print(f"Standard flow shop, same sequence: Cmax = {ex.std_schedule.cmax};  no-wait: Cmax = {ex.nw_schedule.cmax}")

    _h(f"Proposition 1 - Decomposition of Cmax for {fmt(ex.seq_opt)} with every reference machine")
    for bb, d in ex.decomposition.items():
        print(f"  b = M{bb + 1}: load {d['load']} + head {d['head']} + waste {d['waste']} + tail {d['tail']} = {d['total']}")

    _h(f"Section 3.6 - QUBO: {ex.H_P.num_qubits} qubits, P = {ex.P}")
    print(f"H_obj: {qubo.coupler_count(ex.H_obj)} couplers, local fields {ex.H_obj.linear.tolist()}")
    print(f"Optimal state x_opt = encode({fmt(ex.seq_opt)}), as a matrix (rows = jobs, columns = positions):")
    print(qubo.state_to_str(ex.x_opt, inst.n))
    print(f"H_obj(x_opt) = {ex.H_obj.energy(ex.x_opt):g} = Cmax - Lambda_b = {ex.cmax_opt} - {inst.loads[ex.b]}")

    _h("Section 3.8 - Classical spectrum by penalty level (Table 5)")
    for r in ex.spec.penalty_table():
        print(f"  penalty {r['penalty_units']:>2} P: {r['states']:>4} states, H_obj in [{r['h_obj_min']:g}, {r['h_obj_max']:g}]")
    for P in (1, 2, 3):
        low = ", ".join(f"{r['energy']:g}{'' if r['feasible'] else ' (inf.)'}" for r in ex.spec.lowest(P))
        print(f"  P = {P}: lowest levels {low}")
    print(f"Sufficient P (Prop. 5): {ex.spec.sufficient_P():g};  exact threshold: {ex.spec.threshold_ground():g};"
          f"  all feasible lowest: {ex.spec.threshold_all_feasible():g}")

    _h("Section 3.10 - Effect of the reference machine (Table 7)")
    for r in ex.b_comparison:
        print(f"  b = M{r['b'] + 1}: load {r['load']}, H* {r['h_star']:g}, non-zero job pairs "
              f"{r['nonzero_job_pairs']}, sum w {r['sum_waste']}, P > {r['P_sufficient']:g} (bound), "
              f"P > {r['P_ground']:g} (exact), P > {r['P_all_feasible']:g} (all feasible lowest)")

    _h("Section 2 - Second-order perturbation theory")
    for s, k in ex.kappas.items():
        print(f"  kappa({fmt(s)}) = {k:.4f}   E = {ex.H_P.energy(ex.feasible_states[s]):g}")
    c = ex.crossing
    print(f"Sharp candidate {fmt(c['seq_star'])} vs decoy {fmt(c['seq_L'])}: delta = {c['delta']:g},"
          f" Gamma_c (Eq. 4) = {c['gamma_c']:.3f}, Gamma_c / 2P = {c['gamma_c_over_2P']:.2f}")
    a = c["approx"]
    print(f"Approximation (Eqs. 11-13): Sigma_w = {a['sigma_w']:g}, Gamma_c = {a['gamma_c']:.3f},"
          f" delta / Sigma_w = {a['validity_ratio']:.2f}")
    ok = c["gamma_c_over_2P"] < 0.25
    print(f"Gap estimate (Eq. 7): Delta_min ~ {c['gap']['delta_min']:.3e} at Hamming distance {c['gap']['distance']}"
          + ("" if ok else "  -> NOT reliable: Gamma_c is not small compared with 2P"))

    _h("Section 3.9 - Swap between the two optimal schedules")
    print(f"Penalty profile of the 24 flip orders: {ex.swap_profile}")
    sc = ex.swap_coupling
    print(f"|V| / Gamma^4: leading order 5/(2P^3) = {sc['leading_order_per_gamma4']:.4f};"
          f" exact path sum = {sc['exact_path_sum_per_gamma4']:.4f}")

    if ex.sweep:
        _h("Exact low spectrum of H(Gamma)")
        for name, (g, gap) in ex.gaps.items():
            print(f"  min {name}: {gap:.3e} at Gamma = {g:.2f}")
        print("Note: n = 3 is too small for a sharp perturbative crossing. Gamma_c / 2P is near 0.5,")
        print("so second-order theory is not reliable here. The exact gap to the first level outside")
        print("the optimal pair stays near 1 (no small gap), and the levels change smoothly.")
        print("The Interlock-versus-Campaign family (Section 4) is built to produce real crossings.")


if __name__ == "__main__":
    from src import plotting
    import matplotlib
    matplotlib.use("Agg")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out, exist_ok=True)
    ex = build(P=3.0, gammas=np.linspace(0.02, 3.0, 150))
    report(ex)
    plotting.plot_gantt_comparison(ex.inst, ex.seq_opt, os.path.join(out, "fssp_vs_nowait_gantt"))
    plotting.plot_low_spectrum(ex.sweep, title=f"Running example, P = {ex.P:g}",
                               path=os.path.join(out, "low_spectrum.png"))
    plotting.plot_penalty_levels(ex.spec, ex.P, title=f"All 512 states, P = {ex.P:g}",
                                 path=os.path.join(out, "penalty_levels.png"))
    print(f"\nFigures saved in {out}")
