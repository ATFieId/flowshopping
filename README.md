# flowshopping

Companion code for the Phase 0 report *Perturbative Crossings in Quantum Annealing
for Flow Shop Scheduling*, Sections 1 to 3. Section 4 (the Interlock-versus-Campaign
generator) is the next step.

Requirements: Python 3.10+, `numpy`, `scipy`, `matplotlib` (see `requirements.txt`).

## Quick start

```bash
pip install -r requirements.txt
python examples/running_example.py      # walk-through of the report, saves figures in examples/output/
python -i examples/running_example.py   # same, then explore the object `ex`
python scripts/make_figures.py          # reproduces all delivered figures in figures/
python tests/test_report.py             # 21 checks of the propositions and tables
```

## Layout

```
src/
  instance.py      Instance, loads, bottleneck, heads and tails
  nowait.py        delay D, waste w_b, makespan, decomposition, lower bounds, brute-force optimum
  schedules.py     explicit standard and no-wait schedules (operations, waits, holds, idle gaps)
  qubo.py          one-hot encoding, QUBO class, H_obj, H_pen, H_P, Ising conversion
  spectrum.py      enumeration of all 2^(n^2) states (n <= 4), penalty levels, exact P thresholds
  perturbation.py  kappa, Gamma_c, Sigma_w, flip paths, effective couplings, Landau-Zener
  quantum.py       exact low spectrum of H(Gamma) = H_P - Gamma * sum sigma^x
  plotting.py      Gantt comparison figure, low-spectrum plot, penalty-level plot
examples/running_example.py   the n = m = 3 instance of the report, all objects in one dataclass
scripts/make_figures.py       reproduces the figures (PNG, PDF and the interactive HTML page)
tests/test_report.py          checks of Propositions 1-5, Equations 34-36 and Tables 2, 5, 6, 7
```

## Conventions

* Indices are **0-based** in the code. Report machine `M2` is `b = 1`; report job `J2` is `1`.
* `p[j, i]` is the processing time of job `j` on machine `i`; shape `(n, m)`.
* Qubit `x[j, k]` (job `j` at position `k`) has the flat index `a = j * n + k`.
* A state is a 0/1 vector of length `n*n`. `qubo.as_matrix(x, n)` gives the matrix view.
* Quantum basis state `s` has qubit `a` equal to `(s >> a) & 1` (`quantum.basis_index(x)`).
* `Spectrum.pen` is the penalty in units of `P`, so the energy is `P * pen + h_obj`.

## Report equations and code

| Report | Code |
|---|---|
| Eq. (1) `H(Γ) = H_P − Γ Σ σx` | `quantum.TransverseFieldHamiltonian` |
| Eq. (2) `H_P = H_pen + H_obj` | `qubo.problem_qubo` |
| Eq. (3) `κ_z`, second-order energy | `perturbation.kappa`, `perturbation.perturbed_energy` |
| Eq. (4) crossing point `Γ_c` | `perturbation.crossing_gamma` |
| Eqs. (5), (6) hopping matrix, `τ`, degenerate `Γ_c` | `perturbation.hopping_matrix`, `tau`, `crossing_gamma_degenerate` |
| Eq. (7) gap estimate | `perturbation.gap_estimate`, `effective_coupling`, `path_sum` |
| Eq. (8) Landau–Zener | `perturbation.landau_zener` |
| Eq. (9) Hamming distance `2r` | `qubo.hamming` |
| Eqs. (10)–(13) `w±`, `Σ_w`, approximate `Γ_c`, validity | `perturbation.flip_wastes`, `sigma_w`, `crossing_gamma_approx` |
| Eq. (14) `A_j(h)`, `B_l(h)` | `Instance.A`, `Instance.B` |
| Eq. (15) delay `D(j, l)` | `nowait.delay_matrix` |
| Eq. (16) no-wait makespan | `nowait.makespan` |
| Eq. (17) head, tail | `Instance.head(b)`, `Instance.tail(b)` |
| Eq. (18) waste `w_b(j, l)` | `nowait.waste_matrix` |
| Eq. (19), Prop. 1 decomposition | `nowait.decomposition` |
| Eq. (20) load, bottleneck | `Instance.loads`, `Instance.bottleneck` |
| Eq. (21) machine lower bounds | `nowait.machine_lower_bounds` |
| Eq. (22), Prop. 2 starvation / blocking | `nowait.waste_terms`, `nowait.waste_general` |
| Eq. (24) utilisation | `nowait.utilisation`, `Schedule.utilisation` |
| Eq. (25) closed form, clean transitions, slack | `nowait.waste_three_machine`, `clean_transitions`, `slack_three_machine` |
| Eq. (26) `H_obj` | `qubo.objective_qubo` |
| Eq. (28) encoding | `qubo.encode`, `qubo.decode` |
| Eq. (29) feasible subspace | `qubo.is_feasible`, `Spectrum.feasible` |
| Eq. (30), Prop. 3 | checked in `tests/test_report.py` |
| Eq. (31) `H_pen` | `qubo.penalty_qubo`, `qubo.penalty_units` |
| Prop. 4 (minimum penalty `2P`) | checked in `tests/test_report.py` |
| Eq. (32), Prop. 5 (sufficient `P`) | `Spectrum.sufficient_P` |
| Eq. (33) decoy below infeasible states | `Spectrum.threshold_decoy` |
| Tables 5, 6 | `Spectrum.penalty_table`, `Spectrum.lowest` |
| Sec. 3.9 swap flip orders (16 / 8) | `perturbation.path_penalty_profile` |
| Eq. (34) swap amplitude `5Γ⁴/(2P³)` | `perturbation.swap_coupling_leading_order` |
| Eq. (35), Table 7 reference machine | `qubo.objective_qubo(inst, b)`, `Spectrum.threshold_*` |
| Eq. (36) mixed `H_obj^(λ)` | `qubo.objective_qubo_mixed` |
| Figure 1 | `plotting.plot_gantt_comparison` |

## The running example object

`examples/running_example.build()` returns a `RunningExample` with these fields:

| Field | Content |
|---|---|
| `inst`, `b`, `P` | instance, reference machine (bottleneck), penalty weight |
| `D`, `w`, `waste_causes`, `clean` | delay matrix, waste matrix, starvation/blocking terms per pair, clean transitions |
| `makespans`, `seq_opt`, `cmax_opt` | all sequences with their makespan, one optimum |
| `decomposition` | Proposition 1 for every reference machine |
| `lower_bounds` | Equation (21) |
| `std_schedule`, `nw_schedule` | `Schedule` objects (operations, waits, holds, idle gaps) |
| `H_obj`, `H_pen`, `H_P` | `QUBO` objects |
| `feasible_states`, `x_opt`, `X_opt` | state vectors per sequence, optimal state as vector and matrix |
| `spec` | `Spectrum` of all 512 states |
| `b_comparison` | Table 7 |
| `kappas`, `x_star`, `x_L`, `crossing` | second-order analysis of the optimum against the next level |
| `swap_profile`, `swap_coupling` | Section 3.9 |
| `sweep`, `gaps` | exact low spectrum of `H(Γ)` and its minimum gaps |

**Limit of the running example.** With `n = 3`, second-order theory predicts
`Γ_c ≈ 2.9` at `P = 3`, so `Γ_c / 2P ≈ 0.5`. Perturbation theory is not reliable
there, and the exact spectrum shows no small gap. The example demonstrates the
tools. Real perturbative crossings need the larger, structured instances of Section 4.

## Scale limits

* `spectrum` and `quantum`: full Hilbert space, `n <= 4` (16 qubits).
* `nowait.optimum` and `all_makespans`: brute force over `n!` sequences, `n <= 9`.
* `perturbation.flip_paths`: Hamming distance `<= 10`.
