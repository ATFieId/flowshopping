"""Checks of every proposition and table of the report (Sections 1 to 3).

Run with ``python tests/test_report.py`` or with ``pytest``.
"""
from __future__ import annotations

import os
import sys
from itertools import permutations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import example_instance, nowait, perturbation as pt, quantum, qubo, random_instance, schedules, spectrum  # noqa: E402

RANDOM = [random_instance(n, m, seed=s) for s in range(4) for n, m in ((3, 3), (4, 3), (4, 5), (5, 4))]
EX = example_instance()


# ------------------------------------------------------------ Section 3
def test_nowait_makespan_matches_explicit_schedule():
    for inst in RANDOM:
        for seq in permutations(range(inst.n)):
            assert nowait.makespan(inst, seq) == schedules.nowait_schedule(inst, seq).cmax


def test_nowait_schedule_has_no_gaps_and_no_overlaps():
    for inst in RANDOM[:6]:
        for seq in permutations(range(inst.n)):
            s = schedules.nowait_schedule(inst, seq)
            t = s.table()
            assert (t[:, 1:, 0] == t[:, :-1, 1]).all()                   # zero gap between machines
            for i in range(inst.m):                                        # no overlap on a machine
                iv = sorted(tuple(t[j, i]) for j in range(inst.n))
                assert all(a[1] <= b[0] for a, b in zip(iv, iv[1:]))


def test_proposition1_decomposition_every_machine():
    for inst in RANDOM:
        for seq in permutations(range(inst.n)):
            c = nowait.makespan(inst, seq)
            for b in range(inst.m):
                assert nowait.decomposition(inst, seq, b)["total"] == c


def test_waste_nonnegative():
    for inst in RANDOM:
        for b in range(inst.m):
            assert (nowait.waste_matrix(inst, b) >= 0).all()


def test_proposition2_general_waste_formula():
    for inst in RANDOM:
        for b in range(inst.m):
            assert np.array_equal(nowait.waste_general(inst, b), nowait.waste_matrix(inst, b))


def test_closed_form_three_machines():
    for inst in [i for i in RANDOM if i.m == 3]:
        assert np.array_equal(nowait.waste_three_machine(inst), nowait.waste_matrix(inst, 1))


def test_machine_lower_bound():
    for inst in RANDOM:
        assert nowait.optimum(inst)[0] >= nowait.machine_lower_bounds(inst).max()


def test_running_example_values():
    assert EX.loads.tolist() == [7, 8, 6] and EX.bottleneck == 1
    assert nowait.waste_matrix(EX, 1).tolist() == [[0, 0, 1], [1, 0, 2], [0, 0, 0]]     # Table 2
    assert nowait.optimum(EX) == (12, [(1, 0, 2), (1, 2, 0)])
    assert schedules.standard_optimum(EX) == (10, [(1, 0, 2)])


# ------------------------------------------------------ Sections 3.6-3.8
def test_proposition3_objective_on_feasible_states():
    for inst in RANDOM:
        for b in range(inst.m):
            H = qubo.objective_qubo(inst, b)
            for seq in permutations(range(inst.n)):
                assert np.isclose(H.energy(qubo.encode(seq)), nowait.makespan(inst, seq) - inst.loads[b])


def test_encode_decode_roundtrip():
    for seq in permutations(range(4)):
        assert qubo.decode(qubo.encode(seq), 4) == seq


def test_penalty_qubo_equals_penalty_count():
    for n in (2, 3, 4):
        X = spectrum.all_states(n * n)
        assert np.allclose(qubo.penalty_qubo(n, 1.0).energy(X), qubo.penalty_units(X, n))


def test_proposition4_minimum_penalty_is_two():
    for n in (2, 3, 4):
        pen = qubo.penalty_units(spectrum.all_states(n * n), n)
        assert pen[pen > 0].min() == 2


def test_proposition5_sufficient_penalty():
    for inst in [i for i in RANDOM if i.n <= 4][:6] + [EX]:
        spec = spectrum.enumerate_spectrum(inst)
        P = spec.sufficient_P() + 1e-9
        assert spec.lowest(P, 1)[0]["feasible"]
        assert spec.threshold_ground() <= spec.sufficient_P() + 1e-12


def test_tables_5_6_7():
    spec = spectrum.enumerate_spectrum(EX)
    rows = {r["penalty_units"]: r for r in spec.penalty_table()}
    assert (rows[0]["states"], rows[2]["states"], rows[4]["states"], rows[6]["states"]) == (6, 99, 144, 94)
    assert (rows[2]["h_obj_min"], rows[4]["h_obj_min"]) == (1, 0)
    assert [r["feasible"] for r in spec.lowest(1)] == [False, True, True, False, False]
    assert all(r["feasible"] for r in spec.lowest(3))
    expected = {0: (5, 2.5, 4.5), 1: (4, 1.5, 3.5), 2: (6, 3.0, 5.0)}
    for b, (hs, pg, pa) in expected.items():
        s = spectrum.enumerate_spectrum(EX, b)
        assert (s.h_star, s.threshold_ground(), s.threshold_all_feasible()) == (hs, pg, pa)


def test_equation35_reference_machine_shift():
    for inst in RANDOM:
        H = [qubo.objective_qubo(inst, b) for b in range(inst.m)]
        for seq in permutations(range(inst.n)):
            x = qubo.encode(seq)
            for b in range(inst.m):
                assert np.isclose(H[b].energy(x) - H[0].energy(x), inst.loads[0] - inst.loads[b])


def test_equation36_mixed_hamiltonian_is_exact():
    inst = RANDOM[5]
    lam = np.full(inst.m, 1 / inst.m)
    H = qubo.objective_qubo_mixed(inst, lam)
    for seq in permutations(range(inst.n)):
        assert np.isclose(H.energy(qubo.encode(seq)), nowait.makespan(inst, seq) - lam @ inst.loads)


# ------------------------------------------------------ Sections 2 and 3.9
def test_feasible_states_are_not_neighbours():
    for a, b in permutations(list(permutations(range(3))), 2):
        assert qubo.hamming(qubo.encode(a), qubo.encode(b)) >= 4


def test_swap_flip_orders_16_and_8():
    for n in (3, 4):
        x = qubo.encode(tuple(range(n)))
        y = qubo.encode((1, 0) + tuple(range(2, n)))
        assert dict(pt.path_penalty_profile(x, y, n)) == {(2, 2, 2): 16, (2, 4, 2): 8}


def test_swap_coupling_leading_order():
    assert np.isclose(pt.swap_coupling_leading_order(3.0), 2.5 / 27)
    # A pure penalty Hamiltonian has no objective corrections: the path sum equals 5/(2P^3).
    P = 3.0
    H = qubo.penalty_qubo(3, P)
    x, y = qubo.encode((0, 1, 2)), qubo.encode((1, 0, 2))
    assert np.isclose(pt.path_sum(H, x, y), pt.swap_coupling_leading_order(P))


def test_kappa_and_second_order_energy():
    """Second-order energy agrees with exact diagonalisation at small Gamma."""
    H = qubo.problem_qubo(EX, 3.0)
    x = qubo.encode((0, 1, 2))                       # non-degenerate level E = 5
    g = 0.05
    tf = quantum.TransverseFieldHamiltonian(H)
    w, _ = tf.eigen(g, k=8)
    pred = pt.perturbed_energy(H, x, g)
    assert np.min(np.abs(w - pred)) < 1e-5


def test_ising_conversion():
    H = qubo.problem_qubo(EX, 2.0)
    h, J, off = H.to_ising()
    X = spectrum.all_states(9)
    S = 1 - 2 * X.astype(float)
    E = off + S @ h + np.einsum("sa,ac,sc->s", S, J, S)
    assert np.allclose(E, H.energy(X))


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok   {name}")
    print(f"\nall {len(tests)} checks passed")
