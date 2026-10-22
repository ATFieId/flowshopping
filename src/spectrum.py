"""Full enumeration of the classical spectrum for small instances.

Report reference: Section 3.8 (infeasible subspace), Propositions 4 and 5,
Equations (32), (33), Tables 5, 6 and 7.

Basis convention (shared with :mod:`src.quantum`): in basis state number ``s``,
qubit ``a`` has the value ``(s >> a) & 1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .instance import Instance
from .qubo import QUBO, decode, objective_qubo, objective_qubo_mixed, penalty_units

MAX_QUBITS = 20


def all_states(N: int) -> np.ndarray:
    """All ``2^N`` states as an int8 array of shape ``(2^N, N)``."""
    if N > MAX_QUBITS:
        raise ValueError(f"full enumeration is limited to {MAX_QUBITS} qubits (n <= 4)")
    s = np.arange(2 ** N, dtype=np.int64)[:, None]
    return ((s >> np.arange(N)) & 1).astype(np.int8)


@dataclass
class Spectrum:
    """Objective value and penalty level of every state.

    ``energy = P * pen + h_obj`` for any penalty weight P.
    """

    n: int
    states: np.ndarray       # (2^N, N)
    h_obj: np.ndarray        # (2^N,)  objective energy of each state
    pen: np.ndarray          # (2^N,)  penalty in units of P
    feasible: np.ndarray     # (2^N,)  bool

    # ---------------------------------------------------------------- basics
    def energies(self, P: float) -> np.ndarray:
        return P * self.pen + self.h_obj

    @property
    def feasible_levels(self) -> list[tuple[float, tuple[int, ...]]]:
        """Sorted ``(H_obj, sequence)`` for all feasible states."""
        idx = np.flatnonzero(self.feasible)
        return sorted((float(self.h_obj[i]), decode(self.states[i], self.n)) for i in idx)

    @property
    def h_star(self) -> float:
        """Optimal objective energy ``H*_obj``."""
        return float(self.h_obj[self.feasible].min())

    # -------------------------------------------------- Table 5: penalty levels
    def penalty_table(self) -> list[dict]:
        rows = []
        for lev in np.unique(self.pen):
            sel = self.pen == lev
            rows.append({"penalty_units": int(lev), "states": int(sel.sum()),
                         "h_obj_min": float(self.h_obj[sel].min()),
                         "h_obj_max": float(self.h_obj[sel].max())})
        return rows

    # -------------------------------------------------- Table 6: low spectrum
    def lowest(self, P: float, k: int = 5) -> list[dict]:
        E = self.energies(P)
        order = np.lexsort((~self.feasible, E))[:k]          # ties: feasible first
        return [{"energy": float(E[i]), "feasible": bool(self.feasible[i]),
                 "sequence": decode(self.states[i], self.n), "index": int(i)} for i in order]

    # ------------------------------------------- Prop. 5 and exact thresholds
    def sufficient_P(self) -> float:
        """Sufficient bound of Proposition 5: ``P > H*_obj / 2`` (Equation 32)."""
        return self.h_star / 2

    def threshold_below(self, level: float) -> float:
        """Smallest P such that every infeasible state lies strictly above ``level``.

        Infeasible energy ``P * pen + h_obj > level`` for all infeasible states
        holds if and only if ``P > max (level - h_obj) / pen``.
        """
        inf = ~self.feasible
        return float(np.max((level - self.h_obj[inf]) / self.pen[inf]))

    def threshold_ground(self) -> float:
        """Exact P threshold for a feasible ground state."""
        return self.threshold_below(self.h_star)

    def threshold_all_feasible(self) -> float:
        """Exact P threshold such that all n! feasible states are the lowest levels."""
        return self.threshold_below(float(self.h_obj[self.feasible].max()))

    def threshold_decoy(self, E_L: float) -> float:
        """Exact P threshold for Equation (33): the decoy level is below all infeasible states."""
        return self.threshold_below(E_L)


def enumerate_spectrum(inst: Instance, b: int | None = None,
                       lambdas: Sequence[float] | None = None,
                       objective: QUBO | None = None) -> Spectrum:
    """Enumerate all ``2^(n^2)`` states (n <= 4)."""
    n = inst.n
    if objective is None:
        objective = objective_qubo_mixed(inst, lambdas) if lambdas is not None else objective_qubo(inst, b)
    X = all_states(n * n)
    pen = penalty_units(X, n)
    return Spectrum(n=n, states=X, h_obj=objective.energy(X), pen=pen, feasible=pen == 0)


def feasible_fraction(n: int) -> tuple[int, float]:
    """``(n!, n! / 2^(n^2))`` (Table 3)."""
    from math import factorial
    return factorial(n), factorial(n) / 2 ** (n * n)
