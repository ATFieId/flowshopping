"""One-hot permutation encoding and the QUBO Hamiltonians.

Report reference: Equation (2) (problem Hamiltonian), Equation (26) (objective),
Equation (28) (variables), Equation (29) (feasible subspace),
Equation (31) (penalty) and Equation (36) (mixed reference machines).

Encoding
--------
``x[j, k] = 1`` if job j is at position k. The flat qubit index is ``a = j * n + k``.
A *state* is a 0/1 numpy vector of length ``n * n`` (dtype int8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .instance import Instance
from .nowait import waste_matrix


# ============================================================ variables, states
def var_index(n: int, j: int, k: int) -> int:
    """Flat qubit index of ``x[j, k]`` (job j at position k)."""
    return j * n + k


def var_label(n: int, a: int) -> tuple[int, int]:
    """Inverse of :func:`var_index`: ``a -> (j, k)``."""
    return divmod(a, n)


def encode(seq: Sequence[int]) -> np.ndarray:
    """Permutation -> feasible state (Equation 28). ``seq[k]`` is the job at position k."""
    n = len(seq)
    x = np.zeros(n * n, dtype=np.int8)
    for k, j in enumerate(seq):
        x[var_index(n, j, k)] = 1
    return x


def as_matrix(x: np.ndarray, n: int) -> np.ndarray:
    """State vector -> ``n x n`` matrix X (rows = jobs, columns = positions)."""
    return np.asarray(x).reshape(n, n)


def is_feasible(x: np.ndarray, n: int) -> bool:
    """True if X is a permutation matrix (Equation 29)."""
    X = as_matrix(x, n)
    return bool((X.sum(0) == 1).all() and (X.sum(1) == 1).all())


def decode(x: np.ndarray, n: int) -> tuple[int, ...] | None:
    """Feasible state -> job sequence. Returns None for infeasible states."""
    if not is_feasible(x, n):
        return None
    return tuple(int(j) for j in as_matrix(x, n).argmax(axis=0))


def penalty_units(X: np.ndarray, n: int) -> np.ndarray:
    """Penalty in units of P: ``sum_j (1 - r_j)^2 + sum_k (1 - c_k)^2``.

    ``X`` is one state (shape ``(n*n,)``) or a batch (shape ``(S, n*n)``).
    """
    M = np.asarray(X).reshape(-1, n, n).astype(int)
    r = M.sum(axis=2)
    c = M.sum(axis=1)
    out = ((1 - r) ** 2).sum(axis=1) + ((1 - c) ** 2).sum(axis=1)
    return out if np.ndim(X) == 2 else int(out[0])


def hamming(x: np.ndarray, y: np.ndarray) -> int:
    return int(np.sum(np.asarray(x) != np.asarray(y)))


def state_to_str(x: np.ndarray, n: int) -> str:
    """Readable matrix view of a state."""
    return "\n".join(" ".join(str(int(v)) for v in row) for row in as_matrix(x, n))


# ================================================================ QUBO object
@dataclass
class QUBO:
    """``E(x) = offset + linear . x + sum_{a<c} quadratic[a, c] x_a x_c``.

    ``quadratic`` is strictly upper triangular.
    """

    linear: np.ndarray
    quadratic: np.ndarray
    offset: float = 0.0

    @classmethod
    def zeros(cls, N: int) -> "QUBO":
        return cls(np.zeros(N), np.zeros((N, N)), 0.0)

    @property
    def num_qubits(self) -> int:
        return self.linear.size

    def add_linear(self, a: int, v: float) -> None:
        self.linear[a] += v

    def add_quadratic(self, a: int, c: int, v: float) -> None:
        if a == c:                        # x^2 = x
            self.linear[a] += v
        else:
            a, c = min(a, c), max(a, c)
            self.quadratic[a, c] += v

    def energy(self, X: np.ndarray) -> np.ndarray | float:
        """Energy of one state (shape (N,)) or of a batch (shape (S, N))."""
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        X = np.atleast_2d(X)
        e = self.offset + X @ self.linear + np.einsum("sa,ac,sc->s", X, self.quadratic, X)
        return float(e[0]) if single else e

    def __add__(self, other: "QUBO") -> "QUBO":
        return QUBO(self.linear + other.linear, self.quadratic + other.quadratic,
                    self.offset + other.offset)

    def __mul__(self, s: float) -> "QUBO":
        return QUBO(self.linear * s, self.quadratic * s, self.offset * s)

    __rmul__ = __mul__

    def to_dict(self) -> dict[tuple[int, int], float]:
        """``{(a, a): h_a, (a, c): J_ac}``, the usual format of QUBO solvers."""
        d = {(a, a): float(v) for a, v in enumerate(self.linear) if v != 0}
        for a, c in zip(*np.nonzero(self.quadratic)):
            d[(int(a), int(c))] = float(self.quadratic[a, c])
        return d

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Ising form with ``x = (1 - s) / 2``: ``E = offset + h.s + sum_{a<c} J_ac s_a s_c``."""
        Q = self.quadratic
        h = -self.linear / 2 - (Q.sum(axis=1) + Q.sum(axis=0)) / 4
        J = Q / 4
        offset = self.offset + self.linear.sum() / 2 + Q.sum() / 4
        return h, J, offset


# ===================================================== Hamiltonians of the report
def objective_qubo(inst: Instance, b: int | None = None) -> QUBO:
    """Exact objective Hamiltonian ``H_obj^(b)`` (Equation 26).

    On feasible states it equals ``Cmax - Lambda_b`` (Proposition 3).
    ``b`` defaults to the bottleneck machine.
    """
    b = inst.bottleneck if b is None else b
    n = inst.n
    w = waste_matrix(inst, b)
    head, tail = inst.head(b), inst.tail(b)
    q = QUBO.zeros(n * n)
    for k in range(n - 1):                                   # (a) transition couplers
        for j in range(n):
            for l in range(n):
                if j != l and w[j, l] != 0:
                    q.add_quadratic(var_index(n, j, k), var_index(n, l, k + 1), w[j, l])
    for j in range(n):                                       # (b), (c) local fields
        q.add_linear(var_index(n, j, 0), head[j])
        q.add_linear(var_index(n, j, n - 1), tail[j])
    return q


def objective_qubo_mixed(inst: Instance, lambdas: Sequence[float]) -> QUBO:
    """Convex mix of reference machines ``sum_b lambda_b H_obj^(b)`` (Equation 36)."""
    lambdas = np.asarray(lambdas, dtype=float)
    if lambdas.shape != (inst.m,) or (lambdas < 0).any() or not np.isclose(lambdas.sum(), 1):
        raise ValueError("lambdas must be m non-negative weights that sum to 1")
    q = QUBO.zeros(inst.n ** 2)
    for b, lam in enumerate(lambdas):
        if lam:
            q = q + lam * objective_qubo(inst, b)
    return q


def penalty_qubo(n: int, P: float) -> QUBO:
    """One-hot penalty ``H_pen`` (Equation 31), expanded with ``x^2 = x``.

    ``P (1 - sum_k x_k)^2 = P [1 - sum_k x_k + 2 sum_{k<k'} x_k x_k']`` for each row and column.
    """
    q = QUBO.zeros(n * n)
    q.offset = 2 * n * P
    for j in range(n):
        for k in range(n):
            q.add_linear(var_index(n, j, k), -2 * P)          # once from its row, once from its column
    for j in range(n):
        for k in range(n):
            for k2 in range(k + 1, n):
                q.add_quadratic(var_index(n, j, k), var_index(n, j, k2), 2 * P)   # same job
                q.add_quadratic(var_index(n, k, j), var_index(n, k2, j), 2 * P)   # same position
    return q


def problem_qubo(inst: Instance, P: float, b: int | None = None,
                 lambdas: Sequence[float] | None = None) -> QUBO:
    """Full problem Hamiltonian ``H_P = H_pen + H_obj`` (Equation 2)."""
    obj = objective_qubo_mixed(inst, lambdas) if lambdas is not None else objective_qubo(inst, b)
    return penalty_qubo(inst.n, P) + obj


def coupler_count(q: QUBO) -> int:
    return int(np.count_nonzero(q.quadratic))
