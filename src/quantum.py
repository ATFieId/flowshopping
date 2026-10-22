"""Exact numerics for the transverse-field Hamiltonian (small instances only).

``H(G) = H_P - G * sum_i sigma^x_i``  (Equation 1).

Basis convention (same as :func:`src.spectrum.all_states`):
in basis state ``s``, qubit ``a`` has the value ``(s >> a) & 1``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .qubo import QUBO
from .spectrum import all_states

DENSE_LIMIT = 2 ** 11


def basis_index(x: np.ndarray) -> int:
    """Basis-state number of a classical state."""
    return int(np.dot(np.asarray(x, dtype=np.int64), 1 << np.arange(len(x), dtype=np.int64)))


def diagonal(H: QUBO) -> np.ndarray:
    """Classical energies of all basis states, in basis order."""
    return H.energy(all_states(H.num_qubits))


def transverse_field(N: int) -> sp.csr_matrix:
    """``sum_i sigma^x_i`` as a sparse matrix of size ``2^N``."""
    dim = 2 ** N
    s = np.arange(dim)
    rows = np.repeat(s, N)
    cols = (s[:, None] ^ (1 << np.arange(N))[None, :]).ravel()
    return sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(dim, dim))


class TransverseFieldHamiltonian:
    """Builds ``H(G)`` once and diagonalises it for many values of G."""

    def __init__(self, H: QUBO):
        self.N = H.num_qubits
        self.diag = diagonal(H)
        self.X = transverse_field(self.N)

    def matrix(self, gamma: float) -> sp.csr_matrix:
        return sp.diags(self.diag) - gamma * self.X

    def eigen(self, gamma: float, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Lowest k eigenvalues and eigenvectors (columns)."""
        Hm = self.matrix(gamma)
        if Hm.shape[0] <= DENSE_LIMIT:
            w, v = np.linalg.eigh(Hm.toarray())
            return w[:k], v[:, :k]
        w, v = eigsh(Hm, k=k, which="SA")
        order = np.argsort(w)
        return w[order], v[:, order]

    def sweep(self, gammas: Sequence[float], k: int = 4,
              track: dict[str, Sequence[int]] | None = None) -> dict:
        """Low spectrum along G, plus the ground-state weight on groups of basis states.

        ``track`` maps a name to a list of basis indices, for example
        ``{"optimum": [...], "decoy": [...], "feasible": [...]}``.
        """
        gammas = np.asarray(gammas, dtype=float)
        levels = np.zeros((gammas.size, k))
        weights = {name: np.zeros(gammas.size) for name in (track or {})}
        for t, g in enumerate(gammas):
            w, v = self.eigen(g, k)
            levels[t] = w
            for name, idx in (track or {}).items():
                weights[name][t] = float(np.sum(np.abs(v[list(idx), 0]) ** 2))
        return {"gammas": gammas, "levels": levels, "weights": weights}


def min_gap(gammas: np.ndarray, levels: np.ndarray, upper: int = 1, lower: int = 0) -> tuple[float, float]:
    """Smallest value of ``E_upper - E_lower`` along the sweep: returns ``(gamma, gap)``.

    For a non-degenerate optimum use the default (``E_1 - E_0``). If the classical
    optimum is g-fold degenerate, use ``upper = g``: the gap from the ground level to
    the first level outside the optimal manifold.
    """
    gaps = levels[:, upper] - levels[:, lower]
    t = int(np.argmin(gaps))
    return float(gammas[t]), float(gaps[t])
