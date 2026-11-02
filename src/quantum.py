"""Exact numerics for the transverse-field Hamiltonian (small instances only).

``H(G) = H_P - G * sum_i sigma^x_i``  (Equation 1).

Basis convention (same as :func:`src.spectrum.all_states`):
in basis state ``s``, qubit ``a`` has the value ``(s >> a) & 1``.
"""
from __future__ import annotations

import time
import warnings
from typing import Sequence

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg

from .qubo import QUBO
from .spectrum import all_states

DENSE_LIMIT = 2 ** 11
LANCZOS_SLOW = 1.0          # seconds; see TransverseFieldHamiltonian._continuation


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

    def sweep(self, gammas: Sequence[float], k: int | None = 4,
              track: dict[str, Sequence[int]] | None = None) -> dict:
        """Low spectrum along G, plus the ground-state weight on groups of basis states.

        ``k = None`` returns every level (dense sizes only, ``2^N <= DENSE_LIMIT``).
        ``track`` maps a name to a list of basis indices, for example
        ``{"optimum": [...], "decoy": [...], "feasible": [...]}``.

        Large sizes use LOBPCG continued from the previous G (warm start, diagonal
        preconditioner). It stays fast at small G, where the low levels bunch together.
        """
        gammas = np.asarray(gammas, dtype=float)
        dim = 2 ** self.N
        if k is None:
            if dim > DENSE_LIMIT:
                raise ValueError("the full spectrum (k=None) is only available for 2^N <= DENSE_LIMIT")
            k = dim
        levels = np.zeros((gammas.size, k))
        weights = {name: np.zeros(gammas.size) for name in (track or {})}
        if dim <= DENSE_LIMIT:
            order = range(gammas.size)
            solve = lambda g: self.eigen(g, k)                                   # noqa: E731
        else:
            order = np.argsort(-gammas, kind="stable")       # large G first: easy start, then continue
            solve = self._continuation(k)
        for t in order:
            w, v = solve(gammas[t])
            levels[t] = w
            for name, idx in (track or {}).items():
                weights[name][t] = float(np.sum(np.abs(v[list(idx), 0]) ** 2))
        return {"gammas": gammas, "levels": levels, "weights": weights}

    def _continuation(self, k: int, extra: int = 3):
        """Solver for a sequence of decreasing G values.

        Lanczos (eigsh) is fastest at large G. Once a call takes longer than
        ``LANCZOS_SLOW`` seconds (the low levels are bunching up), the solver
        switches to LOBPCG started from the previous eigenvectors.
        """
        state = {"X": None, "lobpcg": False}
        kk = k + extra                                       # buffer: levels entering from above

        def solve(g: float) -> tuple[np.ndarray, np.ndarray]:
            A = self.matrix(g)
            X = state["X"]
            w = None
            if X is not None and state["lobpcg"]:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    shift = state["w0"] - 1.0
                    prec = 1.0 / np.maximum(self.diag - shift, 1.0)
                    M = LinearOperator(A.shape, dtype=float,
                                       matvec=lambda x: prec * x if x.ndim == 1 else prec[:, None] * x)
                    w, X = lobpcg(A, X, M=M, largest=False, tol=1e-7, maxiter=1000)
                if any("not reaching the requested tolerance" in str(c.message) for c in caught):
                    w = None                                 # fall back to Lanczos
            if w is None:
                t0 = time.perf_counter()
                w, X = eigsh(A, k=kk, which="SA", v0=None if X is None else X[:, 0])
                state["lobpcg"] = time.perf_counter() - t0 > LANCZOS_SLOW
            o = np.argsort(w)
            w, X = w[o], X[:, o]
            state["X"], state["w0"] = X, w[0]
            return w[:k], X[:, :k]

        return solve

    def refine_min_gap(self, gammas: np.ndarray, levels: np.ndarray, upper: int = 1,
                       xatol: float = 1e-4) -> tuple[float, float]:
        """Minimum of ``E_upper - E_0`` located on the grid, then refined between its neighbours.

        A sharp avoided crossing can fall between grid points; this returns ``(gamma, gap)``
        with the gap minimised to ``xatol`` in G.
        """
        gaps = levels[:, upper] - levels[:, 0]
        t = int(np.argmin(gaps))
        lo = gammas[max(t - 1, 0)]
        hi = gammas[min(t + 1, gammas.size - 1)]
        lo, hi = min(lo, hi), max(lo, hi)
        if hi <= lo:
            return float(gammas[t]), float(gaps[t])

        def gap(g: float) -> float:
            w, _ = self.eigen(g, k=upper + 3)
            return float(w[upper] - w[0])

        r = minimize_scalar(gap, bounds=(lo, hi), method="bounded", options={"xatol": xatol})
        if r.fun < gaps[t]:
            return float(r.x), float(r.fun)
        return float(gammas[t]), float(gaps[t])

    def ground_state(self, gamma: float) -> np.ndarray:
        return self.eigen(gamma, 1)[1][:, 0]

    def anneal(self, gamma_max: float, T: float, dt: float = 0.01,
               track: dict[str, Sequence[int]] | None = None, samples: int = 200) -> dict:
        """Simulated quantum annealing: Schrodinger evolution under ``H(G(t))``.

        Linear schedule ``G(t) = gamma_max * (1 - t / T)``, starting in the ground state
        of ``H(gamma_max)`` and ending at ``G = 0``, where the measured basis state is
        a classical assignment. Second-order Trotter steps: the diagonal part is a
        phase and ``exp(i G dt sum sigma^x)`` factorises into one-qubit rotations.

        Returns ``gammas`` and ``times`` at ``samples`` points, the probability of each
        tracked group at those points, and the final probabilities ``final``.
        """
        steps = max(1, int(np.ceil(T / dt)))
        dt = T / steps
        psi = self.ground_state(gamma_max).astype(complex)
        half_phase = np.exp(-0.5j * dt * self.diag)
        record = np.unique(np.linspace(0, steps, samples).round().astype(int))
        track = track or {}
        probs = {name: [] for name in track}

        def log() -> None:
            p = np.abs(psi) ** 2
            for name, idx in track.items():
                probs[name].append(float(p[list(idx)].sum()))

        log()
        for step in range(1, steps + 1):
            g = gamma_max * (1 - (step - 0.5) / steps)       # midpoint of the step
            psi *= half_phase
            psi = self._rotate_x(psi, g * dt)
            psi *= half_phase
            if step in record:
                log()
        times = record * dt
        return {"T": T, "times": times, "gammas": gamma_max * (1 - times / T),
                "probs": {k: np.array(v) for k, v in probs.items()},
                "final": {k: v[-1] for k, v in probs.items()}, "state": psi}

    def _rotate_x(self, psi: np.ndarray, theta: float) -> np.ndarray:
        """``exp(i theta sum_a sigma^x_a) psi`` = product of one-qubit rotations."""
        c, s = np.cos(theta), 1j * np.sin(theta)
        for a in range(self.N):
            v = psi.reshape(-1, 2, 2 ** a)
            v0, v1 = v[:, 0, :].copy(), v[:, 1, :]
            v[:, 0, :] = c * v0 + s * v1
            v[:, 1, :] = c * v1 + s * v0
        return psi


def min_gap(gammas: np.ndarray, levels: np.ndarray, upper: int = 1, lower: int = 0) -> tuple[float, float]:
    """Smallest value of ``E_upper - E_lower`` along the sweep: returns ``(gamma, gap)``.

    For a non-degenerate optimum use the default (``E_1 - E_0``). If the classical
    optimum is g-fold degenerate, use ``upper = g``: the gap from the ground level to
    the first level outside the optimal manifold.
    """
    gaps = levels[:, upper] - levels[:, lower]
    t = int(np.argmin(gaps))
    return float(gammas[t]), float(gaps[t])
