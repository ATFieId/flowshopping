"""Perturbation theory of the transverse-field anneal ``H(G) = H_P - G * sum_i sigma^x_i``.

Report reference: Section 2 and Section 3.9.
* Equation (3)  : second-order coefficient kappa_z.
* Equation (4)  : crossing point Gamma_c between a sharp and a wide minimum.
* Equations (5), (6): degenerate decoy sets, hopping matrix and tau.
* Equation (7)  : gap estimate from the effective coupling.
* Equation (8)  : Landau-Zener success probability.
* Equations (10)-(13): penalty form of kappa, Sigma_w and the approximate Gamma_c.
* Equation (34) : swap amplitude 5/2 * G^4 / P^3.

All energies come from a :class:`src.qubo.QUBO` that holds the FULL problem
Hamiltonian ``H_P`` (penalty + objective).
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations
from typing import Sequence

import numpy as np

from .qubo import QUBO, penalty_units

MAX_PATH_LENGTH = 10


# ================================================================ neighbours
def single_flip_neighbours(x: np.ndarray) -> np.ndarray:
    """All ``N`` states at Hamming distance 1 from ``x`` (row a flips qubit a)."""
    x = np.asarray(x, dtype=np.int8)
    X = np.tile(x, (x.size, 1))
    X[np.arange(x.size), np.arange(x.size)] ^= 1
    return X


def flip_excitations(H: QUBO, x: np.ndarray) -> np.ndarray:
    """``E(z') - E(z)`` for every single flip ``z -> z'`` (row a = qubit a)."""
    return H.energy(single_flip_neighbours(x)) - H.energy(x)


# ================================================================ Eq. (3)
def kappa(H: QUBO, x: np.ndarray) -> float:
    """Second-order coefficient ``kappa_z = sum_{d(z,z')=1} 1 / (E_z' - E_z)``.

    The state must be a strict local minimum under single flips.
    """
    dE = flip_excitations(H, x)
    if (dE <= 0).any():
        raise ValueError("state is not a strict single-flip local minimum (some E_z' - E_z <= 0)")
    return float(np.sum(1.0 / dE))


def perturbed_energy(H: QUBO, x: np.ndarray, gamma: float | np.ndarray) -> float | np.ndarray:
    """``E_z(G) ~ E_z - G^2 kappa_z`` (Equation 3, second order)."""
    return H.energy(x) - np.asarray(gamma) ** 2 * kappa(H, x)


# ================================================================ Eq. (4)
def crossing_gamma(H: QUBO, x_star: np.ndarray, x_L: np.ndarray) -> float | None:
    """``Gamma_c = sqrt(delta / (kappa_L - kappa_*))``. None if the levels never cross."""
    delta = H.energy(x_L) - H.energy(x_star)
    dk = kappa(H, x_L) - kappa(H, x_star)
    if delta <= 0:
        raise ValueError("x_L must have a higher classical energy than x_star")
    return float(np.sqrt(delta / dk)) if dk > 0 else None


# ======================================================= Eqs. (10)-(13)
def flip_wastes(H: QUBO, x: np.ndarray, n: int, P: float) -> dict:
    """Split the single flips of a FEASIBLE state into 'off' and 'on' flips (Equation 10).

    Every flip of a feasible state costs a penalty of exactly 2P:
    * flip off:  dE = 2P - w_minus
    * flip on :  dE = 2P + w_plus
    """
    if penalty_units(x, n) != 0:
        raise ValueError("flip_wastes needs a feasible state")
    dE = flip_excitations(H, x)
    on = np.asarray(x) == 0
    return {"w_plus": dE[on] - 2 * P, "w_minus": 2 * P - dE[~on]}


def sigma_w(H: QUBO, x_star: np.ndarray, x_L: np.ndarray, n: int, P: float) -> float:
    """``Sigma_w`` of Equation (11)."""
    a, c = flip_wastes(H, x_star, n, P), flip_wastes(H, x_L, n, P)
    return float(a["w_plus"].sum() - c["w_plus"].sum() + c["w_minus"].sum() - a["w_minus"].sum())


def crossing_gamma_approx(H: QUBO, x_star: np.ndarray, x_L: np.ndarray, n: int, P: float) -> dict:
    """Approximate crossing point ``Gamma_c^2 ~ 4 P^2 delta / Sigma_w`` (Equation 12).

    ``validity_ratio = delta / Sigma_w`` must be small (Equation 13).
    """
    delta = H.energy(x_L) - H.energy(x_star)
    S = sigma_w(H, x_star, x_L, n, P)
    gc = float(np.sqrt(4 * P ** 2 * delta / S)) if S > 0 else None
    return {"delta": delta, "sigma_w": S, "gamma_c": gc,
            "validity_ratio": delta / S if S > 0 else np.inf}


# ====================================================== flip paths (Sec. 3.9)
def flip_paths(x: np.ndarray, y: np.ndarray):
    """Yield every flip order from x to y as a list of the intermediate states."""
    diff = np.flatnonzero(np.asarray(x) != np.asarray(y))
    if diff.size > MAX_PATH_LENGTH:
        raise ValueError("path too long to enumerate")
    for order in permutations(diff):
        z = np.array(x, dtype=np.int8)
        inter = []
        for a in order[:-1]:
            z[a] ^= 1
            inter.append(z.copy())
        yield inter


def path_penalty_profile(x: np.ndarray, y: np.ndarray, n: int) -> Counter:
    """Count flip orders by the penalty levels (units of P) of their intermediate states.

    For a swap of two jobs: 16 orders stay at (2, 2, 2) and 8 pass through (2, 4, 2).
    """
    return Counter(tuple(int(penalty_units(z, n)) for z in inter) for inter in flip_paths(x, y))


def path_sum(H: QUBO, x: np.ndarray, y: np.ndarray, E_ref: float | None = None) -> float:
    """``sum_paths prod_s 1 / (E_s - E_ref)`` over all flip orders from x to y.

    The effective coupling is ``|V_xy| = G^d * path_sum`` with ``d = hamming(x, y)``.
    ``E_ref`` defaults to the mean energy of x and y.
    """
    E_ref = (H.energy(x) + H.energy(y)) / 2 if E_ref is None else E_ref
    total = 0.0
    for inter in flip_paths(x, y):
        dens = H.energy(np.array(inter)) - E_ref if inter else np.array([])
        if (dens <= 0).any():
            raise ValueError("an intermediate state is not above the reference energy")
        total += float(np.prod(1.0 / dens))
    return total


def effective_coupling(H: QUBO, x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """``|V_xy|`` at transverse field ``gamma`` (leading order in G)."""
    d = int(np.sum(np.asarray(x) != np.asarray(y)))
    return gamma ** d * path_sum(H, x, y)


def swap_coupling_leading_order(P: float, gamma: float = 1.0) -> float:
    """``|V| ~ G^4 [16/(2P)^3 + 8/((2P)(4P)(2P))] = 5/2 G^4 / P^3`` (Equation 34)."""
    return gamma ** 4 * (16 / (2 * P) ** 3 + 8 / ((2 * P) * (4 * P) * (2 * P)))


# ================================================== Eqs. (5)-(7): decoy sets
def hopping_matrix(H: QUBO, states: Sequence[np.ndarray], q: int = 4) -> np.ndarray:
    """``T`` of Equation (5): ``T[a, c] = path_sum`` for states at Hamming distance q, else 0."""
    S = [np.asarray(s) for s in states]
    T = np.zeros((len(S), len(S)))
    for a in range(len(S)):
        for c in range(a + 1, len(S)):
            if int(np.sum(S[a] != S[c])) == q:
                T[a, c] = T[c, a] = path_sum(H, S[a], S[c])
    return T


def tau(T: np.ndarray) -> float:
    """Largest eigenvalue of the hopping matrix."""
    return float(np.linalg.eigvalsh(T).max()) if T.size else 0.0


def crossing_gamma_degenerate(delta: float, tau_L: float, tau_star: float, q: int = 4) -> float | None:
    """``Gamma_c ~ (delta / (tau_L - tau_*))^(1/q)`` (Equation 6)."""
    return float((delta / (tau_L - tau_star)) ** (1 / q)) if tau_L > tau_star else None


def gap_estimate(H: QUBO, x_star: np.ndarray, x_L: np.ndarray) -> dict:
    """``Delta_min ~ 2 |V_eff(Gamma_c)|`` (Equation 7), with Gamma_c from Equation (4)."""
    gc = crossing_gamma(H, x_star, x_L)
    if gc is None:
        return {"gamma_c": None, "delta_min": None, "distance": None}
    d = int(np.sum(np.asarray(x_star) != np.asarray(x_L)))
    return {"gamma_c": gc, "distance": d, "delta_min": 2 * effective_coupling(H, x_star, x_L, gc)}


# ================================================================ Eq. (8)
def landau_zener(delta_min: float, sweep_rate: float, hbar: float = 1.0) -> float:
    """``p_success ~ 1 - exp(-pi Delta_min^2 / (2 hbar v))``."""
    return float(1 - np.exp(-np.pi * delta_min ** 2 / (2 * hbar * sweep_rate)))
