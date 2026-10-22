"""No-wait flow shop: delays, waste, makespan and the bottleneck decomposition.

Report reference: Sections 3.2 to 3.5, Equations (15)-(25), Propositions 1 and 2.
All machine indices are 0-based (report machine M2 is ``b = 1``).
"""
from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np

from .instance import Instance


# ---------------------------------------------------------------- Eq. (15)
def delay_matrix(inst: Instance) -> np.ndarray:
    """``D[j, l]`` = minimum start-to-start delay on M1 when job l directly follows job j.

    ``D(j, l) = max_h [A_j(h) - B_l(h)]``. The diagonal is set to 0 (not used).
    """
    D = (inst.A[:, None, :] - inst.B[None, :, :]).max(axis=2)
    np.fill_diagonal(D, 0)
    return D


# ---------------------------------------------------------------- Eq. (16)
def makespan(inst: Instance, seq: Sequence[int]) -> int:
    """No-wait makespan of the job sequence ``seq`` (Equation 16)."""
    D = delay_matrix(inst)
    return int(sum(D[a, c] for a, c in zip(seq, seq[1:])) + inst.A[seq[-1], -1])


def start_times(inst: Instance, seq: Sequence[int]) -> list[int]:
    """Start time on M1 of each job in ``seq`` (no-wait, tightest schedule)."""
    D = delay_matrix(inst)
    s = [0]
    for a, c in zip(seq, seq[1:]):
        s.append(s[-1] + int(D[a, c]))
    return s


# ---------------------------------------------------------------- Eq. (18)
def waste_matrix(inst: Instance, b: int) -> np.ndarray:
    """``w_b[j, l]`` = idle time on machine b when job l directly follows job j (Equation 18)."""
    w = delay_matrix(inst) - (inst.A[:, b][:, None] - inst.B[:, b][None, :])
    np.fill_diagonal(w, 0)
    return w


# ---------------------------------------------------------------- Eq. (22)
def waste_terms(inst: Instance, b: int, j: int, l: int) -> dict:
    """Starvation and blocking terms of Proposition 2 for the transition j -> l.

    Returns a dict with the waste value, the term of every machine h != b,
    and the machine that sets the waste (``None`` if the transition is clean).
    """
    p, m = inst.p, inst.m
    starvation = {h: int(p[l, h:b].sum() - p[j, h + 1:b + 1].sum()) for h in range(b)}
    blocking = {h: int(p[j, b + 1:h + 1].sum() - p[l, b:h].sum()) for h in range(b + 1, m)}
    terms = {**starvation, **blocking}
    value = max([0, *terms.values()])
    cause = None
    if value > 0:
        h = max(terms, key=terms.get)
        cause = ("starvation" if h < b else "blocking", h)
    return {"waste": value, "starvation": starvation, "blocking": blocking, "cause": cause}


def waste_general(inst: Instance, b: int) -> np.ndarray:
    """Waste matrix from Proposition 2 (must equal :func:`waste_matrix`)."""
    w = np.zeros((inst.n, inst.n), dtype=int)
    for j in range(inst.n):
        for l in range(inst.n):
            if j != l:
                w[j, l] = waste_terms(inst, b, j, l)["waste"]
    return w


# ---------------------------------------------------------------- Eq. (25)
def waste_three_machine(inst: Instance) -> np.ndarray:
    """Closed form for m = 3, b = M2: ``max(0, p1l - p2j, p3j - p2l)``."""
    if inst.m != 3:
        raise ValueError("closed form needs exactly 3 machines")
    p = inst.p
    w = np.maximum(0, np.maximum(p[None, :, 0] - p[:, None, 1], p[:, None, 2] - p[None, :, 1]))
    np.fill_diagonal(w, 0)
    return w


def clean_transitions(inst: Instance, b: int) -> np.ndarray:
    """Boolean matrix: True where the transition j -> l has zero waste (Definition 9)."""
    c = waste_matrix(inst, b) == 0
    np.fill_diagonal(c, False)
    return c


def slack_three_machine(inst: Instance) -> np.ndarray:
    """Slack ``min(p2j - p1l, p2l - p3j)`` of each transition (m = 3, b = M2).

    The value is meaningful for clean transitions only (it is negative otherwise).
    """
    p = inst.p
    s = np.minimum(p[:, None, 1] - p[None, :, 0], p[None, :, 1] - p[:, None, 2])
    np.fill_diagonal(s, 0)
    return s


# ------------------------------------------------- Eq. (19), Proposition 1
def decomposition(inst: Instance, seq: Sequence[int], b: int) -> dict:
    """Bottleneck decomposition of the makespan (Equation 19).

    ``Cmax = load_b + head_b(first) + sum of waste_b + tail_b(last)``.
    """
    w = waste_matrix(inst, b)
    wastes = [int(w[a, c]) for a, c in zip(seq, seq[1:])]
    parts = {
        "load": int(inst.loads[b]),
        "head": int(inst.head(b)[seq[0]]),
        "waste": wastes,
        "tail": int(inst.tail(b)[seq[-1]]),
    }
    parts["total"] = parts["load"] + parts["head"] + sum(wastes) + parts["tail"]
    return parts


# ---------------------------------------------------------------- Eq. (21)
def machine_lower_bounds(inst: Instance) -> np.ndarray:
    """``LB_i = Lambda_i + min_j head_i(j) + min_j tail_i(j)`` for every machine i."""
    return np.array([inst.loads[i] + inst.head(i).min() + inst.tail(i).min() for i in range(inst.m)])


# ---------------------------------------------------------------- Eq. (24)
def utilisation(inst: Instance, seq: Sequence[int]) -> np.ndarray:
    """Utilisation ``Lambda_i / Cmax`` of every machine in the no-wait schedule."""
    return inst.loads / makespan(inst, seq)


# ------------------------------------------------------------ exact solver
def all_makespans(inst: Instance) -> list[tuple[int, tuple[int, ...]]]:
    """All ``(Cmax, sequence)`` pairs, sorted by makespan. Practical for n <= 9."""
    return sorted((makespan(inst, s), s) for s in permutations(range(inst.n)))


def optimum(inst: Instance) -> tuple[int, list[tuple[int, ...]]]:
    """Optimal no-wait makespan and all optimal sequences (brute force)."""
    res = all_makespans(inst)
    best = res[0][0]
    return best, [s for c, s in res if c == best]
