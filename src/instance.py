"""Problem instance: processing times, machine loads, heads and tails.

Report reference: Section 3 (no-wait permutation flow shop), Equations (17) and (20).

Index conventions used in the whole package
-------------------------------------------
* Jobs, machines and positions are 0-based in the code.
  The report is 1-based: report machine ``M2`` is ``b = 1`` in the code.
* ``p[j, i]`` is the processing time of job ``j`` on machine ``i``.
  The array shape is ``(n_jobs, n_machines)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Instance:
    """A flow shop instance.

    Parameters
    ----------
    p : array of shape (n, m)
        Positive integer processing times, ``p[j, i]`` = job ``j`` on machine ``i``.
    name : str
        Optional label.
    """

    p: np.ndarray
    name: str = ""
    _cum: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        p = np.asarray(self.p, dtype=int)
        if p.ndim != 2:
            raise ValueError("p must have shape (n_jobs, n_machines)")
        if (p <= 0).any():
            raise ValueError("processing times must be positive")
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "_cum", np.cumsum(p, axis=1))

    # ------------------------------------------------------------------ sizes
    @property
    def n(self) -> int:
        """Number of jobs."""
        return self.p.shape[0]

    @property
    def m(self) -> int:
        """Number of machines."""
        return self.p.shape[1]

    @property
    def job_names(self) -> list[str]:
        return [f"J{j + 1}" for j in range(self.n)]

    @property
    def machine_names(self) -> list[str]:
        return [f"M{i + 1}" for i in range(self.m)]

    # ------------------------------------------------- cumulative times (Eq. 14)
    @property
    def A(self) -> np.ndarray:
        """``A[j, h]`` = sum of p[j, 0..h] (job j has left machine h, relative to its start)."""
        return self._cum

    @property
    def B(self) -> np.ndarray:
        """``B[j, h]`` = sum of p[j, 0..h-1] (job j reaches machine h, relative to its start)."""
        return self._cum - self.p

    # --------------------------------------------- loads and bottleneck (Eq. 20)
    @property
    def loads(self) -> np.ndarray:
        """Machine loads ``Lambda_i = sum_j p[j, i]``."""
        return self.p.sum(axis=0)

    @property
    def bottleneck(self) -> int:
        """Machine with the largest load (0-based). Ties go to the first machine."""
        return int(np.argmax(self.loads))

    # ------------------------------------------------ heads and tails (Eq. 17)
    def head(self, b: int) -> np.ndarray:
        """``head[j]`` = time from the start of job j until it reaches machine b."""
        return self.p[:, :b].sum(axis=1)

    def tail(self, b: int) -> np.ndarray:
        """``tail[j]`` = time from the moment job j leaves machine b until it leaves the line."""
        return self.p[:, b + 1:].sum(axis=1)

    def describe(self) -> str:
        rows = [f"{nm} = {list(map(int, r))}" for nm, r in zip(self.job_names, self.p)]
        return "   ".join(rows)


def example_instance() -> Instance:
    """The running example of the report (Equation 27 and Figure 1)."""
    return Instance(np.array([[2, 3, 1],
                              [1, 3, 4],
                              [4, 2, 1]]), name="running example")


def random_instance(n: int, m: int, low: int = 1, high: int = 9, seed: int | None = None) -> Instance:
    """Uniform random integer processing times in ``[low, high]``."""
    rng = np.random.default_rng(seed)
    return Instance(rng.integers(low, high + 1, size=(n, m)), name=f"random n={n} m={m} seed={seed}")
