"""Explicit schedules (operations with start and end times) for Gantt charts.

Two schedule types for one job sequence:
* ``standard_schedule``: permutation flow shop with buffers (semi-active).
* ``nowait_schedule``:   no-wait flow shop (each job is a rigid block).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Sequence

import numpy as np

from .instance import Instance


@dataclass(frozen=True)
class Operation:
    job: int
    machine: int
    start: int
    end: int


@dataclass(frozen=True)
class Wait:
    """Job waits in the buffer in front of ``machine`` (standard flow shop)."""
    job: int
    machine: int
    start: int
    end: int


@dataclass(frozen=True)
class Hold:
    """M1 is free, but the job is held so that it can run without a stop (no-wait)."""
    job: int
    start: int
    end: int


@dataclass
class Schedule:
    kind: str                       # "standard" or "no-wait"
    sequence: tuple[int, ...]
    m: int
    ops: list[Operation]
    waits: list[Wait] = field(default_factory=list)
    holds: list[Hold] = field(default_factory=list)

    @property
    def cmax(self) -> int:
        return max(o.end for o in self.ops)

    def busy(self) -> np.ndarray:
        """Busy time of each machine."""
        t = np.zeros(self.m, dtype=int)
        for o in self.ops:
            t[o.machine] += o.end - o.start
        return t

    def utilisation(self) -> tuple[np.ndarray, float]:
        """Per-machine and overall utilisation over ``[0, Cmax]``."""
        b = self.busy()
        return b / self.cmax, b.sum() / (self.m * self.cmax)

    def idle_gaps(self, horizon: int | None = None) -> list[tuple[int, int, int]]:
        """Idle intervals ``(machine, start, end)`` inside ``[0, horizon]``."""
        horizon = self.cmax if horizon is None else horizon
        gaps = []
        for i in range(self.m):
            t = 0
            for o in sorted((o for o in self.ops if o.machine == i), key=lambda o: o.start):
                if o.start > t:
                    gaps.append((i, t, o.start))
                t = o.end
            if t < horizon:
                gaps.append((i, t, horizon))
        return gaps

    def table(self) -> np.ndarray:
        """``[job, machine] -> (start, end)`` as an array of shape (n, m, 2)."""
        n = len(self.sequence)
        out = np.zeros((n, self.m, 2), dtype=int)
        for o in self.ops:
            out[o.job, o.machine] = (o.start, o.end)
        return out


def standard_schedule(inst: Instance, seq: Sequence[int]) -> Schedule:
    """Semi-active permutation schedule: start = max(machine free, job free)."""
    ops, waits = [], []
    mfree = [0] * inst.m
    for j in seq:
        jfree = 0
        for i in range(inst.m):
            start = max(mfree[i], jfree)
            if i > 0 and start > jfree:
                waits.append(Wait(j, i, jfree, start))
            end = start + int(inst.p[j, i])
            ops.append(Operation(j, i, start, end))
            mfree[i] = jfree = end
    return Schedule("standard", tuple(seq), inst.m, ops, waits=waits)


def nowait_schedule(inst: Instance, seq: Sequence[int]) -> Schedule:
    """No-wait schedule: earliest start on M1 such that the job never stops."""
    ops, holds = [], []
    mfree = [0] * inst.m
    for j in seq:
        offs = inst.B[j]                                   # arrival offset at each machine
        start = int(max(mfree[i] - offs[i] for i in range(inst.m)))
        if start > mfree[0]:
            holds.append(Hold(j, mfree[0], start))
        for i in range(inst.m):
            s = start + int(offs[i])
            ops.append(Operation(j, i, s, s + int(inst.p[j, i])))
            mfree[i] = s + int(inst.p[j, i])
    return Schedule("no-wait", tuple(seq), inst.m, ops, holds=holds)


def standard_makespan(inst: Instance, seq: Sequence[int]) -> int:
    """Makespan of the permutation flow shop with buffers."""
    return standard_schedule(inst, seq).cmax


def standard_optimum(inst: Instance) -> tuple[int, list[tuple[int, ...]]]:
    """Optimal permutation flow shop makespan and all optimal sequences (brute force)."""
    res = sorted((standard_makespan(inst, s), s) for s in permutations(range(inst.n)))
    return res[0][0], [s for c, s in res if c == res[0][0]]
