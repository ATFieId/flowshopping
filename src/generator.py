"""Small random no-wait flow shop instances, sized for exact diagonalisation.

With n jobs the one-hot QUBO has ``n^2`` qubits, so the Hilbert space has
``2^(n^2)`` states:

    n = 2   4 qubits       16 states
    n = 3   9 qubits      512 states   (full spectrum, seconds)
    n = 4  16 qubits   65 536 states   (low spectrum, minutes)
    n = 5  25 qubits   3.4e7 states    (out of reach on a PC)

so the generator only accepts ``n <= 4``.

Instances are stored as JSON files: ``{"name", "p", "seed", ...}``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterable

import numpy as np

from .instance import Instance
from .nowait import optimum

MAX_JOBS = 4


def generate_instance(n: int = 3, m: int = 3, low: int = 1, high: int = 9, seed: int | None = None,
                      unique_optimum: bool = True, max_tries: int = 10_000) -> Instance:
    """Random no-wait instance with integer processing times in ``[low, high]``.

    ``unique_optimum``: redraw until exactly one job sequence is optimal. A unique
    optimum gives a single target state, so the gap ``E_1 - E_0`` and the success
    probability are easy to read.
    """
    if not 2 <= n <= MAX_JOBS:
        raise ValueError(f"n must be between 2 and {MAX_JOBS} (n^2 qubits, exact diagonalisation)")
    if m < 2:
        raise ValueError("a flow shop needs at least 2 machines")
    if not 1 <= low <= high:
        raise ValueError("need 1 <= low <= high")
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        p = rng.integers(low, high + 1, size=(n, m))
        inst = Instance(p, name=f"nw_n{n}_m{m}_s{seed}")
        if not unique_optimum or len(optimum(inst)[1]) == 1:
            return inst
    raise RuntimeError("no instance with a unique optimum found; widen [low, high] or use unique_optimum=False")


# ------------------------------------------------------------------ JSON files
def save_instance(inst: Instance, path: str, **meta) -> None:
    """Write ``inst`` as JSON. Extra keyword arguments (seed, ...) are stored too."""
    data = {"name": inst.name, "n": inst.n, "m": inst.m, "p": inst.p.tolist(), **meta}
    text = json.dumps(data, indent=2)
    text = re.sub(r"\[\s+([-\d,\s]+?)\s+\]", lambda mt: "[" + re.sub(r"\s+", " ", mt.group(1)) + "]", text)  # one row per line
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def load_instance(path: str) -> Instance:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    name = data.get("name") or os.path.splitext(os.path.basename(path))[0]
    return Instance(np.array(data["p"]), name=name)


def load_folder(folder: str) -> list[Instance]:
    """All ``*.json`` instances in ``folder``, sorted by file name."""
    files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
    return [load_instance(os.path.join(folder, f)) for f in files]


def generate_folder(folder: str, specs: Iterable[dict]) -> list[str]:
    """Generate one instance per spec (keyword arguments of :func:`generate_instance`) into ``folder``."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    for spec in specs:
        inst = generate_instance(**spec)
        path = os.path.join(folder, f"{inst.name}.json")
        save_instance(inst, path, **{k: v for k, v in spec.items() if k != "n" and k != "m"})
        paths.append(path)
    return paths
