"""src: no-wait flow shop QUBO and perturbative-crossing tools.

Companion code of the Phase 0 report, Sections 1 to 3.

Modules
-------
instance      Instance data, loads, bottleneck, heads and tails.
nowait        Delays, waste, makespan, bottleneck decomposition.
schedules     Explicit standard and no-wait schedules (for Gantt charts).
qubo          One-hot encoding, QUBO class, H_obj, H_pen, H_P.
spectrum      Full enumeration of the classical spectrum (n <= 4).
perturbation  kappa, Gamma_c, flip paths, effective couplings, Landau-Zener.
quantum       Exact low spectrum of H(Gamma) = H_P - Gamma sum sigma^x.
plotting      Gantt comparison figure and spectrum plots.
"""
from . import instance, nowait, perturbation, plotting, quantum, qubo, schedules, spectrum
from .instance import Instance, example_instance, random_instance

__all__ = ["Instance", "example_instance", "random_instance", "instance", "nowait", "schedules",
           "qubo", "spectrum", "perturbation", "quantum", "plotting"]
