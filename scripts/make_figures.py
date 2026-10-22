"""Reproduce the figures delivered so far.

    python scripts/make_figures.py            # writes into figures/

Outputs
-------
figures/fssp_vs_nowait_gantt.png / .pdf   Gantt comparison (report Figure 1)
figures/gantt_page.html                   Interactive comparison page (the published artifact)
figures/low_spectrum.png                  Low spectrum of H(Gamma), running example
figures/penalty_levels.png                Classical spectrum by penalty level

The interactive page has text written for the running example. The script checks
that the instance is the running example before it builds the page.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from src import example_instance, plotting, quantum, qubo, spectrum  # noqa: E402
from src.schedules import nowait_schedule, standard_schedule  # noqa: E402

OUT = os.path.join(ROOT, "figures")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gantt_page_template.html")
SEQ = (1, 0, 2)                       # J2 -> J1 -> J3


def schedule_json(inst, seq) -> str:
    """Data block of the interactive page (same field names as the original page)."""
    names = inst.job_names

    def pack(s, kind):
        u, t = s.utilisation()
        d = {"ops": [{"job": names[o.job], "machine": o.machine, "start": o.start, "end": o.end} for o in s.ops],
             "cmax": s.cmax,
             "idle": [{"machine": i, "start": a, "end": b} for i, a, b in s.idle_gaps()],
             "util": [u.tolist(), float(t)]}
        if kind == "std":
            d["waits"] = [{"job": names[w.job], "after": w.machine, "start": w.start, "end": w.end} for w in s.waits]
        else:
            d["holds"] = [{"job": names[h.job], "start": h.start, "end": h.end} for h in s.holds]
        return d

    data = {"P": {nm: inst.p[j].tolist() for j, nm in enumerate(names)},
            "seq": [names[j] for j in seq],
            "std": pack(standard_schedule(inst, seq), "std"),
            "nw": pack(nowait_schedule(inst, seq), "nw")}
    return json.dumps(data)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    inst = example_instance()

    # 1. Gantt comparison (PNG + PDF)
    plotting.plot_gantt_comparison(inst, SEQ, os.path.join(OUT, "fssp_vs_nowait_gantt"))

    # 2. Interactive page
    assert np.array_equal(inst.p, [[2, 3, 1], [1, 3, 4], [4, 2, 1]]), "page text is written for the running example"
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read().replace("__DATA__", schedule_json(inst, SEQ))
    with open(os.path.join(OUT, "gantt_page.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # 3. Spectrum figures (P = 3)
    P = 3.0
    H_P = qubo.problem_qubo(inst, P)
    tf = quantum.TransverseFieldHamiltonian(H_P)
    idx = lambda s: quantum.basis_index(qubo.encode(s))                  # noqa: E731
    track = {"optimal schedules": [idx((1, 0, 2)), idx((1, 2, 0))], "decoy J1J2J3": [idx((0, 1, 2))]}
    sweep = tf.sweep(np.linspace(0.02, 3.0, 150), k=4, track=track)
    plotting.plot_low_spectrum(sweep, title="Running example, P = 3", path=os.path.join(OUT, "low_spectrum.png"))
    plotting.plot_penalty_levels(spectrum.enumerate_spectrum(inst), P, title="All 512 states, P = 3",
                                 path=os.path.join(OUT, "penalty_levels.png"))
    print("figures written to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
