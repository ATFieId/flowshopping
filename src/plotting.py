"""Figures.

* :func:`plot_gantt_comparison` - the standard vs no-wait Gantt figure (report Figure 1).
* :func:`plot_low_spectrum`     - low levels of H(Gamma) and ground-state weights.
* :func:`plot_penalty_levels`   - classical spectrum grouped by penalty level.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .instance import Instance
from .schedules import Schedule, nowait_schedule, standard_schedule

# ---------------------------------------------------------------- style
JOB_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURF, IDLE, GRID = "#fcfcfb", "#ecebe7", "#e3e2de"
HOLD = "#d03b3b"
MH, BH, GAP = 0.8, 0.42, 0.14          # machine row height, buffer lane height, gap


def _colour(j: int) -> str:
    if j >= len(JOB_COLOURS):
        raise ValueError("the Gantt palette supports at most 8 jobs")
    return JOB_COLOURS[j]


def _row_layout(m: int) -> tuple[list, dict, float]:
    """Rows top to bottom: M1, Buffer 1->2, M2, ... Returns (rows, ypos, y_min)."""
    rows = []
    for i in range(m):
        rows.append((f"M{i + 1}", "m", i))
        if i < m - 1:
            rows.append((f"Buffer {i + 1}→{i + 2}", "b", i + 1))
    ypos, y = {}, 0.0
    for _, kind, idx in rows:
        h = MH if kind == "m" else BH
        ypos[(kind, idx)] = (y - h, h)          # (bottom, height)
        y -= h + GAP
    return rows, ypos, y


def _base_axes(ax, rows, ypos, ymin, xmax, title, c):
    ax.set_facecolor(SURF)
    for x in range(xmax + 1):
        ax.axvline(x, color=GRID, lw=0.6, zorder=0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin - 0.1, 0.12)
    ax.set_yticks([ypos[(k, i)][0] + ypos[(k, i)][1] / 2 for _, k, i in rows])
    ax.set_yticklabels([nm for nm, _, _ in rows])
    for lab, (_, k, _) in zip(ax.get_yticklabels(), rows):
        lab.set_fontsize(11 if k == "m" else 8.5)
        lab.set_color(INK if k == "m" else MUTED)
        lab.set_fontweight("bold" if k == "m" else "normal")
    ax.set_xticks(range(xmax + 1))
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", color=INK, pad=10)
    ax.axvline(c, color=INK, lw=1.6, zorder=6)
    ax.text(c + 0.08, 0.0, rf"$C_{{\max}} = {c}$", fontsize=11, fontweight="bold",
            color=INK, va="bottom", ha="left")


def _draw_ops(ax, sched: Schedule, ypos, names):
    for o in sched.ops:
        y0, h = ypos[("m", o.machine)]
        ax.add_patch(Rectangle((o.start + 0.03, y0), o.end - o.start - 0.06, h,
                               facecolor=_colour(o.job), edgecolor="none", zorder=3))
        ax.text((o.start + o.end) / 2, y0 + h / 2, names[o.job], ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=INK, zorder=4)


def _draw_idle(ax, sched: Schedule, ypos):
    for i, s, e in sched.idle_gaps():
        y0, h = ypos[("m", i)]
        ax.add_patch(Rectangle((s, y0), e - s, h, facecolor=IDLE, edgecolor="none", zorder=1))


def _draw_links(ax, sched: Schedule, ypos):
    """Arrow from the end of each operation to the start of the same job on the next machine."""
    by = {}
    for o in sched.ops:
        by.setdefault(o.job, {})[o.machine] = o
    for job, d in by.items():
        for i in range(sched.m - 1):
            a, b = d[i], d[i + 1]
            ya = ypos[("m", i)][0]
            yb = ypos[("m", i + 1)][0] + ypos[("m", i + 1)][1]
            ax.annotate("", xy=(b.start, yb), xytext=(a.end, ya),
                        arrowprops=dict(arrowstyle="-|>", color=_colour(job), lw=1.3,
                                        shrinkA=0, shrinkB=0, mutation_scale=9), zorder=5)


def _utilisation_title(ax, sched: Schedule):
    u, t = sched.utilisation()
    parts = " · ".join(f"M{i + 1} {v:.0%}" for i, v in enumerate(u))
    ax.set_title(f"Utilisation  {parts} · overall {t:.0%}", loc="right", fontsize=10, color=INK2, pad=12)


def plot_gantt_comparison(inst: Instance, seq: Sequence[int], path: str | None = None, dpi: int = 220):
    """Two stacked Gantt charts: standard flow shop (with buffers) and no-wait flow shop.

    Both use the same job sequence and the same time axis. ``path`` without an
    extension saves ``path.png`` and ``path.pdf``.
    """
    std, nw = standard_schedule(inst, seq), nowait_schedule(inst, seq)
    names = inst.job_names
    rows, ypos, ymin = _row_layout(inst.m)
    c1, c2 = std.cmax, nw.cmax
    xmax = max(c1, c2) + 1

    fig, axes = plt.subplots(2, 1, figsize=(11, 2.8 * inst.m), facecolor=SURF)
    plt.subplots_adjust(left=0.11, right=0.97, top=0.85, bottom=0.12, hspace=0.38)
    order = " → ".join(names[j] for j in seq)
    fig.suptitle(f"Same jobs, same order ({order}): buffered flow shop vs no-wait flow shop",
                 x=0.11, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.97)
    mach = ", ".join(inst.machine_names)
    times = "   ".join(f"{nm} = {list(map(int, r))}" for nm, r in zip(names, inst.p))
    fig.text(0.11, 0.915, f"Processing times [{mach}]:  {times}", fontsize=10.5, color=INK2)

    # ---- panel 1: standard flow shop
    ax = axes[0]
    _base_axes(ax, rows, ypos, ymin, xmax, "1  Standard flow shop: buffers allowed", c1)
    _draw_idle(ax, std, ypos)
    _draw_ops(ax, std, ypos, names)
    for w in std.waits:
        y0, h = ypos[("b", w.machine)]
        ax.add_patch(Rectangle((w.start + 0.03, y0), w.end - w.start - 0.06, h, facecolor="none",
                               edgecolor=_colour(w.job), hatch="////", lw=1.2, zorder=3))
        ax.text(w.start - 0.12, y0 + h / 2, f"{names[w.job]} waits {w.end - w.start}",
                fontsize=9.5, fontweight="bold", color=INK, va="center", ha="right", zorder=6)
    _draw_links(ax, std, ypos)
    _utilisation_title(ax, std)

    # ---- panel 2: no-wait flow shop
    ax = axes[1]
    _base_axes(ax, rows, ypos, ymin, xmax, "2  No-wait flow shop: a started job never stops", c2)
    _draw_idle(ax, nw, ypos)
    _draw_ops(ax, nw, ypos, names)
    for hd in nw.holds:
        y0, h = ypos[("m", 0)]
        ax.add_patch(Rectangle((hd.start + 0.03, y0 + 0.04), hd.end - hd.start - 0.06, h - 0.08,
                               facecolor="none", edgecolor=HOLD, linestyle=(0, (4, 2)), lw=1.6, zorder=4))
        ax.text((hd.start + hd.end) / 2, y0 + h / 2, f"{names[hd.job]} held\n{hd.end - hd.start} units",
                ha="center", va="center", fontsize=8.5, fontweight="bold", color=HOLD, zorder=5,
                linespacing=1.0)
    note_x = int(inst.p[seq[0], 0]) + 0.15          # just right of the first M1 -> M2 arrow
    for k in range(1, inst.m):
        y0, h = ypos[("b", k)]
        ax.text(note_x, y0 + h / 2, "empty: buffers not allowed", fontsize=8.5, color=MUTED,
                va="center", ha="left", style="italic")
    _draw_links(ax, nw, ypos)
    if c2 != c1:                                    # makespan increase against the standard shop
        ax.axvline(c1, color=MUTED, lw=1.2, ls=(0, (3, 3)), zorder=5)
        ax.text(c1 - 0.08, 0.0, f"standard {c1}", fontsize=9, color=MUTED, va="bottom", ha="right")
        ax.add_patch(Rectangle((c1, ymin - 0.09), c2 - c1, 0.17, facecolor=HOLD, alpha=0.18,
                               edgecolor="none", zorder=2))
        ax.text((c1 + c2) / 2, ymin - 0.005, f"+{c2 - c1}", fontsize=10, fontweight="bold",
                color=HOLD, ha="center", va="center", zorder=6)
    _utilisation_title(ax, nw)
    ax.set_xlabel("Time (units)", fontsize=10.5, color=INK2)

    # ---- legend
    handles = [Patch(facecolor=_colour(j), label=names[j]) for j in range(inst.n)] + [
        Patch(facecolor="none", edgecolor=INK2, hatch="////", label="Waiting in buffer"),
        Patch(facecolor="none", edgecolor=HOLD, linestyle="--", label="Held before M1"),
        Patch(facecolor=IDLE, label="Machine idle"),
        Line2D([0], [0], color=INK2, lw=1.3, marker=">", markersize=5, label="Job moves to next machine")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.015), ncol=len(handles),
               frameon=False, fontsize=9.5, handlelength=1.5, columnspacing=1.2, labelcolor=INK2)

    if path:
        for ext in ("png", "pdf"):
            fig.savefig(f"{path}.{ext}", dpi=dpi, facecolor=SURF)
    return fig


def plot_low_spectrum(sweep: dict, title: str = "", path: str | None = None):
    """Lowest levels of H(Gamma) (top, relative to the ground level) and ground-state weights (bottom)."""
    g, L = sweep["gammas"], sweep["levels"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, facecolor=SURF,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for k in range(1, L.shape[1]):
        ax1.plot(g, L[:, k] - L[:, 0], lw=2, color=JOB_COLOURS[(k - 1) % len(JOB_COLOURS)],
                 label=f"$E_{k} - E_0$")
    ax1.set_ylabel("Gap to ground level", color=INK2)
    ax1.legend(frameon=False, fontsize=9)
    for i, (name, w) in enumerate(sweep["weights"].items()):
        ax2.plot(g, w, lw=2, color=JOB_COLOURS[i % len(JOB_COLOURS)], label=name)
    ax2.set_ylabel("Ground-state weight", color=INK2)
    ax2.set_xlabel(r"Transverse field $\Gamma$", color=INK2)
    ax2.set_ylim(-0.02, 1.02)
    if sweep["weights"]:
        ax2.legend(frameon=False, fontsize=9)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.6)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK2, length=0)
    ax1.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, facecolor=SURF)
    return fig


def plot_penalty_levels(spec, P: float, title: str = "", path: str | None = None):
    """Classical energy of every state against its penalty level (feasible states highlighted)."""
    import numpy as np
    E = spec.energies(P)
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURF)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.25, 0.25, E.size)
    inf = ~spec.feasible
    ax.scatter(spec.pen[inf] + jitter[inf], E[inf], s=10, color=MUTED, alpha=0.5, label="infeasible")
    ax.scatter(spec.pen[~inf], E[~inf], s=40, color=JOB_COLOURS[0], zorder=3, label="feasible")
    ax.axhline(spec.h_star, color=INK, lw=1, ls=(0, (3, 3)))
    ax.text(ax.get_xlim()[1], spec.h_star, r" $H^*_{\mathrm{obj}}$", va="center", color=INK, fontsize=10)
    ax.set_xlabel("Penalty level (units of P)", color=INK2)
    ax.set_ylabel(f"Energy at P = {P:g}", color=INK2)
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, facecolor=SURF)
    return fig
