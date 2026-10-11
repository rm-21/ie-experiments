# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.9",
#     "numpy>=2.0",
# ]
# ///
"""
Visual explainer for the memory-bandwidth benchmark.

    uv run visualize_bandwidth.py [path/to/benchmarks]

If week-01-bandwidth.csv and week-01-bandwidth-sweep.csv are found (in the
given folder, ./benchmarks, or next to this file), panels C and D use YOUR
measured numbers. Otherwise they show illustrative numbers with the right
shape, clearly labelled as such.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SPEC_GBPS = 273

# ── palette ──────────────────────────────────────────────────────────────────
INK = "#1f2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"
CORE = "#ef6c3a"
L1 = "#f4a259"
L2 = "#8cb369"
SLC = "#5b8e7d"
DRAM = "#3d5a80"
READ = "#3d5a80"
WRITE = "#ef6c3a"
COPY = "#8cb369"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
    }
)


# ── load measured data if it exists ──────────────────────────────────────────
def find_bench_dir() -> Path | None:
    candidates = [Path(sys.argv[1])] if len(sys.argv) > 1 else []
    candidates += [Path.cwd() / "benchmarks", Path(__file__).parent / "benchmarks", Path.cwd()]
    for d in candidates:
        if (d / "week-01-bandwidth.csv").exists() and (d / "week-01-bandwidth-sweep.csv").exists():
            return d
    return None


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


bench = find_bench_dir()
if bench:
    dram = read_csv(bench / "week-01-bandwidth.csv")
    sweep = read_csv(bench / "week-01-bandwidth-sweep.csv")
    threads_all = max(int(r["threads"]) for r in dram)
    dram_vals = {(r["job"], int(r["threads"])): float(r["gbps"]) for r in dram}
    sweep_sizes = np.array([int(r["array_bytes"]) for r in sweep])
    sweep_gbps = np.array([float(r["gbps"]) for r in sweep])
    measured = True
else:
    # Illustrative only: plausible shape for an M4 Pro, NOT measurements.
    threads_all = 14
    dram_vals = {
        ("read", 1): 85,
        ("read", 14): 225,
        ("write", 1): 55,
        ("write", 14): 170,
        ("copy", 1): 70,
        ("copy", 14): 215,
    }
    sweep_sizes = 2 ** np.arange(16, 31, 2)
    sweep_gbps = np.array([190, 150, 145, 135, 90, 72, 70, 70])
    measured = False

data_tag = "measured on your machine" if measured else "ILLUSTRATIVE numbers — run the benchmark for yours"


fig = plt.figure(figsize=(17, 12.5), facecolor="white")
gs = fig.add_gridspec(
    2, 2, height_ratios=[0.78, 1], hspace=0.38, wspace=0.18, left=0.04, right=0.97, top=0.88, bottom=0.1
)

fig.text(0.04, 0.955, "How fast can this machine move bytes?", fontsize=24, weight="bold", color=INK)
fig.text(
    0.04,
    0.925,
    "Bandwidth = bytes moved ÷ seconds.  The cores are fast; the bottleneck is getting data to them.",
    fontsize=13,
    color=MUTED,
)


# ── A. The memory hierarchy ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.set_title("A.  Data lives far from the cores. Closer = smaller but faster.", color=INK)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")

levels = [
    # name, size label, x, width, height, colour
    ("CPU cores", "do the math", 1, 14, 18, CORE),
    ("L1 cache", "~128 KB/core", 21, 15, 22, L1),
    ("L2 cache", "~16 MB", 42, 15, 30, L2),
    ("System\ncache", "tens of MB", 63, 15, 36, SLC),
    ("DRAM", "24–48 GB", 84, 15, 44, DRAM),
]
cy = 30
for name, size, x, w, h, col in levels:
    ax.add_patch(
        FancyBboxPatch((x, cy - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=col, ec="none", alpha=0.92)
    )
    ax.text(x + w / 2, cy + 2.6, name, ha="center", va="center", color="white", weight="bold", fontsize=11)
    ax.text(
        x + w / 2, cy - 3.6 if "\n" in name else cy - 2.6, size, ha="center", va="center", color="white", fontsize=9.5
    )

# pipes between levels: thick = fast, thin = slow
pipes = [(15.6, 20.4, 7), (36.6, 41.4, 5), (57.6, 62.4, 3.5), (78.6, 83.4, 2)]
for x0, x1, lw in pipes:
    ax.add_patch(
        FancyArrowPatch(
            (x1, cy), (x0, cy), arrowstyle="-|>", mutation_scale=10 + 3 * lw, lw=lw, color=INK, shrinkA=0, shrinkB=0
        )
    )

ax.annotate("", xy=(2, 6), xytext=(98, 6), arrowprops={"arrowstyle": "<->", "color": MUTED, "lw": 1.2})
ax.text(4, 2.2, "faster, tiny", color=MUTED, fontsize=10)
ax.text(96, 2.2, "slower, huge", color=MUTED, fontsize=10, ha="right")
ax.text(
    70,
    57,
    f"Apple's spec for this pipe: {SPEC_GBPS} GB/s",
    ha="center",
    va="top",
    fontsize=10,
    color=DRAM,
    weight="bold",
)
ax.annotate("", xy=(81, 31.5), xytext=(78, 54), arrowprops={"arrowstyle": "->", "color": DRAM, "lw": 1.2})
ax.text(
    30, 50, "Arrow thickness ≈ how fast data flows to the cores", ha="center", fontsize=10, color=MUTED, style="italic"
)


# ── B. The three jobs and their byte counts ──────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.set_title("B.  Three jobs with a known byte count (almost no math)", color=INK)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")


def array_strip(x, y, label, col, filled=True):
    n = 10
    for i in range(n):
        ax.add_patch(
            FancyBboxPatch(
                (x + i * 2.6, y),
                2.2,
                5,
                boxstyle="round,pad=0.1,rounding_size=0.4",
                fc=col if filled else "white",
                ec=col,
                lw=1.4,
            )
        )
    ax.text(
        x - 1.5,
        y + 2.5,
        label,
        ha="right",
        va="center",
        fontsize=12,
        weight="bold",
        color=INK,
        family="DejaVu Sans Mono",
    )


rows = [
    ("read", "a.max()", READ, 47, "1 × size"),
    ("write", "b.fill(1.0)", WRITE, 28, "1 × size"),
    ("copy", "np.copyto(b, a)", COPY, 9, "2 × size"),
]
for job, code, col, y, bytes_lbl in rows:
    ax.text(0, y + 10, job.upper(), fontsize=12, weight="bold", color=col)
    ax.text(12, y + 10, code, fontsize=11, color=MUTED, family="DejaVu Sans Mono")
    if job == "read":
        array_strip(8, y, "a", col)
        ax.add_patch(
            FancyArrowPatch((36, y + 2.5), (48, y + 2.5), arrowstyle="-|>", mutation_scale=16, lw=2, color=col)
        )
        ax.text(50, y + 2.5, "core\n(keeps biggest)", va="center", fontsize=9.5, color=INK)
    elif job == "write":
        ax.text(14, y + 2.5, "core", va="center", ha="center", fontsize=9.5, color=INK)
        ax.add_patch(
            FancyArrowPatch((18, y + 2.5), (27, y + 2.5), arrowstyle="-|>", mutation_scale=16, lw=2, color=col)
        )
        array_strip(30, y, "b", col)
    else:
        array_strip(8, y, "a", col)
        ax.add_patch(
            FancyArrowPatch((36, y + 2.5), (45, y + 2.5), arrowstyle="-|>", mutation_scale=16, lw=2, color=col)
        )
        array_strip(50, y, "b", col, filled=True)
    ax.add_patch(FancyBboxPatch((80, y - 0.5), 18, 6, boxstyle="round,pad=0.3,rounding_size=1", fc=col, ec="none"))
    ax.text(89, y + 2.5, bytes_lbl, ha="center", va="center", color="white", weight="bold", fontsize=12)

ax.text(89, 58, "bytes moved", ha="center", fontsize=10, color=MUTED)
ax.text(
    50,
    1,
    "Why max(), not sum()?  Adding is slow enough on one core that the math becomes the limit,\n"
    "not memory. A benchmark must make the thing you're measuring the slowest part.",
    ha="center",
    va="bottom",
    fontsize=9.5,
    color=MUTED,
    style="italic",
)


# ── C. One thread vs all threads, against the spec ───────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.set_title(f"C.  One ape vs. a team: 1 thread vs. {threads_all} threads (1 GiB arrays → DRAM)", color=INK)
jobs = ["read", "write", "copy"]
cols = [READ, WRITE, COPY]
x = np.arange(len(jobs))
w = 0.36
one = [dram_vals[(j, 1)] for j in jobs]
many = [dram_vals[(j, threads_all)] for j in jobs]
b1 = ax.bar(x - w / 2, one, w, color=cols, alpha=0.38, edgecolor=cols, lw=1.5)
b2 = ax.bar(x + w / 2, many, w, color=cols, edgecolor=cols, lw=1.5)
ax.axhline(SPEC_GBPS, color=INK, ls="--", lw=1.4)
ax.text(2.62, SPEC_GBPS + 5, f"Apple spec {SPEC_GBPS} GB/s", ha="right", va="bottom", fontsize=10, color=INK)
for bars, vals, n in ((b1, one, 1), (b2, many, threads_all)):
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 4,
            f"{v:.0f}\n{v / SPEC_GBPS:.0%}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            6,
            f"{n}T",
            ha="center",
            fontsize=9.5,
            color="white" if n > 1 else INK,
            weight="bold",
        )
ax.set_xticks(x, [j.upper() for j in jobs], fontsize=12, weight="bold")
ax.set_ylabel("GB/s")
ax.set_ylim(0, SPEC_GBPS * 1.25)
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.text(
    0.0,
    -0.13,
    f"One core can't drain the pipe alone; splitting the array into {threads_all} chunks gets "
    f"much closer to spec.   ({data_tag})",
    transform=ax.transAxes,
    fontsize=9.5,
    color=MUTED,
    style="italic",
)


# ── D. The cache staircase ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.set_title("D.  Shrink the data and it suddenly gets 'faster': you're hitting cache", color=INK)
# working set = a + b = 2 × array size, so a level holding C bytes fits arrays up to C/2
regions = [
    (2**15, 2**16, L1, "L1"),
    (2**16, 2**23, L2, "L2"),
    (2**23, 2**25, SLC, "system\ncache"),
    (2**25, 2**31, DRAM, "DRAM"),
]
ymax = max(sweep_gbps.max(), SPEC_GBPS) * 1.18
for lo, hi, col, name in regions:
    ax.axvspan(lo, hi, color=col, alpha=0.13, lw=0)
    ax.text(np.sqrt(lo * hi), ymax * 0.95, name, ha="center", va="top", fontsize=10.5, color=col, weight="bold")
ax.plot(sweep_sizes, sweep_gbps, "-o", color=INK, lw=2.4, ms=8, mfc="white", mew=2, zorder=3)
for s, g in zip(sweep_sizes, sweep_gbps):
    ax.text(float(s), g + ymax * 0.03, f"{g:.0f}", ha="center", fontsize=9.5, color=INK)
ax.axhline(SPEC_GBPS, color=INK, ls="--", lw=1.0, alpha=0.5)
ax.text(2**30.8, SPEC_GBPS + 4, "spec (all cores)", ha="right", va="bottom", fontsize=9, color=MUTED)
ax.set_xscale("log", base=2)
ax.set_xlim(2**15, 2**31)
ax.set_ylim(0, ymax)
ticks = 2 ** np.arange(16, 31, 2)
ax.set_xticks(ticks, [f"{t // 2**20} MiB" if t >= 2**20 else f"{t // 2**10} KiB" for t in ticks], fontsize=9.5)
ax.set_xlabel("size of each array (copy job, 1 thread; working set = 2 × size)")
ax.set_ylabel("GB/s")
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.text(
    0.0,
    -0.22,
    f"Left: data sits in the small fast pile next to the core. Right: every byte comes from DRAM.\n"
    f"Cache boundaries approximate.   ({data_tag})",
    transform=ax.transAxes,
    fontsize=9.5,
    color=MUTED,
    style="italic",
)

out = Path(__file__).with_name("bandwidth_explainer.png")
fig.savefig(out, dpi=160)
print(f"saved {out}  ({'measured' if measured else 'illustrative'} data)")
