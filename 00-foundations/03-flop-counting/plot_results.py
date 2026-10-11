"""Extra figures for the notes, drawn from the saved CSV so the benchmark doesn't need re-running."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BANDWIDTH = 273e9  # bytes/s, Apple's published M4 Pro memory bandwidth
ITEMSIZE = 4  # float32

LESSON_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd() / "00-foundations/03-flop-counting"
BENCH_DIR = LESSON_DIR / "benchmarks"
COLORS = {"loops": "#2a78d6", "rows": "#eb6834", "np": "#1baf7a"}
INK, MUTED, GRID = "#3d3d3a", "#73726c", "#e4e3dc"

rows = list(csv.DictReader((BENCH_DIR / "week-01-matmul.csv").open()))
by_impl: dict[str, list[dict]] = {}
for r in rows:
    by_impl.setdefault(r["impl"], []).append(r)


def style(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


# ── Roofline view: measured GFLOP/s against minimal-traffic intensity ─────────

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
intensity = np.logspace(np.log10(3), np.log10(1000), 100)
ax.plot(intensity, BANDWIDTH * intensity / 1e9, color=MUTED, linestyle="--", linewidth=1.5)
ax.text(
    100,
    BANDWIDTH * 100 / 1e9 / 2.2,
    "memory ceiling = 273 GB/s × intensity",
    color=MUTED,
    rotation=13,
    rotation_mode="anchor",
    ha="left",
    va="top",
)

best = max(float(r["gflops"]) for r in rows)
ax.axhline(best, color=MUTED, linestyle=":", linewidth=1.5)
ax.annotate(
    f"highest measured ≈ {best:,.0f} GFLOP/s (not a peak)",
    (1000, best),
    xytext=(0, 6),
    ha="right",
    textcoords="offset points",
    color=MUTED,
)

for impl, rs in by_impl.items():
    x = [float(r["intensity_flop_per_byte"]) for r in rs]
    y = [float(r["gflops"]) for r in rs]
    ax.plot(x, y, marker="o", markersize=5, linewidth=2, color=COLORS[impl], label=impl)
    ax.annotate(impl, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("intensity = FLOPs / bytes_min  (FLOP per byte, minimal-traffic model)")
ax.set_ylabel("GFLOP/s (best of 3)")
ax.set_title("Every implementation sits far below the memory ceiling")
ax.legend(frameon=False, loc="lower right")
style(ax)
fig.tight_layout()
fig.savefig(BENCH_DIR / "week-01-roofline.png")
print(f"wrote {BENCH_DIR}/week-01-roofline.png")


# ── DRAM traffic bounds for np.matmul ─────────────────────────────────────────

np_rows = by_impl["np"]
ns = np.array([int(r["n"]) for r in np_rows])
seconds = np.array([float(r["seconds"]) for r in np_rows])
bytes_min = np.array([float(r["bytes_min"]) for r in np_rows])
naive = (2 * ns**3 + ns**2) * ITEMSIZE  # no reuse at all: a row of A and a column of B per output, plus writing C
ceiling = BANDWIDTH * seconds  # the most DRAM could have delivered in the measured time

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.fill_between(ns, bytes_min, np.minimum(ceiling, naive), color=COLORS["np"], alpha=0.15, linewidth=0)
ax.plot(ns, naive, color=COLORS["rows"], marker="o", markersize=5, linewidth=2, label="no reuse: 2n³ + n² elements")
ax.plot(
    ns,
    ceiling,
    color=COLORS["loops"],
    marker="o",
    markersize=5,
    linewidth=2,
    label="upper bound: 273 GB/s × measured time",
)
ax.plot(
    ns,
    bytes_min,
    color=COLORS["np"],
    marker="o",
    markersize=5,
    linewidth=2,
    label="bytes_min = 3n² × 4 B (lower bound if nothing is cached)",
)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(ns, labels=[str(n) for n in ns])
ax.set_xlabel("n")
ax.set_ylabel("bytes per call (log scale)")
ax.set_title("np.matmul: its real DRAM traffic must lie in the shaded band")
ax.legend(frameon=False, loc="upper left")
style(ax)
fig.tight_layout()
fig.savefig(BENCH_DIR / "week-01-dram-bounds.png")
print(f"wrote {BENCH_DIR}/week-01-dram-bounds.png")

for n, s, lo, hi, nv in zip(ns, seconds, bytes_min, ceiling, naive, strict=True):
    print(
        f"n={n:>5}  min {lo / 1e6:10.2f} MB  max {hi / 1e6:10.2f} MB  no-reuse {nv / 1e6:12.2f} MB  "
        f"reuse needed ≥ {nv / hi:6.1f}x"
    )
