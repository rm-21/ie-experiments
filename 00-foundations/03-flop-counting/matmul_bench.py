import csv
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MatmulFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def matmul_loops(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc
    return C


def matmul_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # one vectorized inner product per output element: C[i, j] = A[i, :] . B[:, j]
    M, _ = A.shape
    _, N = B.shape
    C = np.empty((M, N), dtype=A.dtype)
    for i in range(M):
        for j in range(N):
            C[i, j] = np.dot(A[i, :], B[:, j])
    return C


def matmul_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


rng = np.random.default_rng(0)
A = rng.random((16, 32), dtype=np.float32)
B = rng.random((32, 24), dtype=np.float32)

loops = matmul_loops(A, B)
rows = matmul_rows(A, B)
mnp = matmul_np(A, B)

assert np.allclose(loops, rows, rtol=1e-5, atol=1e-6)
assert np.allclose(rows, mnp, rtol=1e-5, atol=1e-6)
assert np.allclose(loops, mnp, rtol=1e-5, atol=1e-6)

print("dtype:", loops.dtype)
print("max |loops - np|:", np.max(np.abs(loops - mnp)))


# ── 3. Count FLOPs and bytes ──────────────────────────────────────────────────


def flops_and_bytes(M: int, K: int, N: int, itemsize: int) -> tuple[int, int]:
    flops = 2 * M * N * K  # one multiply + one add per (i, j, k)
    bytes_min = (M * K + K * N + M * N) * itemsize  # read A, read B, write C, each once
    return flops, bytes_min


# unit check from the lesson: n=128 is 2 * 128**3 FLOPs
assert flops_and_bytes(128, 128, 128, 4) == (4_194_304, 3 * 128 * 128 * 4)


# ── 4. Time each implementation across sizes ─────────────────────────────────

TRIALS = 3  # recorded trials per (impl, n); best and median are taken over these
MIN_BLOCK_SECONDS = 0.05  # grow the repeat count until one timed block is at least this long
MAX_REPEATS = 10_000

IMPLS: dict[str, tuple[MatmulFn, tuple[int, ...]]] = {
    "loops": (matmul_loops, (32, 64, 128)),
    "rows": (matmul_rows, (32, 64, 128, 256, 512)),
    "np": (matmul_np, (32, 64, 128, 256, 512, 1024, 2048, 4096)),
}

# __file__ only exists when run as a script; a REPL gets the code as a string, so fall back to the lesson folder
LESSON_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd() / "00-foundations/03-flop-counting"
BENCH_DIR = LESSON_DIR / "benchmarks"


def time_block(fn: MatmulFn, A: np.ndarray, B: np.ndarray, repeats: int) -> float:
    # times the call including its output allocation, as all three functions allocate C
    start = time.perf_counter()
    for _ in range(repeats):
        fn(A, B)
    return time.perf_counter() - start


def calibrate_repeats(fn: MatmulFn, A: np.ndarray, B: np.ndarray) -> int:
    # Double the repeat count until a block is long enough to time reliably.
    # A slow call (the triple loop) already exceeds the target at repeats=1, so it stays at 1.
    repeats = 1
    while repeats < MAX_REPEATS and time_block(fn, A, B, repeats) < MIN_BLOCK_SECONDS:
        repeats = min(repeats * 2, MAX_REPEATS)
    return repeats


def fmt_time(seconds: float) -> str:
    # human-readable unit for printing only; the CSVs stay in seconds
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:7.2f} {unit}"
    return f"{seconds / 1e-9:7.2f} ns"


def fmt_count(x: float) -> str:
    # 4194304 -> "4.19 M", for printing FLOP and byte counts
    for unit, scale in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if x >= scale:
            return f"{x / scale:6.2f} {unit}"
    return f"{x:6.0f}  "


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@cache
def make_inputs(n: int) -> tuple[np.ndarray, np.ndarray]:
    # made once per size (seed = n), outside the timer; @cache hands every implementation the same arrays
    rng = np.random.default_rng(n)
    return rng.random((n, n), dtype=np.float32), rng.random((n, n), dtype=np.float32)


summary: list[dict] = []
raw: list[dict] = []

print("FLOPs = 2·n³   bytes_min = 3·n²·4 (float32: read A, read B, write C once)")
print("GFLOP/s = FLOPs / best / 1e9   intensity = FLOPs / bytes_min")
print(
    f"{'impl':<6}{'n':>5}{'reps':>7}{'best':>11}{'median':>11}"
    f"{'FLOPs':>10}{'bytes_min':>10}{'GFLOP/s':>10}{'intensity':>10}"
)

for impl, (fn, sizes) in IMPLS.items():
    for n in sizes:
        A, B = make_inputs(n)
        flops, bytes_min = flops_and_bytes(n, n, n, A.itemsize)

        fn(A, B)  # warm-up, not recorded
        repeats = calibrate_repeats(fn, A, B)

        per_call: list[float] = []
        for trial in range(TRIALS):
            block = time_block(fn, A, B, repeats)
            per_call.append(block / repeats)
            raw.append({"impl": impl, "n": n, "seed": n, "trial": trial, "repeats": repeats, "block_seconds": block})

        best, median = min(per_call), statistics.median(per_call)
        gflops, intensity = flops / best / 1e9, flops / bytes_min
        summary.append(
            {
                "impl": impl,
                "n": n,
                "seconds": best,
                "gflops": gflops,
                "bytes_min": bytes_min,
                "intensity_flop_per_byte": intensity,
            }
        )
        print(
            f"{impl:<6}{n:>5}{repeats:>7} {fmt_time(best)} {fmt_time(median)}"
            f"{fmt_count(flops):>10}{fmt_count(bytes_min):>10}{gflops:>10.3f}{intensity:>10.1f}"
        )

BENCH_DIR.mkdir(exist_ok=True)
write_csv(BENCH_DIR / "week-01-matmul.csv", summary)
write_csv(BENCH_DIR / "week-01-matmul-raw.csv", raw)
print(f"wrote {BENCH_DIR}/week-01-matmul.csv and week-01-matmul-raw.csv")


# ── 5. Plot and look ─────────────────────────────────────────────────────────

# fixed categorical order (blue, orange, aqua) so each impl keeps its color
COLORS = {"loops": "#2a78d6", "rows": "#eb6834", "np": "#1baf7a"}

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
for impl in IMPLS:
    ns = [r["n"] for r in summary if r["impl"] == impl]
    gf = [r["gflops"] for r in summary if r["impl"] == impl]
    ax.plot(ns, gf, marker="o", markersize=5, linewidth=2, color=COLORS[impl], label=impl)
    ax.annotate(impl, (ns[-1], gf[-1]), xytext=(6, 0), textcoords="offset points", va="center", color="#3d3d3a")

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(
    [32, 64, 128, 256, 512, 1024, 2048, 4096], labels=["32", "64", "128", "256", "512", "1024", "2048", "4096"]
)
ax.set_xlabel("n (square matrices, n × n @ n × n)")
ax.set_ylabel("GFLOP/s (best of 3, log scale)")
ax.set_title("float32 matmul throughput: Python loops vs np.dot rows vs np.matmul")
ax.grid(True, which="major", color="#e4e3dc", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(BENCH_DIR / "week-01-matmul.png")
print(f"wrote {BENCH_DIR}/week-01-matmul.png")


def cpu_name() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return platform.processor()


# BLAS backend, dtype and thread settings, so the numbers can be interpreted later
env = {
    "cpu": cpu_name(),
    "logical_cpus": os.cpu_count(),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "dtype": "float32",
    # unset means the BLAS library picks its own thread count
    "thread_env": {k: os.environ.get(k) for k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    "numpy_config": np.show_config(mode="dicts"),
}
(BENCH_DIR / "week-01-env.json").write_text(json.dumps(env, indent=2, default=str))
print(f"wrote {BENCH_DIR}/week-01-env.json (BLAS: {env['numpy_config']['Build Dependencies']['blas']['name']})")
