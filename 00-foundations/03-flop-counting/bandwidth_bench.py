import csv
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Measure how fast this machine can actually move bytes, to compare against
# Apple's published 273 GB/s for the M4 Pro.
#
# Idea: run a job whose byte count we know exactly, time it, divide.
#   read  : a.max()            reads a once                -> 1 × size bytes
#   write : b.fill(1.0)        writes b once               -> 1 × size bytes
#   copy  : np.copyto(b, a)    reads a once, writes b once -> 2 × size bytes
# There is almost no arithmetic, so the time is spent waiting on memory.
# (Not a.sum(): on one core it reads at only ~38 GB/s against ~85 GB/s for
# a.max(), so the adding, not memory, is the limit. Checked on this machine.)

SPEC_GBPS = 273  # Apple's published M4 Pro memory bandwidth, for comparison only
TRIALS = 5  # best of 5, like the matmul benchmark takes best of 3
THREADS = os.cpu_count() or 1

LESSON_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd() / "00-foundations/03-flop-counting"
BENCH_DIR = LESSON_DIR / "benchmarks"


# ── 1. The three jobs, each split across `threads` chunks ────────────────────
# NumPy runs one call on one core. To use every core we cut the array into
# equal chunks and give each thread one chunk. NumPy releases Python's lock
# (the GIL) during these calls, so the threads really run in parallel.


def chunks(x: np.ndarray, threads: int) -> list[np.ndarray]:
    return np.array_split(x, threads)  # views into x, no copying


def run(job: str, a: np.ndarray, b: np.ndarray, pool: ThreadPoolExecutor, threads: int) -> None:
    pairs = list(zip(chunks(a, threads), chunks(b, threads)))
    futures: list[Future] = []
    for ca, cb in pairs:
        if job == "read":
            futures.append(pool.submit(np.max, ca))
        elif job == "write":
            futures.append(pool.submit(cb.fill, 1.0))
        else:  # copy
            futures.append(pool.submit(np.copyto, cb, ca))
    for future in futures:
        future.result()  # wait for every chunk to finish


BYTES_PER_ELEMENT_MOVED = {"read": 1, "write": 1, "copy": 2}  # times itemsize


def best_gbps(job: str, a: np.ndarray, b: np.ndarray, threads: int) -> float:
    bytes_moved = BYTES_PER_ELEMENT_MOVED[job] * a.nbytes
    with ThreadPoolExecutor(max_workers=threads) as pool:
        run(job, a, b, pool, threads)  # warm-up, not recorded
        times = []
        for _ in range(TRIALS):
            start = time.perf_counter()
            run(job, a, b, pool, threads)
            times.append(time.perf_counter() - start)
    return bytes_moved / min(times) / 1e9


# sanity check on a tiny case: the jobs do what the byte counts claim
_a, _b = np.arange(10, dtype=np.float32), np.zeros(10, dtype=np.float32)
with ThreadPoolExecutor(max_workers=3) as _pool:
    run("copy", _a, _b, _pool, 3)
    assert np.array_equal(_a, _b)
    run("write", _a, _b, _pool, 3)
    assert np.all(_b == 1.0)


# ── 2. DRAM bandwidth: arrays far bigger than any cache ──────────────────────
# 1 GB per array. The M4 Pro's caches are on the order of 10s of MB, so almost
# every byte must come from (or go to) DRAM.

N_BIG = 256 * 1024 * 1024  # float32 elements -> 1 GiB per array
a = np.ones(N_BIG, dtype=np.float32)  # np.ones (not np.empty) so the pages are really allocated
b = np.ones(N_BIG, dtype=np.float32)

print(f"array size: {a.nbytes / 2**30:.0f} GiB each, float32, {THREADS} logical cores")
print(f"Apple's spec: {SPEC_GBPS} GB/s\n")
print(f"{'job':<7}{'threads':>8}{'GB/s':>9}{'% of spec':>11}")

dram_rows: list[dict] = []
for job in ("read", "write", "copy"):
    for threads in (1, THREADS):
        gbps = best_gbps(job, a, b, threads)
        dram_rows.append({"job": job, "threads": threads, "array_bytes": a.nbytes, "gbps": gbps})
        print(f"{job:<7}{threads:>8}{gbps:>9.1f}{gbps / SPEC_GBPS:>10.0%}")

del a, b  # free 2 GiB before the sweep


# ── 3. Sweep the array size: watch the caches appear ─────────────────────────
# Same copy job, one thread, array size from 64 KB to 1 GB. While both arrays
# fit in on-chip cache, the "bandwidth" is cache speed, not DRAM speed. Once
# they don't fit, it drops to DRAM speed.
#
# One thread, because handing a 64 KB copy to 14 threads costs more than the
# copy itself (the overhead lesson from Result 1). Small copies take only
# microseconds, so we repeat each one until a timed block lasts >= 50 ms.

MIN_BLOCK_SECONDS = 0.05


def timed_block(a: np.ndarray, b: np.ndarray, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        np.copyto(b, a)
    return time.perf_counter() - start


def sweep_gbps(a: np.ndarray, b: np.ndarray) -> float:
    np.copyto(b, a)  # warm-up: also pulls both arrays into cache if they fit
    repeats = 1
    while timed_block(a, b, repeats) < MIN_BLOCK_SECONDS:
        repeats *= 2
    best = min(timed_block(a, b, repeats) for _ in range(TRIALS))
    return 2 * a.nbytes * repeats / best / 1e9


print("\ncopy, 1 thread, by array size (working set a + b = 2 × size)")
print(f"{'size':>12}{'GB/s':>9}")

sweep_rows: list[dict] = []
for exp in range(16, 31, 2):  # 64 KiB, 256 KiB, ..., 1 GiB
    n = 2**exp // 4  # float32 elements
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    gbps = sweep_gbps(a, b)
    sweep_rows.append({"array_bytes": a.nbytes, "threads": 1, "gbps": gbps})
    size = f"{a.nbytes / 2**20:.0f} MiB" if a.nbytes >= 2**20 else f"{a.nbytes // 1024} KiB"
    print(f"{size:>12}{gbps:>9.1f}")


# ── 4. Save ──────────────────────────────────────────────────────────────────


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


BENCH_DIR.mkdir(exist_ok=True)
write_csv(BENCH_DIR / "week-01-bandwidth.csv", dram_rows)
write_csv(BENCH_DIR / "week-01-bandwidth-sweep.csv", sweep_rows)
print(f"\nwrote {BENCH_DIR}/week-01-bandwidth.csv and week-01-bandwidth-sweep.csv")
