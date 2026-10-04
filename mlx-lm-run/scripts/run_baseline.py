"""Step 2: run the CLI on three prompt lengths -> benchmarks/week-01-baseline.csv.

Each prompt runs twice; only the second (warm-cache) run goes in the CSV.
Every run's full terminal output is kept in benchmarks/raw/.
"""

import csv
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from mlx_lm_run import BENCH_DIR, MODEL, PROJECT_DIR, resolved_revision

MAX_TOKENS = 64
RUNS = 2

QUESTION = "Explain a matrix in two sentences."
CONTEXT = (
    "I am learning linear algebra for machine learning. I know what a vector is: an ordered "
    "list of numbers that can represent a point or a direction. I have seen weights in neural "
    "networks described as matrices, and I want to understand why. "
)
PROMPTS = {
    "short": QUESTION,
    "medium": CONTEXT + QUESTION,
    "long": CONTEXT * 4 + QUESTION,
}

# What the CLI prints at the end, e.g. "Prompt: 43 tokens, 689.220 tokens-per-sec".
PATTERNS = {
    "prompt": r"Prompt: (\d+) tokens, ([\d.]+) tokens-per-sec",
    "generation": r"Generation: (\d+) tokens, ([\d.]+) tokens-per-sec",
    "peak_memory_gb": r"Peak memory: ([\d.]+) GB",
}
NOT_REPORTED = "not reported"


def parse(output: str) -> dict[str, str]:
    row = {}
    m = re.search(PATTERNS["prompt"], output)
    row["prompt_tokens"], row["prompt_tps"] = m.groups() if m else (NOT_REPORTED, NOT_REPORTED)
    m = re.search(PATTERNS["generation"], output)
    row["generation_tokens"], row["generation_tps"] = m.groups() if m else (NOT_REPORTED, NOT_REPORTED)
    m = re.search(PATTERNS["peak_memory_gb"], output)
    row["peak_memory_gb"] = m.group(1) if m else NOT_REPORTED
    return row


def main() -> None:
    cli = Path(sys.executable).parent / "mlx_lm.generate"
    (BENCH_DIR / "prompts").mkdir(parents=True, exist_ok=True)
    (BENCH_DIR / "raw").mkdir(parents=True, exist_ok=True)
    revision = resolved_revision()

    rows = []
    for name, prompt in PROMPTS.items():
        (BENCH_DIR / "prompts" / f"{name}.txt").write_text(prompt + "\n")
        cmd = [str(cli), "--model", MODEL, "--prompt", prompt, "--max-tokens", str(MAX_TOKENS)]

        for run in range(1, RUNS + 1):
            print(f"{name} run {run}/{RUNS} ...")
            start = time.perf_counter()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Launch-to-exit: includes Python startup + model load. NOT TTFT.
            wall_s = time.perf_counter() - start
            output = result.stdout + result.stderr
            (BENCH_DIR / "raw" / f"{name}-run{run}.txt").write_text(output)

        # `output` / `wall_s` are from the last (second) run here.
        rows.append(
            {
                "prompt_name": name,
                **parse(output),
                "process_wall_s": f"{wall_s:.2f}",
                "max_tokens": MAX_TOKENS,
                "model": MODEL,
                "revision": revision,
                "command": shlex.join(["mlx_lm.generate", *cmd[1:]]),
            }
        )

    out = BENCH_DIR / "week-01-baseline.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
