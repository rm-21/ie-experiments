# 01-first-model

Lesson `p0-b-first-model`: run `mlx-community/Llama-3.2-1B-Instruct-4bit` locally with MLX and record a baseline.

## Run

From the workspace root (`inference_engineering/`), in order:

```bash
uv sync --all-packages                                      # once: install into the shared .venv

uv run 00-foundations/01-first-model/record_env.py             # 1. environment -> ENVIRONMENT.md
uv run 00-foundations/01-first-model/run_baseline.py           # 2. 3 prompts x 2 CLI runs -> benchmarks/
uv run 00-foundations/01-first-model/inspect_tokens.py         # 3. plain-text tokenization -> benchmarks/tokens.json
uv run 00-foundations/01-first-model/write_notes.py            # 4. results -> notes/week-01.md
```

Outputs always land in this folder, whichever directory you run from. Step 4 reads the files from steps 2 and 3, so run those first.

Optional, in-process timing (model loaded once, warmup excluded):

```bash
uv run 00-foundations/01-first-model/bench.py --prompt-lengths 64 512 2048 --runs 5
```

## Generated files (don't edit by hand, re-run the script)

| File | From |
|---|---|
| `ENVIRONMENT.md` | `record_env.py` |
| `benchmarks/week-01-baseline.csv` | `run_baseline.py` (second run of each prompt) |
| `benchmarks/prompts/*.txt` | `run_baseline.py` (exact prompts used) |
| `benchmarks/raw/*.txt` | `run_baseline.py` (full terminal output of every run) |
| `benchmarks/tokens.json` | `inspect_tokens.py` |
| `notes/week-01.md` | `write_notes.py`: everything from `## My observation` down is yours and is kept on re-runs |

## Reading the numbers

- **Prompt tok/s** is prefill: the whole prompt in one forward pass.
- **Generation tok/s** is decode: one token per forward pass.
- **Process wall s** runs from launch to exit and includes Python startup and model load. It is not TTFT.
- Each CLI run reloads the model, so compare the reported rates, not wall time.
