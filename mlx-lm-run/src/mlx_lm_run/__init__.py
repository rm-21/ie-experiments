"""Shared constants and helpers for the week-01 lab scripts."""

from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

# mlx-lm-run/ — every script writes its outputs relative to this.
PROJECT_DIR = Path(__file__).resolve().parents[2]
BENCH_DIR = PROJECT_DIR / "benchmarks"
NOTES_DIR = PROJECT_DIR / "notes"


def model_cache_dir(model: str = MODEL) -> Path:
    # The HF cache stores "org/name" as "models--org--name".
    return Path(HF_HUB_CACHE) / f"models--{model.replace('/', '--')}"


def resolved_revision(model: str = MODEL) -> str:
    """The commit hash that `main` pointed to when the model was downloaded."""
    return (model_cache_dir(model) / "refs" / "main").read_text().strip()


def snapshot_dir(model: str = MODEL) -> Path:
    """The exact files the CLI loaded: snapshots/<revision>/."""
    return model_cache_dir(model) / "snapshots" / resolved_revision(model)


def main() -> None:
    print(f"model:    {MODEL}")
    print(f"revision: {resolved_revision()}")
    print(f"snapshot: {snapshot_dir()}")
