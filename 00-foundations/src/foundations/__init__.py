"""Shared helpers for Phase 00 lessons."""

from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def model_cache_dir(model: str = MODEL) -> Path:
    # The HF cache stores "org/name" as "models--org--name".
    return Path(HF_HUB_CACHE) / f"models--{model.replace('/', '--')}"


def resolved_revision(model: str = MODEL) -> str:
    """The commit hash that `main` pointed to when the model was downloaded."""
    return (model_cache_dir(model) / "refs" / "main").read_text().strip()


def snapshot_dir(model: str = MODEL) -> Path:
    """The exact files the CLI loaded: snapshots/<revision>/."""
    return model_cache_dir(model) / "snapshots" / resolved_revision(model)
