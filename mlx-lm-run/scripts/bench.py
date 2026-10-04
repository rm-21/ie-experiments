"""Warm in-process benchmark for mlx-lm.

Loads the model once, does a warmup run, then times prefill and decode
across several prompt lengths.

    uv run scripts/bench.py
    uv run scripts/bench.py --prompt-lengths 64 512 2048 --runs 5
"""

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

FILLER = "The quick brown fox jumps over the lazy dog. "


def make_prompt(tokenizer, n_tokens: int) -> list[int]:
    """Build a chat-formatted prompt of roughly n_tokens tokens."""
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    body = tokenizer.decode(filler_ids * (n_tokens // len(filler_ids) + 1))
    messages = [{"role": "user", "content": f"Summarize this text:\n{body}"}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return ids[-n_tokens:] if len(ids) > n_tokens else ids


def run_once(model, tokenizer, prompt: list[int], max_tokens: int) -> dict:
    start = time.perf_counter()
    ttft = 0.0
    response = None
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        if not ttft:
            ttft = time.perf_counter() - start
    if response is None:
        raise RuntimeError("stream_generate produced no tokens")
    return {
        "prompt_tokens": response.prompt_tokens,
        "prompt_tps": response.prompt_tps,
        "generation_tokens": response.generation_tokens,
        "generation_tps": response.generation_tps,
        "ttft_ms": ttft * 1000,
        "total_s": time.perf_counter() - start,
        "peak_gb": response.peak_memory,
    }


def summarize(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:8.1f}"
    return f"{statistics.median(values):8.1f} ± {statistics.stdev(values):5.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[43, 256, 1024, 4096])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    start = time.perf_counter()
    model, tokenizer = load(args.model)[:2]
    mx.eval(model.parameters())
    print(f"Model load: {time.perf_counter() - start:.2f} s")

    # Warmup: compiles Metal kernels and pages weights in; not counted.
    run_once(model, tokenizer, make_prompt(tokenizer, 128), max_tokens=16)

    print(f"\n{'prompt':>7} | {'prefill tok/s':>16} | {'TTFT ms':>16} | {'decode tok/s':>16} | {'peak GB':>7}")
    print("-" * 75)
    for n in args.prompt_lengths:
        prompt = make_prompt(tokenizer, n)
        results = [run_once(model, tokenizer, prompt, args.max_tokens) for _ in range(args.runs)]
        print(
            f"{results[0]['prompt_tokens']:>7} | "
            f"{summarize([r['prompt_tps'] for r in results]):>16} | "
            f"{summarize([r['ttft_ms'] for r in results]):>16} | "
            f"{summarize([r['generation_tps'] for r in results]):>16} | "
            f"{max(r['peak_gb'] for r in results):7.3f}"
        )
    print(f"\n{args.runs} runs per row after 1 warmup; values are median ± stdev.")


if __name__ == "__main__":
    main()
