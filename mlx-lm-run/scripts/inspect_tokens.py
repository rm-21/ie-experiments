"""Step 3: tokenize plain text with the same snapshot -> benchmarks/tokens.json.

Predict first: will the token count equal the word count?
This is plain text only (add_special_tokens=False). The CLI's prompt count is
higher because the chat template adds role/control tokens around your text.
"""

import json

from mlx_lm import load
from mlx_lm_run import BENCH_DIR, PROJECT_DIR, resolved_revision, snapshot_dir

TEXTS = {
    "original": "Explain a matrix in two sentences.",
    "punctuation": "Explain a matrix, in two sentences!",  # only punctuation changed
}


def main() -> None:
    _, tokenizer = load(str(snapshot_dir()))[:2]

    records = []
    for label, text in TEXTS.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        record = {
            "label": label,
            "text": text,
            "word_count": len(text.split()),
            "utf8_bytes": len(text.encode()),
            "token_count": len(ids),
            "ids": ids,
            # Each id decoded on its own: shows where the tokenizer split the text.
            "pieces": [tokenizer.decode([i]) for i in ids],
            "decoded": tokenizer.decode(ids),
            "revision": resolved_revision(),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))

    out = BENCH_DIR / "tokens.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(f"saved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
