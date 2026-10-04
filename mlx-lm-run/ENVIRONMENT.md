# Run environment

Recorded 2026-10-04. The first run downloads and loads the weights, so it is a cold run, not a warm inference benchmark.

## Model
- Repo: `mlx-community/Llama-3.2-1B-Instruct-4bit`
- Resolved snapshot revision (HF cache `refs/main`): `08231374eeacb049a0eade7922910865b8fce912`

## Hardware / OS
- Mac model: Mac16,7 (Apple M4 Pro)
- Memory: 48 GB
- macOS: 27.0.1 (build 26A434)

## Command
```bash
uv run mlx_lm.generate --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --prompt "Explain a matrix in 2 sentences" --max-tokens 64
```

## Toolchain
- Python 3.14.3 (workspace `.venv` at repo root)
- uv 0.12.23

## Packages (`uv pip freeze --python ../.venv/bin/python`)
```
annotated-doc==0.0.5
anyio==4.15.1
certifi==2026.7.22
click==8.5.0
filelock==4.0.10
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.33.0
idna==3.20
jinja2==3.1.6
markdown-it-py==4.2.0
markupsafe==3.0.4
mdurl==0.1.2
mlx==0.32.3
mlx-lm==0.32.0
-e file:///Users/rishabhmittal/dev/learning/inference_engineering/mlx-lm-run
mlx-metal==0.32.3
numpy==2.5.3
packaging==26.3
protobuf==7.36.2
pygments==2.21.0
pyyaml==6.0.3
regex==2026.9.29
rich==15.0.0
safetensors==0.8.0
sentencepiece==0.2.2
shellingham==1.5.4
tokenizers==0.23.2
tqdm==4.70.1
transformers==5.18.0
typer==0.27.2
typing-extensions==4.16.0
```

## Warm runs (weights cached, 2026-10-04)
Each run is a fresh process: no download, but the model still loads from disk each time (load time isn't included in tokens/sec). The output text was the same on every run (greedy decoding).

| Run | Prompt (43 tok) tok/s | Generation (64 tok) tok/s | Peak memory |
|---|---|---|---|
| 1 | 689.2 | 245.1 | 0.799 GB |
| 2 | 745.8 | 250.3 | 0.799 GB |
| 3 | 819.2 | 246.3 | 0.799 GB |
| 4 | 780.2 | 251.7 | 0.798 GB |
| 5 | 810.3 | 250.0 | 0.799 GB |
| 6 | 734.1 | 242.1 | 0.799 GB |
| 7 | 749.4 | 245.2 | 0.799 GB |
| 8 | 823.3 | 254.7 | 0.799 GB |
| 9 | 830.5 | 254.5 | 0.798 GB |

- Prompt tok/s: 775.8 ± 49.0 (median 780.2, min 689.2, max 830.5)
- Generation tok/s: 248.9 ± 4.4 (median 250.0, min 242.1, max 254.7)
- Peak memory: 0.798–0.799 GB
