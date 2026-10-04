# CLI repeat runs (manual, 2026-10-04)

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
