# train — lex-lite (v2)

Corpus, training, export and WGSL generation for `lex`. Python deps are managed
with `uv` (`pyproject.toml`); the labeller needs Node (`package.json`).

## Pipeline

```sh
# 1. corpus and Shiki labels (already cached under corpus/)
uv run python fetch_corpus.py
uv run python build_labels.py

# 2. dataset cache: whitespace-free v2 features, dialect-filtered, content-hash splits
uv run python data_pipeline.py --total-tokens 42000000 --workers 10   # -> corpus/dataset_v2

# 3. teacher -> distilled student -> fine-tune; best ends up in checkpoints_v2
./run_v2.sh              # or: ./run_v2.sh teacher | student | finetune

# 4. export, fixtures, bundle into ../lex/src, tests, evaluations
uv run python finalize.py
```

Then, with `bun run dev` in `../demo`:
`/validate.html` checks the shader against the numpy reference on a real GPU, and
`/capture.html` re-measures the benchmark into `demo/src/results.json`
(regenerate its corpora with `make_demo_corpus.py` first if the corpus changed).

## Files

| File | Role |
| --- | --- |
| `tokenizer.py` | CPU pre-tokenizer; `tokenize_v2` is the model's input (mirrored by `lex/src/tokenizer.js`) |
| `data_pipeline.py` | label alignment, dialect filter, dedup, content-hash splits, cached windows |
| `model.py` | the network (`LexerConfig()` defaults are the shipped architecture) |
| `quant.py` | quantizers shared by training and export, including 8-bit scalars |
| `train.py` | training loop: QAT, distillation, auxiliary heads, sampling, selection |
| `run_v2.sh`, `run_v2_finetune.sh` | the full training recipe |
| `export.py` | `weights.bin` + `weights.meta.json`, with a round-trip check |
| `reference.py` | numpy forward pass from the exported file — the shader's specification |
| `wgsl.py` | WGSL generator and minifier |
| `bundle_lex.py` | writes `lex/src/weights.js`, `shader.js`, `shader.wgsl` |
| `make_refcases.py` | shader fixtures for `demo/validate.html` |
| `make_demo_corpus.py` | demo benchmark corpora |
| `eval.py`, `eval_real_bench.py`, `diagnose.py`, `lang_health.py` | evaluation and diagnostics |
| `test_all.py` | test suite |

## Checkpoints

| Directory | Contents |
| --- | --- |
| `checkpoints_v2` | shipped model (91.27% on the gpu-lexer verification corpus) |
| `checkpoints_v2_teacher` | full-precision teacher (92.66%) |
| `checkpoints_v2_student` | 40-epoch distilled student (90.96%) |
| `checkpoints_v2_ft1` | first fine-tuning round (91.06%) |

`../train_large` reuses `model.py`, `train.py`, `export.py`, `quant.py` and
`data_pipeline.py` with the base feature layout (`--feature-version 1`,
`--ctx-views full`, `--scalar-bits 16`); its wrapper sets those flags.
