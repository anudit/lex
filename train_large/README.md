# lex large training target

This directory defines a separate approximately 100 KiB model covering every
canonical grammar bundled by Highlight.js 11.12.0. The checked-in manifest pins
**193 grammars** from release commit
`f7f7d3803bd898e37c017ffb881317f0cde04a70`; aliases such as `html` remain covered
by their canonical grammar (`xml`) and are not counted twice.

See [LOG_ANALYSIS.md](LOG_ANALYSIS.md) for the evidence behind the design.
See [note.txt](note.txt) for the complete local dataset, Hugging Face, RTX
training, evaluation, and export commands.

## Shipping architecture

| component | configuration |
| --- | --- |
| hidden sequence width | 96 |
| factorized embedding | 64 |
| layers | 4 bidirectional selective BidiGLU blocks |
| local context | depthwise kernel 5, dilations 1/2/4/8 |
| document conditioning | masked mean/max signature, FiLM rank 64 per layer |
| recurrent reset | directional rank-16 input-conditioned erase |
| classifier | 192-wide GELU head, 9 output classes |
| lexical hashes | 1024 primary, 256 secondary |
| structural inputs | paren/brace/bracket depth, line position, indent, quote state |
| quantization | 3-bit embedding, 1-bit projections, 4-bit depthwise kernels |
| exact packed weights | **102,368 bytes / 99.97 KiB** |
| parameters | **434,602** |

All matrix widths and ranks are multiples of 32, so packed rows do not waste a
partial word. The extra budget is concentrated in shared representations rather
than 193 language-specific heads. Language identity is inferred from the file
signature and routed continuously through FiLM, preserving the API's no-language-
argument behavior.

The model and binary exporter are runnable now. The compact WGSL generator is
specialized for width 64 and a 96-wide head, so promotion of this target also
requires a generalized 96-wide WebGPU lowering and browser parity suite. A large
checkpoint should not be copied into `lex/` until that parity work is complete.

## Data composition

The default target is **160 million labelled lexer tokens**. Allocation is exact:

- Every grammar receives at least **300,000 tokens** (57.9M total floor).
- Remaining tokens follow the square root of a 75/25 mixture of GitHub usage and
  uniform grammar coverage.
- Popular languages therefore retain more examples without starving the long
  tail or letting JavaScript dominate the optimizer.
- Source archives keep the existing per-repository cap, 30 KB file truncation,
  generated/vendor filtering, content-hash file splits, and structural-window
  deduplication.
- Compatible raw files from the completed compact run are hardlinked into the
  large corpus and relabelled under the 193-language manifest. Aliases such as
  `shell` to `bash` and `html` to `xml` are mapped without duplicating disk data.
- Aim for at least 12 repositories and 250 files per grammar. Sparse historical
  languages should use upstream compiler/library suites plus Highlight.js and
  GitHub Linguist fixtures; synthetic snippets should stay below 5%.
- Label with Shiki where a bundled grammar exists. Use Highlight.js for the
  remaining canonical grammars and as a per-file fallback when Shiki rejects an
  input, normalized into the same nine output classes.

At the observed compact-corpus density, expect roughly 1.0-1.2 GB of UTF-8 source,
100k-140k source files, about 330k fixed-length windows, and a dataset cache near
14 GB. Keep 20% disk headroom during labeling and cache construction.

## Setup

```bash
cd /path/to/lexer
uv venv train_large/.venv --python 3.12
uv pip install --python train_large/.venv/bin/python -r requirements.txt
cd train_large
bun install
bun run sync-languages

# Extend the checked-in popular-language seeds to the full Linguist-backed set.
# GITHUB_TOKEN is strongly recommended because GitHub search is rate limited.
GITHUB_TOKEN=... .venv/bin/python discover_repos.py

.venv/bin/python bootstrap_fixtures.py
.venv/bin/python fetch_corpus.py
.venv/bin/python build_labels.py --shards 8
.venv/bin/python build_dataset.py
.venv/bin/python test_setup.py
```

`discover_repos.py` is resumable and writes `repos.json` after every language.
Review that file before fetching: repository search is a candidate generator,
not a substitute for checking language purity and source diversity.
`bootstrap_fixtures.py` adds the pinned Highlight.js conformance cases and GitHub
Linguist samples. They improve rare-construct coverage but are never duplicated
to manufacture the 300k-token floor; `build_dataset.py` refuses an underfilled
language unless `--allow-underfilled` is explicitly supplied for a smoke run.

## Training

Train the unquantized teacher first:

```bash
.venv/bin/python -u train_teacher.py \
  2>&1 | tee train_teacher.log
```

Cache the teacher outputs once, then train the 99.97 KiB student. This avoids a
second teacher forward pass during every one of the 48 student epochs:

```bash
.venv/bin/python -u cache_teacher_logits.py \
  2>&1 | tee cache_teacher_logits.log

.venv/bin/python -u train.py \
  2>&1 | tee train_student.log
```

Audit the frozen test split, including the two teacher subsets:

```bash
.venv/bin/python eval_coverage.py \
  --checkpoint ./checkpoints_student/best_model.pt
```

The student defaults to 48 epochs: six FP warmup epochs, 34 broadly tempered QAT
epochs, and eight natural-calibration epochs. CUDA defaults are BF16 autocast,
`torch.compile(mode="max-autotune")`, fused AdamW, pinned/prefetched input, and
batch size 64 for a 16 GiB RTX 5060 Ti. The language auxiliary loss is reduced
to 0.15 because a 193-way identity target otherwise overwhelms the token
objective.

## Acceptance gates

- Packed `weights.bin` is at most 100 KiB.
- All 193 grammars have train, validation, and test data with at least 300k total
  tokens before splitting.
- Report popularity-weighted, macro-language, micro, and boundary metrics; do
  not hide an unsupported grammar inside the weighted aggregate.
- Require no more than a 2-point gap between Shiki-labelled and Highlight.js-
  fallback subsets after controlling for language weight.
- Validate NumPy and WebGPU parity before promotion, then measure single-call and
  batched latency independently of the compact model.
