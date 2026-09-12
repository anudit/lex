# lex large training target

This directory defines a separate model with a 111,000-byte weight budget covering every
canonical grammar bundled by Highlight.js 11.12.0. The checked-in manifest pins
**185 grammars** from release commit
`f7f7d3803bd898e37c017ffb881317f0cde04a70`; aliases such as `html` remain covered
by their canonical grammar (`xml`) and are not counted twice. Eight grammars
(`erlang-repl`, `julia-repl`, `python-repl`, `clojure-repl`, `node-repl`,
`php-template`, `xl`, `excel`) are excluded in `sync_languages.mjs` — the REPL
variants and `php-template` are fetched through the exact same extension as a
plain sibling grammar (`.erl`/`.jl`/`.py`/`.clj`/`.js`/`.php`), so their
"corpus" is just relabelled sibling content, not real REPL transcripts or
embedded HTML+PHP templates; `xl`'s own Highlight.js grammar mislabels 73-97%
of every real `.xl` file as a comment regardless of content, a broken teacher
rather than a data problem; `excel`'s only extension is `.xlsx`, a binary
zip-based (OOXML) format that can never decode as UTF-8 source, so its corpus
was structurally 0 tokens no matter how much was fetched.

See [LOG_ANALYSIS.md](LOG_ANALYSIS.md) for the evidence behind the design.
See [note.txt](note.txt) for the complete local dataset, Hugging Face, RTX
training, evaluation, and export commands.

## Default student architecture

| component | configuration |
| --- | --- |
| hidden sequence width | 96 |
| factorized embedding | 64 |
| layers | 4 bidirectional selective BidiGLU blocks |
| local context | depthwise kernel 7, dilations 1/2/4/8 |
| document conditioning | masked mean/max signature, FiLM rank 64 per layer |
| recurrent reset | directional rank-32 input-conditioned erase |
| classifier | 192-wide GELU head, 9 output classes |
| lexical hashes | 1024 primary, 256 secondary |
| structural inputs | paren/brace/bracket depth, line position, indent, quote state |
| quantization | 3-bit embedding/input projection, 2-bit hidden classifier, 3-bit output classifier, binary backbone, 4-bit depthwise kernels |
| QAT gradients | clipped binary straight-through estimator; derived binary scales detached in backward |
| exact packed weights | **110,352 bytes / 107.77 KiB** |
| parameters | **453,866** |

All matrix widths and ranks are multiples of 32, so packed rows do not waste a
partial word. The extra budget is concentrated in shared representations rather
than 185 language-specific heads. Language identity is inferred from the file
signature and routed continuously through FiLM, preserving the API's no-language-
argument behavior.

`wgsl_large.py` supports this configuration, including mixed-precision decoding
and kernel 7. The MPS smoke screen selected this candidate at 66.84% weighted
token accuracy, versus 66.36% for corrected gradients with the old architecture
and 61.45% for the old gradient behavior. These are single-seed, 1,600-step
supervised runs without distillation, not final model accuracy estimates.

`NeuralLexer()` and `train.py` use the new student defaults. `LexerConfig()`
retains historical fallback fields so older checkpoints that omit precision
settings still load correctly; use `student_config()` for the new configuration.
Resuming a checkpoint uses its stored architecture. Changing defaults does not
replace the packaged weights; train and validate a new checkpoint before bundling.

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
  large corpus and relabelled under the 185-language manifest. Aliases such as
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

The teacher remains fully unquantized: width 192, embedding width 96, four layers,
384-wide head, FiLM rank 96, erase rank 32, and now kernel 7. Its weight budget is
disabled. Its capacity has not been increased based on student smoke results.
After retraining the teacher, regenerate its logits cache before training the
student; an existing cache is not automatically refreshed when defaults change.

Cache the teacher outputs once, then train the 110,352-byte student. This avoids a
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
to 0.15 because a 185-way identity target otherwise overwhelms the token
objective.

## Acceptance gates

- Packed `weights.bin` is at most 111,000 bytes.
- All 185 grammars have train, validation, and test data with at least 300k total
  tokens before splitting.
- Report popularity-weighted, macro-language, micro, and boundary metrics; do
  not hide an unsupported grammar inside the weighted aggregate.
- Require no more than a 2-point gap between Shiki-labelled and Highlight.js-
  fallback subsets after controlling for language weight.
- Validate NumPy and WebGPU parity before promotion, then measure single-call and
  batched latency independently of the compact model.
