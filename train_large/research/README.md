# Updated-corpus architecture experiments

This investigation uses the local 185-language corpus and the MPS backend on
2026-09-12. It does not change the production student defaults or packaged
weights. Numerical results are recorded under `../smoke_results/`; the final
comparison is summarized below after the runs complete.

## What accuracy means here

The numeric cache contains 145,368,132 tokens from 100,460 files and 298,959
training windows. Its language IDs match the current manifest. Labels come from
Shiki for 96 languages and Highlight.js for 89. Eighteen languages still have
fewer than 300,000 tokens; subunit has only 1,736.

The primary smoke metric is per-language token accuracy combined with the
repository's 75% popularity / 25% uniform weighting over 185 languages. Micro
accuracy and per-class recall are also recorded. This is not the demo's
non-whitespace character agreement with Shiki. The old 193-language results
are not a controlled baseline for this corpus.

The budgeted control has 110,352 packed bytes. We retain the existing
111,000-byte hard cap when interpreting approximately 110 KB. Full-precision
diagnostics are explicitly outside this budget, not eligible student candidates.

## Experiment design

- Real 512-token windows, at most 48 sampled training windows and 8 validation
  windows per language. Selection is seeded and uses the original cached splits.
- 1,600 steps, batch 8, 256 full-precision warmup steps followed by QAT.
- Identical sampled batches across candidates; initialization seeds are recorded
  separately from the data seed for repeat experiments.
- Boundary-weighted cross-entropy, language auxiliary loss 0.15, structural
  auxiliary loss 0.2, AdamW and cosine learning-rate decay.
- No pretrained neural teacher or cached distillation logits are available here.
- MPS embedding-gradient accumulation is not fully deterministic. A single-seed
  difference is screening evidence, not a conclusive architecture ranking.
- The frozen test split is not used for architecture selection. These are early
  training runs, not convergence studies or evidence of an accuracy ceiling.

| Experiment | Hypothesis | Shipping weight bytes |
| --- | --- | ---: |
| `r_control` | Current mixed-precision, erase32, kernel7 student | 110,352 |
| `r_local` | Dilations 1/1/2/4 better preserve local syntax | 110,352 |
| `r_row` | Per-row embedding scales justify reducing FiLM rank to 32 | 109,742 |
| `r_repeat` | Apply the same four blocks twice for more computation per weight | 110,352 |
| `r_jepa` | Masked latent prediction improves representations during training | 110,352 |
| `r_ce` | Uniform class weights improve the target accuracy metric | 110,352 |
| `r_soft_ce` | A gentler inverse-frequency exponent, 0.25 rather than 0.5 | 110,352 |
| `r_rms` | Clip embeddings at 2.5 times field RMS instead of field maximum | 110,352 |
| `r_precision` | Width64, 2-bit backbone, 4-bit embeddings trade width for precision | 108,496 |
| `r_float` | Diagnose quantization versus short-run optimization | 1,815,464 at FP32; ineligible |

`r_repeat` is weight sharing, not diffusion. `r_jepa` is a small auxiliary
experiment inspired by masked latent prediction, not a reproduction of I-JEPA
or data2vec. It masks 15% of word positions, predicts normalized hidden vectors
from an EMA target network, and adds a 0.1-weight Huber loss. The predictor and
EMA network are discarded at deployment. Repetition adds inference computation;
JEPA adds training computation. Equal steps therefore do not imply equal compute.

RMS clipping is a fixed clipping rule, not learned-step-size quantization.
The width64 candidate uses ordinary 2-bit weights, not BitNet ternary encoding.
Experimental checkpoints include their variant settings; checkpoints requiring
a different inference graph or quantizer are rejected by the production export
entry point. Export support and browser latency must be checked before promotion.

## Relevant research and its practical implications

**FastGRNN is genuinely kilobyte-scale recurrent modeling.** It shares gate
matrices and combines low-rank structure, sparsity, and quantization; the paper
demonstrates a 1 KB wake-word model. These are useful parameter-efficiency ideas,
but wake-word classification is not 185-language token labeling. Our selective
recurrence and low-rank erase already use related economical mechanisms.
[Microsoft Research](https://www.microsoft.com/en-us/research/publication/fastgrnn-a-fast-accurate-stable-and-tiny-kilobyte-sized-gated-recurrent-neural-network/)

**Magika supports a cheap document-level signal, not a 99% lexer claim.** Its
original model samples 512 bytes from the beginning, middle, and end of a file,
embeds byte chunks, applies small dense layers and global max pooling. It reports
99% file-type F1 with approximately 1 MB of weights. The transferable idea is
conditioning token processing on a useful file signature, which this lexer
already does through FiLM. Its task and budget are different.
[Paper](https://arxiv.org/html/2409.13768v1)

**BitNet's small-model evidence supports precision/capacity ablations.**
The “Reloaded” study investigates ternary quantization and mean/median scaling.
The 100K-parameter example is MNIST; its language models start at 6M parameters.
It finds that quantized language models can need greater hidden width to match
full-precision counterparts. This is evidence for testing the allocation of
precision, not a guarantee that a 110 KB lexer retains full-precision quality.
Also, ternary values occupy two bits in the current bit-plane codec; 1.58 is
not automatically the actual packed cost here.
[Paper](https://arxiv.org/html/2407.09527v1)

**ALBERT motivates shared computation.** Cross-layer parameter sharing reduces
stored parameters while retaining repeated transformations. Our two-pass
backbone experiment tests that idea without adding weights. It can still be
slower and harder to optimize, so bytes and runtime must both be reported.
[Google Research](https://research.google/blog/albert-a-lite-bert-for-self-supervised-learning-of-language-representations/)

**JEPA/data2vec are primarily representation-learning objectives.** I-JEPA
predicts masked image representations; data2vec includes text and predicts
contextual latent targets from masked inputs. Neither result establishes
accuracy for this lexer. A training-only auxiliary objective is a lower-cost
test than replacing the output classifier: syntax highlighting still needs
exact local labels, including punctuation and boundaries.
[I-JEPA](https://arxiv.org/abs/2301.08243),
[data2vec](https://arxiv.org/abs/2202.03555)

**Diffusion is not the first architecture to pursue here.** Discrete diffusion
can predict multiple generated tokens in parallel, and recent methods accelerate
it with blockwise autoregression, caching and distillation. Our discriminative
lexer already reads the complete source and predicts every label in one pass.
My inference is that iterative denoising would need a measured consistency gain
to justify extra passes; generation-speed comparisons against autoregressive
LLMs do not establish such a gain. No diffusion model was trained in this study.
[Discrete Diffusion Forcing](https://arxiv.org/abs/2508.09192)

**Learned quantization and richer distillation remain strong follow-ups.** LSQ
learns quantizer step sizes with scaled surrogate gradients. TinyBERT distills
intermediate transformer representations in addition to task behavior. A lexer
could similarly distill contextual states or boundary behavior. These would
require controlled implementations and a well-trained teacher on this corpus;
they are proposals, not measured gains from the current experiments.
[LSQ](https://arxiv.org/abs/1902.08153),
[TinyBERT](https://arxiv.org/abs/1909.10351)

## Final results

All figures are single-seed final-step smoke measurements unless a candidate
was repeated, in which case the mean and standard deviation across seeds
20260913, 20260914 and 20260915 are reported (same data seed and batch order
throughout, so only initialization differs). Sorted by weighted token
accuracy.

| Candidate | Packed bytes | Weighted accuracy | Micro accuracy | Seeds |
| --- | ---: | ---: | ---: | ---: |
| `r_soft_ce`: inverse-frequency exponent 0.25 | 110,352 | 67.4424% ± 0.6034% | 58.0884% | 3 |
| `r_jepa`: masked latent auxiliary | 110,352 | 66.8522% | 53.1173% | 1 |
| `r_row`: per-row embedding scales + FiLM32 | 109,742 | 66.8010% ± 0.2379% | 54.3158% | 3 |
| `r_float`: FP32 diagnostic (not budget-eligible) | n/a (1,815,464 FP32 bytes) | 66.7752% | 53.7389% | 1 |
| `r_rms`: fixed 2.5-RMS embedding clipping | 110,352 | 66.3198% | 53.5267% | 1 |
| `r_control`: current production architecture | 110,352 | 66.1641% ± 0.0487% | 53.0388% | 3 |
| `r_local`: dilations 1/1/2/4 | 110,352 | 66.1034% | 53.3395% | 1 |
| `r_repeat`: same four blocks applied twice | 110,352 | 65.2713% | 52.9997% | 1 |
| `r_precision`: width64, 2-bit backbone | 108,496 | 64.9623% | 52.3986% | 1 |
| `r_ce`: uniform class weights | 110,352 | 64.5077% | 64.7343% | 1 |

Per-class recall (plain, comment, string, number, keyword, type, function,
constant, operator), mean across available seeds, for the candidates under
active consideration:

| Candidate | function | constant | operator |
| --- | ---: | ---: | ---: |
| `r_control` | 51.64% | 20.73% | 83.47% |
| `r_row` | 51.39% | 20.39% | 81.12% |
| `r_jepa` | 53.32% | 19.73% | 83.21% |
| `r_soft_ce` | 38.26% | 11.96% | 78.28% |

Reading the aggregate table alone is misleading:

- `r_soft_ce` has the highest weighted score across all three seeds (not just
  one), but confirms the earlier single-seed warning: function recall is
  roughly 13 points lower and constant recall is roughly 9 points lower than
  `r_control`. This is a real, repeatable rare-class regression, not noise.
  It should not be promoted on the aggregate number.
- `r_row` repeats its single-seed result across three seeds: a modest but
  consistent ~0.6 point weighted gain over `r_control`, at fewer packed bytes
  (109,742 vs 110,352), with per-class recall essentially unchanged from
  control (no regression on function/constant/operator). This is the most
  defensible finalist from this screen. The FiLM32-only ablation needed to
  isolate the scale effect from the FiLM-rank change was still not run.
- `r_jepa` (training-only, no extra inference cost or bytes) shows the best
  function recall of any candidate and a competitive weighted score, but was
  only run at one seed; it needs repeat seeds before being taken seriously
  as a finalist.
- `r_control`'s own three-seed spread (66.10%–66.18%, std ≈ 0.05%) sets the
  noise floor for this protocol: differences much under a point, including
  `r_row`'s ~0.6 point gain, are close to that floor and should be treated
  as suggestive, not conclusive, without more seeds or longer runs.
- `r_float` (FP32 diagnostic) does not outscore the quantized candidates by
  much, so the current quantization scheme is not obviously the accuracy
  bottleneck at this step count.
- `r_precision` (width64, 2-bit) and `r_rms` (fixed RMS clipping) each
  underperform `r_control` at one seed and are not promising leads by
  themselves.

## Recommendation

**No configuration in this screen reaches, or comes close to, >90% weighted
token accuracy.** The best three-seed mean (`r_soft_ce`, 67.44%) is a rare-class
regression, and the best finalist that preserves per-class recall (`r_row`,
66.80%) is only ~0.6 points above production. Production defaults are left
unchanged: this was a research screen, not a validated improvement, and none
of these candidates has had exact export byte verification, checkpoint reload
fidelity, or browser parity/latency checks against the current shader.

If further work is invested, in priority order:

1. Repeat `r_jepa` across the same three seeds before deciding whether its
   function-recall gain is real.
2. Run the outstanding FiLM32-only ablation to isolate `r_row`'s scale effect
   from its FiLM-rank change, since `r_row` is currently the most defensible
   accuracy finalist.
3. Treat `r_soft_ce` as informative about the accuracy/rare-class tradeoff of
   loss weighting, not as a candidate to ship as-is; a milder exponent between
   0.25 and 0.5, or per-class floors, may recover rare-class recall without
   giving up all of the weighted gain.
4. None of this establishes an accuracy ceiling for the architecture. Getting
   toward >90% plausibly requires a converged teacher/student run with real
   distillation on the updated corpus, consistent token-vs-character
   evaluation, genuinely held-out repositories, and per-class/boundary
   checks — all still open, per the handoff's next actions.

## Reproduction

From the repository root, using the Python environment with PyTorch and MPS:

```sh
train/.venv/bin/python -u train_large/smoke_mps.py \
  --out train_large/smoke_results/corpus185_screen_s1 \
  --candidates r_control r_ce r_soft_ce r_local r_row r_repeat r_jepa r_float \
  --steps 1600 --warmup 256 --batch-size 8 \
  --train-per-lang 48 --val-per-lang 8 \
  --seed 20260913 --data-seed 20260913
```

Output directories must be new. Each contains the sampled window IDs, batch
order fingerprint, dataset metadata fingerprint, checkpoints, timings, and
per-language/per-class validation records.
