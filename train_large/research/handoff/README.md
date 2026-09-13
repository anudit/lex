# Handoff: updated-corpus tiny lexer research

Snapshot: 2026-09-12 09:03 UTC. Read the live results before continuing; the
background experiment can advance beyond this snapshot.

## User request and scope

The user updated `train_large/corpus` and asked for small local experiments and
ablations to improve the large student under approximately 110 KB, aiming for
greater than 90% accuracy. They also asked for research into similar tiny models
and whether diffusion or JEPA could help. The latest request is to write this
handoff. Do not mistake preliminary smoke scores for achieved production accuracy.

Work is in `/Users/anudit/Documents/GitHub/lexer`. No subagents were used; current
instructions prohibit delegation unless explicitly requested. Use `apply_patch`
for edits. MPS requires execution outside the sandbox on this Mac. The approved
command prefix for training is:

```text
train/.venv/bin/python -u train_large/smoke_mps.py
```

## Existing production configuration — leave intact

Earlier work fixed binary QAT's missing straight-through gradient and made the
selected student the default at the user's explicit request:

- Width 96, embedding width 64, four layers, classifier width 192.
- Embedding 3-bit, input projection 3-bit, hidden classifier 2-bit, output 3-bit.
- Binary backbone projections, 4-bit convolution, kernel 7, dilations 1/2/4/8.
- FiLM rank 64, erase rank 32, corrected binary gradients.
- Exactly **110,352 packed bytes**, against a **111,000-byte hard cap**.
- Teacher remains full precision, width 192, embedding 96, head 384, four layers,
  FiLM96, erase32, kernel7. It was not enlarged to width320.

`student_config()` and `NeuralLexer()` implement the current student defaults.
`LexerConfig()` intentionally retains historical defaults for loading old
checkpoints that omit newer precision settings. Do not casually change that.

The previous implementation also fixed JS/Python Unicode hashing parity and
added mixed-precision/kernel7 support to the large WGSL generator. Earlier
seven-candidate WebGPU parity checks passed, but those checks concern old smoke
checkpoints, not the new research checkpoints.

This investigation has not changed production defaults or packaged weights.

## Updated corpus and evaluation

- 145,368,132 tokens; 100,460 files; 298,959 training windows.
- **185 languages**, not the previous 193. Numeric language IDs were checked
  against the current manifest and match.
- Shiki labels 96 languages; Highlight.js labels 89.
- 18 languages remain below 300K tokens. Examples: subunit 1,736; rsl 9,890;
  profile 14,393; n1ql 15,222. Coverage does not imply sufficient data.
- No neural teacher checkpoint or teacher-logit cache is available locally.

Primary smoke score: language-weighted **token accuracy**, using the repository's
75% popularity / 25% uniform weights over 185 languages. Micro and macro-language
accuracy and nine-class recall are recorded too. This is not the demo's
non-whitespace **character agreement with Shiki**. Earlier 193-language scores
are not controlled comparisons. The frozen test split has not been used.

Current run protocol:

- 1,600 training steps; batch 8; 512-token windows.
- 256 FP warmup steps followed by QAT, except the always-FP diagnostic.
- Up to 48 train and 8 validation windows per language: actual totals are
  **8,799 training windows and 1,447 validation windows**.
- Initialization seed and data seed both 20260913. These are arbitrary RNG seeds,
  not dates of execution.
- Same sampled batch sequence across candidates. `--data-seed` permits changing
  initialization while keeping train/validation windows and batches fixed.
- Boundary-weighted CE; language auxiliary loss 0.15; structure auxiliary 0.2;
  AdamW; cosine decay. No distillation and no full-run weak-language mining or
  final natural-sampling calibration phase.
- MPS accumulation is not completely deterministic. The runner uses deterministic
  algorithms with `warn_only=True`, since strict mode rejects embedding backward.
- More compute is spent by repeat-depth and JEPA variants; same steps are not
  same FLOPs. Timings include training plus the validation passes.

Dataset metadata SHA-256:
`78c686093972384867cd85c808fc031bd8d83cf385da41715315e0ee37b519ca`

Batch sequence SHA-256:
`ad850d31c216e5ac7c9c4031b0566292097c7cfcdcae642c19c307d28bd4c555`

## Background processes

**Main MPS screen is still running at the snapshot.**

- PID **18128**; unified exec session **96529**.
- Output directory: `train_large/smoke_results/corpus185_screen_s1`.
- Last saved result: `r_repeat` completed all 1,600 steps at 65.2713% weighted
  accuracy. The process advances to `r_jepa` next.
- Completed: `r_control`, `r_ce`, `r_soft_ce`, `r_local`, `r_row`, `r_repeat`.
- Remaining in this process: `r_jepa`, then `r_float`.

Exact running command:

```sh
train/.venv/bin/python -u train_large/smoke_mps.py \
  --out train_large/smoke_results/corpus185_screen_s1 \
  --candidates r_control r_ce r_soft_ce r_local r_row r_repeat r_jepa r_float \
  --steps 1600 --warmup 256 --batch-size 8 \
  --train-per-lang 48 --val-per-lang 8 \
  --seed 20260913 --data-seed 20260913
```

Poll session 96529 if it is still accessible; otherwise inspect the PID and
`results.json`. Results and the candidate checkpoint are written after each
completed candidate, not after every step. Do not launch a duplicate screen.
Do not run another MPS training experiment concurrently if comparing timings.

Two older validation processes are also still present:

- PID 5135: Python HTTP server, `127.0.0.1:4184`.
- PID 5149: isolated headless Chrome, debugging port 9224, profile
  `/private/tmp/lexer-architecture-chrome.oyVNRV`.

They serve the earlier WebGPU validation page and are not training processes.
No processes were stopped for this handoff. Validate PIDs before any cleanup.

## Completed new-corpus results

All figures below are **single-seed final-step smoke measurements**.

| Candidate | Packed bytes | Weighted token accuracy | Micro accuracy |
| --- | ---: | ---: | ---: |
| `r_control`: current production architecture | 110,352 | 66.2114% | 53.9449% |
| `r_ce`: uniform class weights | 110,352 | 64.5077% | 64.7343% |
| `r_soft_ce`: inverse-frequency exponent 0.25 | 110,352 | 67.0429% | 57.7423% |
| `r_local`: dilations 1/1/2/4 | 110,352 | 66.1034% | 53.3395% |
| `r_row`: per-row embedding scales + FiLM32 | 109,742 | 66.7700% | 54.5804% |
| `r_repeat`: same four blocks applied twice | 110,352 | 65.2713% | 52.9997% |

Interpretation:

- No configuration has demonstrated >90% accuracy.
- Local dilations have not improved the primary metric in this screen.
- Repeated depth did not improve the primary metric either: 65.27% versus
  66.21%, taking about 156 seconds versus 99 seconds for the control.
- Uniform class weights substantially improve micro accuracy but hurt the primary
  weighted score and rare-class recall. Function recall falls from 49.46% to
  8.15%; constant recall falls from 19.60% to 1.86%.
- Gentler weighting improves the weighted score by 0.83 points, but function
  recall falls to 39.77% and constant recall to 13.64%. Do not promote it based
  on aggregate accuracy alone.
- Per-row scales plus FiLM32 improve weighted accuracy by 0.56 points while
  using fewer bytes. Its function recall is 52.08% and constant recall 22.99%,
  but operator recall decreases. This looks like a useful finalist, subject to
  repeated seeds. It changes two factors; a FiLM32-only ablation is still needed
  to isolate the scale effect.
- Differences below one point need repeated seeds. These short runs cannot
  establish convergence or an architectural accuracy ceiling.

## Research variants and implementation status

`research_variants.py` keeps changes isolated from production:

- `r_repeat`: applies the same four blocks twice, with the original document
  signature/FiLM. No extra weights; greater inference work. This is **not diffusion**.
- `r_jepa`: training-only masked latent prediction. Mask 15% of word positions,
  use an EMA target network (update 0.01), normalized target hidden states,
  a training-only linear predictor, and Huber loss weight 0.1. This is inspired
  by data2vec/JEPA, not a faithful reproduction of either published system.
- `r_float`: same student shape, always FP32. Diagnostic only: **1,815,464 weight
  bytes**, ineligible for the budget. `packed_bytes` is null in its results.
- `r_rms`: implemented and CPU-tested but **not trained in the screen**. Replaces
  per-field max-derived embedding scales with fixed 2.5-RMS clipping; 110,352
  packed bytes. No learned scales; do not call this LSQ.
- `r_precision`: implemented and CPU-tested but **not trained in the screen**.
  Width64, 2-bit backbone, 4-bit embeddings, FiLM32, erase16, otherwise current
  config; **108,496 bytes**. This uses ordinary two-bit quantization, not ternary
  BitNet. The current large shader rejects width64; deployment work would remain.

Research checkpoints carry `research_variant` and `requires_research_loader`.
The latter marks repeat-depth, full precision, and the RMS quantizer. The
production checkpoint exporter now rejects those checkpoints instead of silently
using the wrong graph/quantizer. `prepare_smoke_webgpu.py` skips them. To inspect
them, reconstruct `ResearchLexer` with the checkpoint's variant flags.

## Files changed or added during this investigation

- `train_large/smoke_mps.py`: research variant support, separate data seed,
  language-ID validation, JEPA auxiliary training, richer checkpoint metadata,
  less noisy progress output. Existing original candidate names still work.
- `train_large/research_variants.py`: isolated research models and variant catalog.
- `train_large/summarize_research.py`: combines compatible completed runs and
  rejects mismatched datasets, validation windows, schedules or batch orders.
- `train_large/test_research_variants.py`: research correctness checks.
- `train_large/export_checkpoint.py`: guard against unsupported research loading.
- `train_large/prepare_smoke_webgpu.py`: skip incompatible research checkpoints.
- `train_large/research/README.md`: sourced research notes, experiment design,
  reproduction command; its final results section still needs to be added.
- This handoff report.

The worktree was clean at the beginning of this investigation. These changes
are uncommitted. `smoke_results/` is gitignored; it contains real local artifacts
but they will not be included in a normal commit automatically.

Validation already completed:

- Existing architecture regression suite: **6 tests passed**.
- New research suite: **4 tests passed**.
- Four-step MPS pilot completed for control, repeat-depth, JEPA, and FP diagnostic
  in `smoke_results/corpus185_pilot`; it establishes runnable graphs, not accuracy.
- RMS embedding forward/backward finite and packed export round-trip checked;
  exact byte count 110,352, maximum observed tensor error about 0.00192.
- No current research checkpoint has received a new browser parity/latency run.

The source files were extended with RMS/precision variants after the main screen
started; the already-running process does not include those new candidates.

## Next actions, in order

1. **Finish and inspect the existing screen.** Check live results, not just the
   table above. Record final shared-depth, JEPA, and full-precision scores and
   timings. Do not extrapolate 90% from a short run.
2. **Run the two implemented quantization candidates**, sequentially after the
   main MPS process exits, with exactly the same protocol:

   ```sh
   train/.venv/bin/python -u train_large/smoke_mps.py \
     --out train_large/smoke_results/corpus185_quant_s1 \
     --candidates r_rms r_precision \
     --steps 1600 --warmup 256 --batch-size 8 \
     --train-per-lang 48 --val-per-lang 8 \
     --seed 20260913 --data-seed 20260913
   ```

3. **Repeat the control and actual finalists with additional initialization
   seeds**, retaining `--data-seed 20260913` and all other settings. Suggested
   seeds: 20260914 and 20260915. New output directory per run. Do not simply
   assume the gentler-loss candidate wins; inspect per-class tradeoffs. Consider
   a FiLM32-only control if `r_row` stays promising.
4. Use the summarizer, then add the final table and recommendation to the research
   README. Example (only list directories that actually exist and completed):

   ```sh
   train/.venv/bin/python train_large/summarize_research.py \
     train_large/smoke_results/corpus185_screen_s1 \
     train_large/smoke_results/corpus185_quant_s1
   ```

5. If recommending a new inference architecture, verify exact exported bytes,
   reload fidelity and browser parity/latency before promoting it. This user asked
   for experiments; leave current production defaults unchanged unless justified
   and explicitly communicated. Do not replace shipping weights with smoke models.
6. Explain what is actually needed to assess >90%: a converged teacher/student
   run on the updated corpus, consistent token versus character evaluation,
   genuinely held-out repositories, and per-class/boundary checks. Intermediate
   representation distillation and learned quantization are worthwhile follow-ups,
   not gains already measured here.

## Research conclusions so far

The detailed notes and citations are in `../README.md`. Main distinctions:

- [FastGRNN](https://www.microsoft.com/en-us/research/publication/fastgrnn-a-fast-accurate-stable-and-tiny-kilobyte-sized-gated-recurrent-neural-network/)
  is genuinely kilobyte-scale recurrence; transferable ideas are matrix sharing,
  low rank and quantization, not its task-specific accuracy.
- [Magika](https://arxiv.org/html/2409.13768v1) is whole-file classification at
  roughly 1 MB. Its file signature is relevant; its 99% F1 is not a lexer target
  achieved at our budget.
- [BitNet Reloaded](https://arxiv.org/html/2407.09527v1) supports testing precision
  versus width. Its 100K-parameter example is MNIST; language examples start at
  6M parameters. Ternary encoding would cost two bits in this repository's codec.
- [ALBERT](https://research.google/blog/albert-a-lite-bert-for-self-supervised-learning-of-language-representations/)
  motivates sharing weights across repeated computation.
- [data2vec](https://arxiv.org/abs/2202.03555) supplies a text-relevant precedent
  for masked latent prediction. Our JEPA auxiliary is still experimental.
- [Discrete diffusion](https://arxiv.org/abs/2508.09192) is researched but not
  implemented. The lexer already predicts all labels in parallel, so a speed
  comparison against autoregressive generation does not justify denoising passes.
- [LSQ](https://arxiv.org/abs/1902.08153) and
  [TinyBERT](https://arxiv.org/abs/1909.10351) motivate learned quantizer scales and
  intermediate-state distillation. Neither is implemented in this screen.

No published result found establishes >90% performance on this exact task,
corpus, metric and 110 KB budget. Report that uncertainty plainly.
