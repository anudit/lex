# lex - neural syntax highlighting on WebGPU

| path | what |
| --- | --- |
| [`lex/`](lex/) | the library - zero-dependency WebGPU engine + React bindings |
| [`train/`](train/) | corpus, training, export, WGSL generator, diagnostics |
| [`demo/`](demo/) | head-to-head benchmark vs gpu-lexer, Shiki, Prism, Sugar High |

```js
import { createLexer } from 'lex';
const lex = await createLexer();
const spans = await lex.highlight(code);   // [{ type, start, end }, ...]
```

## Accuracy

Popularity-weighted character agreement with Shiki in the reproducible demo
(`demo/src/results.json`):

- **Unseen repositories** (1,196 files) - **lex 93.76%**, gpu-lexer 80.75%,
  Prism.js 73.28%.
- **Held-out files** (1,050 files) - **lex 94.03%**, gpu-lexer 80.76%,
  Prism.js 71.00%.
- **gpu-lexer's own verification corpus, scored gpu-lexer's own way** (1,101
  held-out files across the top-25 GitHub-popularity-weighted languages, using
  gpu-lexer's scope-to-class taxonomy and confidence-gated denominator;
  gpu-lexer@0.0.3) - **lex 89.36%**, gpu-lexer 86.75%, Prism.js 80.14%.

The demo corpora exclude files whose extension does not match the grammar they
would be labelled with (`.scss` under CSS, `.tsx` under TypeScript, `.mdx` under
Markdown). With lex's token-level evaluator (`train/eval_real_bench.py`) the
model scores 91.27% on the verification corpus.

## Size

- **Weights** - 28,762 bytes packed (28.09 KiB); 27.75 KiB of `weights.js`
  after Brotli.
- **Library** - 34.78 KiB for the whole package, minified, after Brotli
  (weights, runtime, tokenizer and a minified WGSL shader). gpu-lexer@0.0.3:
  27.64 KiB.

## Latency

Median of 25 timed iterations after warmup, over four runs on one machine
(Apple M2 Max, Chrome):

- **Single call** - lex 1.1-3.3 ms, gpu-lexer 0.6-1.6 ms.
- **Ten blocks in one turn** - lex 1.2-4.2 ms, gpu-lexer 0.6-2.1 ms.

Most of a call is the GPU readback round-trip, and it is nearly flat in input
size. Issue calls together and they share one submit and one readback:

```js
const out = await Promise.all(blocks.map((b) => lex.highlight(b)));
```

## Architecture vs gpu-lexer

- **Parameters** - 127,682 vs 41,321.
- **Packed weights** - 28.09 KiB vs ~31 KiB.
- **Quantization** - quantization-aware 3-bit embedding / 1-bit projections / 4-bit depthwise, stored as bit-planes; scales, biases, norms and decays as 8-bit codes. gpu-lexer applies 6-bit zigzag *after* training.
- **Hidden dim** - 64, embedding factorized through 32. gpu-lexer uses 32.
- **Word-hash buckets** - 256 primary / 128 secondary, same as gpu-lexer.
- **Tokens** - words and symbols only; whitespace becomes per-token gap, indent, first-symbol-on-line and bracket-depth fields.
- **Local context** - depthwise conv, kernel 5, dilations 1/2/4. gpu-lexer uses a ±2 window plus nearest non-whitespace neighbours.
- **Sequence model** - 3 stacked bidirectional gated linear recurrences, vs 1.
- **Memory** - directional decay banks retain long context, while a rank-8
  input-conditioned erase projection can clear selected channels at closing
  delimiters instead of waiting for passive decay. gpu-lexer uses a single rate.
- **File context** - declaration-gated **prefix mean and suffix mean**, which vary by position, so the model can distinguish "declared earlier" from "appears from nowhere". gpu-lexer uses a binary tree reduction.
- **Conditioning** - **FiLM**: a document signature pooled *before* the layers produces per-layer scale and shift, so "this file is Markdown" reaches the conv and the recurrence rather than only the classifier. gpu-lexer has none.
- **WebGPU** - 18 entry points, one workgroup per 16-token tile, all in one submit and one readback. gpu-lexer uses 7 compute passes.

See [ARCH.md](ARCH.md) for the full architecture.

## Training

`train/run_v2.sh` runs the whole pipeline.

- **Corpus - 39,685,134 tokens** from 40,780 files, labelled by Shiki. gpu-lexer: 4,688,781 tokens.
- **Languages - 52** in the training sampler; all 57 cached languages remain in
  evaluation.
- **Teacher** - a full-precision model (dim 160, 4 layers) trained first; the
  shipped model learns from its output distribution as well as from the labels.
- **Schedule** - on Apple Silicon (MPS): 40 epochs (4 full-precision warmup,
  36 quantization-aware, natural-popularity calibration for the final 8), then
  two fine-tuning rounds at lower learning rates.
- **Objectives** - class-balanced and boundary-weighted token loss, distillation
  from the teacher, plus training-only language and structural-state heads that
  never ship.
- **Checkpoint selection** - on gpu-lexer's verification corpus, evaluated
  exactly as shipped, not on the internal validation split.
- **Splits by file**, assigned from a hash of the file content, so duplicated
  files cannot leak between train and validation.

## Requirements

WebGPU - Chrome 113+, Edge 113+, Safari 18+, Firefox 141+. `isSupported()` lets you fall back; there is no CPU path.

## License

MIT - see [LICENSE](LICENSE).
