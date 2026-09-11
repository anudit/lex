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

Popularity-weighted character agreement with Shiki in the reproducible demo:

- **Unseen repositories** - lex 88.10%, gpu-lexer 74.65%, Prism.js 71.73%.
- **Held-out files** - lex 90.24%, gpu-lexer 74.83%, Prism.js 70.32%.

The training pipeline's separate top-25 token benchmark selected epoch 34 at
86.04% weighted accuracy, with 89.57% boundary F1.

## Latency

Median of per-snippet medians, 25 timed iterations after warmup, same machine and run:

- **Single call** - **lex 1.65 ms**, gpu-lexer 2.20 ms.
- **Ten blocks in one turn** - **lex 2.60 ms**, gpu-lexer 4.20 ms.

~96% of a call is the GPU readback round-trip, and it is nearly flat in input
size. Issue calls together and they share one submit and one readback:

```js
const out = await Promise.all(blocks.map((b) => lex.highlight(b)));
```

## Architecture vs gpu-lexer

- **Parameters** - 142,274 vs 41,321.
- **Size** - 36.15 KiB vs ~31 KiB packed.
- **Quantization** - quantization-aware 3-bit embedding / 1-bit projections / 4-bit depthwise, stored as bit-planes. gpu-lexer applies 6-bit zigzag *after* training.
- **Hidden dim** - 64, embedding factorized through 32. gpu-lexer uses 32.
- **Word-hash buckets** - 512 primary / 128 secondary, vs 256 / 128.
- **Local context** - depthwise conv, kernel 5. gpu-lexer uses a ±2 window plus nearest non-whitespace neighbours.
- **Sequence model** - 3 stacked bidirectional gated linear recurrences, vs 1.
- **Memory** - directional decay banks retain long context, while a rank-8
  input-conditioned erase projection can clear selected channels at closing
  delimiters instead of waiting for passive decay. gpu-lexer uses a single rate.
- **File context** - masked mean, max, **prefix mean and suffix mean**. The last two vary by position, so the model can distinguish "declared earlier" from "appears from nowhere". gpu-lexer uses a binary tree reduction.
- **Conditioning** - **FiLM**: a document signature pooled *before* the layers produces per-layer scale and shift, so "this file is Markdown" reaches the conv and the recurrence rather than only the classifier. gpu-lexer has none.
- **WebGPU** - 18 entry points, one workgroup per 16-token tile, all in one submit and one readback. gpu-lexer uses 7 compute passes.

## Training

- **Corpus - 55,343,547 tokens** from 37,420 files. gpu-lexer: 4,688,781 tokens.
- **Languages - 52** in the primary training sampler; all 57 cached languages
  remain in diagnostic evaluation. gpu-lexer demonstrates ~10.
- **Teacher** - Shiki.
- **Schedule** - 40 epochs on Apple Silicon (MPS): 4 full-precision warmup
  epochs, then 36 under quantization-aware training, with natural-popularity
  calibration for the final 8 epochs.
- **Objectives** - class-balanced and boundary-weighted token loss, plus
  training-only language and structural-state heads that never ship.
- **Checkpoint selection** - the best quantized epoch is selected on the frozen
  gpu-lexer verification corpus, not the internal validation split.
- **Splits by file**, assigned from a hash of the filename, so near-identical chunks cannot leak between train and validation.

## Requirements

WebGPU - Chrome 113+, Edge 113+, Safari 18+, Firefox 141+. `isSupported()` lets you fall back; there is no CPU path.

## License

MIT - see [LICENSE](LICENSE).
