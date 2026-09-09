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


## Latency

Median of per-snippet medians, 25 timed iterations after warmup, same machine and run:

- **Single call** — lex 2.55 ms, gpu-lexer 2.20 ms.
- **Ten blocks in one turn** — **lex 2.50 ms**, gpu-lexer 2.70 ms.

~96% of a call is the GPU readback round-trip, and it is nearly flat in input
size. Issue calls together and they share one submit and one readback:

```js
const out = await Promise.all(blocks.map((b) => lex.highlight(b)));
```

## Architecture vs gpu-lexer

- **Parameters** - 153,609 vs 41,321.
- **Size** - 34.9 KB vs ~31 KB packed.
- **Quantization** - quantization-aware 3-bit embedding / 1-bit projections / 4-bit depthwise, stored as bit-planes. gpu-lexer applies 6-bit zigzag *after* training.
- **Hidden dim** - 64, embedding factorized through 32. gpu-lexer uses 32.
- **Word-hash buckets** - 512 primary / 128 secondary, vs 256 / 128.
- **Local context** - depthwise conv, kernel 5. gpu-lexer uses a ±2 window plus nearest non-whitespace neighbours.
- **Sequence model** - 3 stacked bidirectional gated linear recurrences, vs 1.
- **Memory** - decay rates initialized across **2–200 tokens**, so some channels track a block comment opened hundreds of tokens back. gpu-lexer uses a single rate.
- **File context** - masked mean, max, **prefix mean and suffix mean**. The last two vary by position, so the model can distinguish "declared earlier" from "appears from nowhere". gpu-lexer uses a binary tree reduction.
- **Conditioning** - **FiLM**: a document signature pooled *before* the layers produces per-layer scale and shift, so "this file is Markdown" reaches the conv and the recurrence rather than only the classifier. gpu-lexer has none.
- **WebGPU** - 18 entry points, one workgroup per 16-token tile, all in one submit and one readback. gpu-lexer uses 7 compute passes.

## Training

- **Corpus - 24,238,712 tokens** from 12,638 files across ~330 repositories. gpu-lexer: 4,688,781 tokens.
- **Languages - 33** (Sugar High's 29 plus the GitHub top-25). gpu-lexer demonstrates ~10.
- **Teacher** - Shiki.
- **Schedule** - 30 epochs on Apple Silicon (MPS): 2 full-precision warmup epochs, then 28 under quantization-aware training. AdamW, one-cycle, inverse-sqrt class weights.
- **Auxiliary language head** - attached to the document signature during training only, reaches 99%, never ships. The FiLM conditioning is only as good as the signature it reads.
- **Splits by file**, assigned from a hash of the filename, so near-identical chunks cannot leak between train and validation.

## Requirements

WebGPU - Chrome 113+, Edge 113+, Safari 18+, Firefox 141+. `isSupported()` lets you fall back; there is no CPU path.

## License

MIT - see [LICENSE](LICENSE).
