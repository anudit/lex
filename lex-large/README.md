# lex-large

Syntax highlighting from a 99.97 KiB neural model running on WebGPU, covering
193 Highlight.js grammars. This is the larger sibling of
[`lex`](../lex) (36.15 KiB, 52 languages, referred to as **lex-lite** where the
two are compared) -- same API, same zero-dependency, no-grammar approach, more
parameters and wider language coverage.

```js
import { parse } from 'lex-large';

const spans = await parse(code);
// [{ type: 'keyword', start: 0, end: 6 }, ...]
```

`parse` matches [gpu-lexer](https://github.com/shuding/gpu-lexer)'s API
exactly, the same way `lex`'s does. For an explicit session, per-token classes
instead of merged spans, or a debug-stage hook, use the lower-level API below.

```js
import { createLexer } from 'lex-large';

const lex = await createLexer();
const spans = await lex.highlight(code);
```

React:

```jsx
import { Lex } from 'lex-large/react';
import 'lex-large/theme.css';

<Lex code={source} />
```

## How it works

Same architecture family as `lex` -- no grammar, a CPU pre-tokenizer, a small
recurrent model reading sparse per-token features and a pooled document
signature that FiLM-modulates every layer -- just wider: 96-dim hidden state
(vs. 64), 64-dim embedding (vs. 32), 192-wide classifier head (vs. 96), and
four recurrent layers with dilations 1/2/4/8 (vs. three layers at 1/2/4). It
was trained on a 193-language target set (`train_large/`) rather than `lex`'s
52-language primary set, so it covers most of the Highlight.js grammar list
rather than the top of it.

The model has 434,602 parameters, quantized the same way as `lex` (3 bits for
the embedding table, 1 bit for projections, 4 bits for the depthwise kernels).
It ships inline with no model fetch.

## WebGPU shader

`lex-large`'s WGSL kernels are **not** `lex`'s shader retargeted with new
constants -- they're a bespoke generator
(`train_large/wgsl_large.py`, forked from `train/wgsl.py`) written
specifically for this checkpoint's tensor widths (a 96-wide `DIM` dot product,
a 192-wide `2*DIM`/`HEAD_HIDDEN` dot product, a 384-wide/12-word
`global_ctx.summary` projection, and workgroups sized to 96 lanes rather than
64). `lex`'s generator is intentionally left untouched and un-generalized;
see `train_large/note.txt`'s "IMPORTANT DEPLOYMENT LIMIT" note and
`wgsl_large.py`'s module docstring for why the two stay separate rather than
merged into one parametrized generator.

## Accuracy

Scored the same way as `lex`: agreement with [Shiki](https://shiki.style)
over held-out files, per non-whitespace character, weighted by GitHub
language popularity. `bun run dev` in `demo/` reproduces it, including the
193-language corpus bench under `train_large/corpus`.

## Performance notes

Same batching and shared-device behavior as `lex` -- see that package's
README for the details, which apply unchanged here.

## API

Identical surface to `lex`: `parse`, `createLexer`, `Lexer#highlight`,
`Lexer#classify`, `Lexer#destroy`, `isSupported`, `CLASS_NAMES`, and the
`lex-large/react` bindings (`<Lex code>` / `useLex(code)`).

## Requirements

WebGPU. Chrome 113+, Edge 113+, Safari 18+, Firefox 141+. `isSupported()` lets
you fall back; there is no CPU path.

## License

MIT — see [LICENSE](LICENSE).
