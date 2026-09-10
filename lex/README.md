# lex

Syntax highlighting from a 35 KB neural model running on WebGPU. Zero
dependencies, no grammars, no per-language bundles.

```js
import { createLexer } from 'lex';

const lex = await createLexer();
const spans = await lex.highlight(code);
// [{ type: 'keyword', start: 0, end: 6 }, ...]
```

React:

```jsx
import { Lex } from 'lex/react';
import 'lex/theme.css';

<Lex code={source} />
```

## How it works

There is no grammar. A CPU pre-tokenizer splits the source into runs of four
kinds -- word, space, newline, symbol -- and derives a handful of sparse features
per token: a length bucket, first and last character, two hashes of the word,
eight character-class flags, and the bigram transitions on either side. A small
model reads those and predicts one of nine classes: `plain`, `comment`, `string`,
`number`, `keyword`, `type`, `function`, `constant`, `operator`.

Because the features are language-agnostic, so is the model. It was trained on
33 languages, and it will make a reasonable attempt at one it has never seen
rather than failing outright.

The model is 154k parameters, quantized to 3 bits for the embedding table and
1 bit for every projection, and ships inline as base85 -- 35 KB, no fetch.

It also reads a pooled *document signature* before any recurrent layer runs and
uses it to modulate every layer. That is what lets one grammar-free model treat
`$` as an operator in a shell script and as ordinary text in Markdown.

## Accuracy

Scored as agreement with [Shiki](https://shiki.style) over held-out files,
per non-whitespace character, weighted by GitHub language popularity.
`bun run dev` in `demo/` reproduces it.

## Performance notes

Two measurements shape the runtime, and both are worth knowing if you are
embedding this:

**Calls in the same microtask turn are batched.** About 96% of a single
`highlight()` is the GPU readback round-trip, and that cost is nearly flat in
input size. Highlighting ten code blocks with ten separate `await`s pays it ten
times; issuing them together pays it once:

```js
// ~10x the round-trip
for (const block of blocks) out.push(await lex.highlight(block));

// one round-trip for all of them
const out = await Promise.all(blocks.map((b) => lex.highlight(b)));
```

**The device is shared by default.** `createLexer()` returns the same instance
across calls, so independent components on a page share one GPU device *and* one
readback. Pass `{ shared: false }` if you need an isolated one.

## API

- `createLexer(options?)` -> `Promise<Lexer>`. `{ shared = true }`.
- `lexer.highlight(code)` -> `Promise<Array<{type, start, end}>>`, whitespace
  absorbed into surrounding spans.
- `lexer.classify(code)` -> `Promise<{tokens, classes}>` for custom span logic.
- `lexer.destroy()` releases GPU resources.
- `isSupported()` -> whether WebGPU is available.
- `CLASS_NAMES` -> the nine class names, indexed by class id.

From `lex/react`: `<Lex code>` and `useLex(code)`, which returns
`{ spans, ready, error }` -- `spans` is `null` until the first result, so you can
render plain text immediately instead of flashing empty.

`lex/theme.css` styles the nine `lex-*` classes and follows
`prefers-color-scheme`; override the custom properties on `.lex` to retheme.

## Requirements

WebGPU. Chrome 113+, Edge 113+, Safari 18+, Firefox 141+. `isSupported()` lets
you fall back; there is no CPU path.

## License

MIT — see [LICENSE](LICENSE).
