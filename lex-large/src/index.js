// lex -- syntax highlighting by a small neural model running on WebGPU.
//
// Public surface:
//   const spans = await parse(code);   // [{type, start, end}, ...]
//
// `parse` matches gpu-lexer's package.json/index.d.ts API exactly (same
// signature, same lazily-created-and-cached-singleton behavior, same
// SyntaxSpan shape -- half-open UTF-16 code-unit offsets, spans in source
// order) so this runtime, shader and weight set can be merged into that
// package as a drop-in swap later. `createLexer`/`Lexer` below are the
// lower-level surface `parse` is built on and stay exported for callers
// (this repo's own demo and validate page included) that want an explicit
// session, per-token classes, or the debug-stage hook -- none of which
// gpu-lexer's API exposes.
//
// The API is language-agnostic on purpose: there is no grammar to select and no
// per-language bundle to load. The model reads the same character-level features
// for every language it was trained on.

import { tokenize } from './tokenizer.js';
import { LexRuntime, CLASS_NAMES } from './runtime.js';
import { unpackWeights } from './weights-codec.js';
import { WEIGHTS_SYM, WEIGHTS_F16_B85, META, PIPELINE } from './weights.js';
import { SHADER } from './shader.js';

export { CLASS_NAMES };

export class Lexer {
  #runtime;

  constructor(runtime) {
    this.#runtime = runtime;
  }

  /**
   * @param {string} code
   * @returns {Promise<Array<{type:string,start:number,end:number}>>}
   *   Adjacent tokens of the same class are merged, and whitespace is absorbed
   *   into the surrounding span so consumers get renderable runs rather than one
   *   element per token.
   */
  async highlight(code) {
    const tok = tokenize(code);
    if (!tok.count) return [];
    const classes = await this.#runtime.classify(tok.packed, tok.count);
    return toSpans(code, tok, classes);
  }

  /** Per-token classes, for callers that want to do their own span assembly. */
  async classify(code) {
    const tok = tokenize(code);
    if (!tok.count) return { tokens: tok, classes: new Uint32Array(0) };
    return { tokens: tok, classes: await this.#runtime.classify(tok.packed, tok.count) };
  }

  /**
   * Hidden state after an intermediate stage (1 = embedding, 2..n = after each
   * recurrent layer). Used by the test page to diff the kernel against the
   * numpy reference stage by stage.
   */
  async debugStage(code, stage) {
    const tok = tokenize(code);
    return this.#runtime.classify(tok.packed, tok.count, stage);
  }

  /** Names of the pipeline stages, so a debug dump can be labelled. */
  get stages() {
    return this.#runtime.stageNames();
  }

  destroy() {
    this.#runtime.destroy();
  }
}

export function toSpans(code, tok, classes) {
  const spans = [];
  let cur = null;
  for (let i = 0; i < tok.count; i++) {
    if (tok.kinds[i] === 1 || tok.kinds[i] === 2) continue; // whitespace
    const name = CLASS_NAMES[classes[i]] || 'plain';
    if (cur && cur.type === name) {
      cur.end = tok.ends[i];
    } else {
      if (cur) spans.push(cur);
      cur = { type: name, start: tok.starts[i], end: tok.ends[i] };
    }
  }
  if (cur) spans.push(cur);
  return spans;
}

let shared = null;

/**
 * @param {{shared?: boolean}} [options] `shared` (default true) reuses one GPU
 *   device across every call in the page, which also lets independent callers
 *   share a single readback round-trip.
 */
export async function createLexer(options = {}) {
  const { shared: useShared = true } = options;
  if (useShared && shared) return shared;
  const { planes, fp } = unpackWeights(WEIGHTS_SYM, WEIGHTS_F16_B85, META);
  const runtime = await LexRuntime.create({
    shader: SHADER, planes, fp, steps: PIPELINE, dim: META.config.dim,
  });
  const lexer = new Lexer(runtime);
  if (useShared) shared = lexer;
  return lexer;
}

export function isSupported() {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

/**
 * gpu-lexer-compatible entry point: identical signature and behavior to that
 * package's only export, so a caller (or a future merge into gpu-lexer
 * itself) can swap one for the other without touching call sites.
 * @param {string} code
 * @returns {Promise<Array<{type:string,start:number,end:number}>>}
 */
export async function parse(code) {
  const lexer = await createLexer();
  return lexer.highlight(code);
}
