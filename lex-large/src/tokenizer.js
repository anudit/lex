// CPU pre-tokenizer for lex-large. A direct port of train_large/tokenizer.py,
// which is itself train/tokenizer.py (lex's own tokenizer, see lex/src/tokenizer.js)
// plus two changes this file must also make -- getting either wrong trains/runs
// the model on features it never saw, which silently degrades every prediction
// rather than erroring:
//
//   * hash1/hash2 are wider (1024/256 buckets instead of 512/128, per
//     train_large/tokenizer.py's docstring), so the finalizing mask differs.
//   * six zero-parameter structural fields are added: paren/brace/bracket
//     nesting depth, position in line, indent width bucket, and quote-string
//     state. lex's two-word packed layout has no spare bits for these (tw0 is
//     used to the last bit, tw1 has 2 bits free), so they ride in a third word.
//
// Kinds: 0 word, 1 space, 2 newline, 3 symbol.

const TRANSITIONS = new Map([
  [(47 << 8) | 47, 1],   // //
  [(47 << 8) | 42, 2],   // /*
  [(42 << 8) | 47, 3],   // */
  [(45 << 8) | 45, 4],   // --
  [(61 << 8) | 62, 5],   // =>
  [(58 << 8) | 58, 6],   // ::
  [(60 << 8) | 47, 7],   // </
  [(123 << 8) | 123, 8], // {{
  [(36 << 8) | 123, 9],  // ${
  [(125 << 8) | 125, 10],// }}
  [(45 << 8) | 62, 11],  // ->
  [(63 << 8) | 63, 12],  // ??
  [(63 << 8) | 46, 13],  // ?.
  [(60 << 8) | 62, 14],  // <>
]);

const isSpace = (c) => c === 9 || c === 11 || c === 12 || c === 32;
const isWord = (c) =>
  c > 127 || c === 95 || (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122);
const charBucket = (c) => (c > 127 ? 95 : c & 127);

function lenBucket(n) {
  if (n <= 1) return 0;
  return Math.min(7, 31 - Math.clz32(n));
}

function symbolHash(a, b) {
  const v = ((Math.imul(a + 1, 131) >>> 0) ^ b) >>> 0;
  return 1 + (v % 31);
}

/**
 * @param {string} text
 * @returns {{count:number, starts:Int32Array, ends:Int32Array, kinds:Uint8Array, packed:Uint32Array}}
 *   `packed` holds three u32 per token in exactly the layout the shader unpacks.
 */
export function tokenize(text) {
  const n = text.length;
  // Upper bound: every character its own token.
  const starts = new Int32Array(n);
  const ends = new Int32Array(n);
  const kinds = new Uint8Array(n);
  const firstC = new Uint8Array(n);
  const lastC = new Uint8Array(n);
  const lenB = new Uint8Array(n);
  const h1 = new Uint16Array(n);
  const h2 = new Uint8Array(n);
  const flags = new Uint8Array(n);
  const transPrev = new Uint8Array(n);
  const transNext = new Uint8Array(n);
  const symPrev = new Uint8Array(n);
  const symNext = new Uint8Array(n);

  let count = 0;
  let pos = 0;
  let lineStart = true;

  while (pos < n) {
    const start = pos;
    const c = text.charCodeAt(pos);
    let kind;
    if (c === 10 || c === 13) kind = 2;
    else if (isSpace(c)) kind = 1;
    else if (isWord(c)) kind = 0;
    else kind = 3;

    let f = lineStart ? 16 : 0;
    const first = charBucket(c);
    let last = first;
    let wordLength = 0;

    if (kind === 0) {
      let a = 2166136261 >>> 0;
      let b = 2654435769 >>> 0;
      while (pos < n && isWord(text.charCodeAt(pos))) {
        // Python hashes Unicode code points. Keep offsets in UTF-16 units for
        // JS slicing, but count/hash a surrogate pair only once.
        const cp = text.codePointAt(pos);
        const cb = charBucket(cp);
        last = cb;
        a = Math.imul(a ^ cb, 16777619) >>> 0;
        b = Math.imul(b ^ cb, 2246822519) >>> 0;
        if (cb >= 97 && cb <= 122) f |= 1;
        else if (cb >= 65 && cb <= 90) f |= 2;
        else if (cb >= 48 && cb <= 57) f |= 4;
        else if (cb === 95) f |= 8;
        pos += cp > 0xffff ? 2 : 1;
        wordLength++;
      }
      // Wider than lex's &511/&127: lex-large's embedding table has 1024/256
      // rows for these two fields (see field_sizes in weights.meta.json).
      h1[count] = (a ^ (a >>> 16)) & 1023;
      h2[count] = (b ^ (b >>> 16)) & 255;
      if ((f & 2) && !(f & 1)) f |= 128;
    } else if (kind === 1) {
      while (pos < n && isSpace(text.charCodeAt(pos))) {
        const cc = text.charCodeAt(pos);
        last = charBucket(cc);
        if (cc === 9) f |= 32;
        pos++;
      }
    } else if (kind === 2) {
      pos++;
      if (c === 13 && pos < n && text.charCodeAt(pos) === 10) {
        last = 10;
        pos++;
      }
    } else {
      if (c === 92) f |= 64;
      pos++;
    }

    starts[count] = start;
    ends[count] = pos;
    kinds[count] = kind;
    firstC[count] = first;
    lastC[count] = last;
    lenB[count] = lenBucket(kind === 0 ? wordLength : pos - start);
    flags[count] = f;
    count++;

    if (kind === 2) lineStart = true;
    else if (kind !== 1) lineStart = false;
  }

  for (let i = 1; i < count; i++) {
    const t = TRANSITIONS.get((lastC[i - 1] << 8) | firstC[i]) || 0;
    transPrev[i] = t;
    transNext[i - 1] = t;
    if (kinds[i - 1] === 3 || kinds[i] === 3) {
      const s = symbolHash(lastC[i - 1], firstC[i]);
      symPrev[i] = s;
      symNext[i - 1] = s;
    }
  }

  // Zero-parameter structural hints: bracket/paren/brace nesting depth,
  // position in the line, indent width, and quote-string state. A second pass
  // over the already-split tokens, mirroring train_large/tokenizer.py's
  // tokens_to_arrays exactly (same clamping, same reset-on-newline, same
  // escape/quote toggling) since the model was trained on that exact sequence.
  const parenD = new Uint8Array(count);
  const braceD = new Uint8Array(count);
  const bracketD = new Uint8Array(count);
  const linePos = new Uint8Array(count);
  const indentB = new Uint8Array(count);
  const quoteS = new Uint8Array(count);
  {
    let paren = 0; let brace = 0; let bracket = 0; let line = 0; let indent = 0;
    let quote = 0; let escaped = false; let atLineStart = true;
    for (let i = 0; i < count; i++) {
      parenD[i] = Math.min(paren, 7);
      braceD[i] = Math.min(brace, 7);
      bracketD[i] = Math.min(bracket, 7);
      linePos[i] = Math.min(line, 7);
      indentB[i] = Math.min(32 - Math.clz32(indent), 7);
      quoteS[i] = quote;

      if (kinds[i] === 2) {
        line = 0; indent = 0;
        atLineStart = true;
        escaped = false;
        continue;
      }
      if (kinds[i] === 1) {
        if (atLineStart) {
          for (let p = starts[i]; p < ends[i]; p++) indent += text.charCodeAt(p) === 9 ? 4 : 1;
        }
        continue;
      }

      const len = ends[i] - starts[i];
      const ch0 = len === 1 ? text.charCodeAt(starts[i]) : -1;
      if (kinds[i] === 3 && len === 1 && (ch0 === 39 || ch0 === 34 || ch0 === 96) && !escaped) {
        const state = ch0 === 39 ? 1 : ch0 === 34 ? 2 : 3;
        quote = quote === state ? 0 : (quote === 0 ? state : quote);
      } else if (quote === 0 && kinds[i] === 3 && len === 1) {
        if (ch0 === 40) paren += 1;         // (
        else if (ch0 === 41) paren = Math.max(0, paren - 1); // )
        else if (ch0 === 123) brace += 1;   // {
        else if (ch0 === 125) brace = Math.max(0, brace - 1); // }
        else if (ch0 === 91) bracket += 1;  // [
        else if (ch0 === 93) bracket = Math.max(0, bracket - 1); // ]
      }
      escaped = kinds[i] === 3 && len === 1 && ch0 === 92 && !escaped; // \
      atLineStart = false;
      line += 1;
    }
  }

  // Pack into the three-word layout the shader reads. Keeping the packing here
  // means the GPU never does field arithmetic it can avoid. Word 0 retains
  // lex's layout; word 1 widens hash2, and word 2 carries the six
  // structural fields (17 of its 32 bits used).
  const packed = new Uint32Array(count * 3);
  for (let i = 0; i < count; i++) {
    packed[i * 3] =
      (kinds[i] & 3) |
      ((lenB[i] & 7) << 2) |
      ((firstC[i] & 127) << 5) |
      ((lastC[i] & 127) << 12) |
      ((flags[i] & 255) << 19) |
      ((symNext[i] & 31) << 27);
    packed[i * 3 + 1] =
      (h1[i] & 1023) |
      ((h2[i] & 255) << 10) |
      ((transPrev[i] & 15) << 18) |
      ((transNext[i] & 15) << 22) |
      ((symPrev[i] & 31) << 26);
    packed[i * 3 + 2] =
      (parenD[i] & 7) |
      ((braceD[i] & 7) << 3) |
      ((bracketD[i] & 7) << 6) |
      ((linePos[i] & 7) << 9) |
      ((indentB[i] & 7) << 12) |
      ((quoteS[i] & 3) << 15);
  }

  return {
    count,
    starts: starts.subarray(0, count),
    ends: ends.subarray(0, count),
    kinds: kinds.subarray(0, count),
    packed,
  };
}
