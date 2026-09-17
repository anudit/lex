// CPU pre-tokenizer. Scans source into runs of four kinds (0 word, 1 space,
// 2 newline, 3 symbol), keeps the word and symbol tokens, and derives the sparse
// feature fields the model embeds -- whitespace survives as per-token fields.
// This is a direct port of train/tokenizer.py's tokenize_v2; the two must agree
// exactly, because a feature computed differently at inference is a feature the
// model was never trained on.

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

// Words 0 and 1 of a token's packed features.
function packWords(t, src, dst, packed, wpt) {
  packed[dst * wpt] =
    (t.kinds[src] & 3) |
    ((t.lenB[src] & 7) << 2) |
    ((t.firstC[src] & 127) << 5) |
    ((t.lastC[src] & 127) << 12) |
    ((t.flags[src] & 255) << 19) |
    ((t.symNext[src] & 31) << 27);
  packed[dst * wpt + 1] =
    (t.h1[src] & 1023) |
    ((t.h2[src] & 127) << 10) |
    ((t.transPrev[src] & 15) << 17) |
    ((t.transNext[src] & 15) << 21) |
    ((t.symPrev[src] & 31) << 25);
}

// Whitespace and newline runs leave the sequence; what they carried moves into
// a third word per token:
//   bits 0-1 gap_prev, 2-3 gap_next (0 none, 1 space, 2 newline, 3 blank line)
//   bits 4-6 indent bucket, 7-11 line_first, 12-13 brace depth, 14-15 paren depth
const LINE_FIRST = new Uint8Array(128);
{
  const syms = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~';
  for (let i = 0; i < syms.length; i++) LINE_FIRST[syms.charCodeAt(i)] = i + 1;
}

function indentBucket(w) {
  if (w <= 2) return w;
  if (w <= 4) return 3;
  if (w <= 16) return 4 + ((w - 5) >> 2);
  return 7;
}

/**
 * @param {string} text
 * @returns {{count:number, starts:Int32Array, ends:Int32Array, kinds:Uint8Array,
 *            packed:Uint32Array}}
 *   `packed` holds three u32 per token in exactly the layout the shader unpacks.
 */
export function tokenize(text) {
  const t = scan(text);
  const { count, kinds, firstC, lastC, flags, h1, transPrev, transNext, symPrev, symNext } = t;
  const keep = new Int32Array(count);
  const extra = new Uint32Array(count);
  let m = 0;
  let spaces = false;
  let newlines = 1;
  let indentWidth = 0;
  let indentTab = false;
  let lineFirst = -1;
  let braces = 0;
  let parens = 0;
  let prev = -1;
  let prevGap = 0;
  for (let i = 0; i < count; i++) {
    const kind = kinds[i];
    if (kind === 2) {
      newlines++;
      indentWidth = 0;
      indentTab = false;
      lineFirst = -1;
      continue;
    }
    if (kind === 1) {
      spaces = true;
      if (lineFirst < 0) {
        for (let p = t.starts[i]; p < t.ends[i]; p++) {
          const tab = text.charCodeAt(p) === 9;
          indentWidth += tab ? 4 : 1;
          indentTab = indentTab || tab;
        }
      }
      continue;
    }
    if (lineFirst < 0) lineFirst = kind === 0 ? 0 : LINE_FIRST[firstC[i]];
    h1[i] &= 255;
    const gap = newlines >= 2 ? 3 : newlines === 1 ? 2 : spaces ? 1 : 0;
    if (indentTab) flags[i] |= 32;
    const c = kind === 3 ? firstC[i] : -1;
    let brace;
    if (c === 123) { brace = Math.min(3, braces); braces++; }
    else if (c === 125) { braces = Math.max(0, braces - 1); brace = Math.min(3, braces); }
    else brace = Math.min(3, braces);
    let paren;
    if (c === 40 || c === 91) { paren = Math.min(3, parens); parens++; }
    else if (c === 41 || c === 93) { parens = Math.max(0, parens - 1); paren = Math.min(3, parens); }
    else paren = Math.min(3, parens);
    transPrev[i] = transNext[i] = symPrev[i] = symNext[i] = 0;
    if (prev >= 0) {
      extra[m - 1] = (extra[m - 1] & ~(3 << 2)) | (gap << 2);
      if (gap === 0) {
        const tr = TRANSITIONS.get((lastC[prev] << 8) | firstC[i]) || 0;
        transPrev[i] = tr;
        transNext[prev] = tr;
      }
      if (kinds[prev] === 3 || kind === 3) {
        const s = symbolHash(lastC[prev], firstC[i]);
        symPrev[i] = s;
        symNext[prev] = s;
      }
    }
    // gap_next defaults to "newline" for the last token, as in Python.
    extra[m] = gap | (2 << 2) | (indentBucket(indentWidth) << 4) | (lineFirst << 7)
      | (brace << 12) | (paren << 14);
    keep[m++] = i;
    prev = i;
    spaces = false;
    newlines = 0;
  }
  const packed = new Uint32Array(m * 3);
  const starts = new Int32Array(m);
  const ends = new Int32Array(m);
  const outKinds = new Uint8Array(m);
  for (let j = 0; j < m; j++) {
    const i = keep[j];
    packWords(t, i, j, packed, 3);
    packed[j * 3 + 2] = extra[j];
    starts[j] = t.starts[i];
    ends[j] = t.ends[i];
    outKinds[j] = kinds[i];
  }
  return { count: m, starts, ends, kinds: outKinds, packed };
}

function scan(text) {
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

    if (kind === 0) {
      let a = 2166136261 >>> 0;
      let b = 2654435769 >>> 0;
      while (pos < n && isWord(text.charCodeAt(pos))) {
        const cb = charBucket(text.charCodeAt(pos));
        last = cb;
        a = Math.imul(a ^ cb, 16777619) >>> 0;
        b = Math.imul(b ^ cb, 2246822519) >>> 0;
        if (cb >= 97 && cb <= 122) f |= 1;
        else if (cb >= 65 && cb <= 90) f |= 2;
        else if (cb >= 48 && cb <= 57) f |= 4;
        else if (cb === 95) f |= 8;
        pos++;
      }
      h1[count] = (a ^ (a >>> 16)) & 511;
      h2[count] = (b ^ (b >>> 16)) & 127;
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
    lenB[count] = lenBucket(pos - start);
    flags[count] = f;
    count++;

    if (kind === 2) lineStart = true;
    else if (kind !== 1) lineStart = false;
  }

  return {
    count, starts, ends, kinds, firstC, lastC, lenB, h1, h2, flags,
    transPrev, transNext, symPrev, symNext,
  };
}
