// Ground-truth labeller. Shiki is the normalization reference for this project:
// it tokenizes with the same TextMate grammars VS Code uses, and its scopes are
// projected onto gpu-lexer's 9 classes by scope_map.mjs.
//
// Emits, per file, a run-length encoding of the per-character class so the whole
// corpus can be labelled by one node process -- spawning per file costs ~50 ms
// and would dominate the run at 20M+ tokens.
//
// Protocol
//   argv[2] = manifest path, one JSON object per line {lang, path}
//   argv[3] = output path
//   out     = one line per file:  <path>\t<rle>
//             rle is <classDigit><runLength> pairs covering the file exactly,
//             where class 9 means whitespace/uncovered (masked downstream).
//
// A file whose RLE does not reconstruct to its exact character length is dropped
// rather than emitted, so downstream alignment can never silently drift.

import fs from 'node:fs';
import readline from 'node:readline';
import { createHighlighter } from 'shiki';
import { classifyScopes } from './scope_map.mjs';

const MASK = 9;

const [, , manifestPath, outPath, langsCsv] = process.argv;
const langs = langsCsv.split(',').filter(Boolean);

const highlighter = await createHighlighter({
  themes: ['github-dark'],
  langs,
});
// Shiki resolves plain-text ids without registering a grammar, so they never
// appear in getLoadedLanguages(); they still tokenize, with empty scope stacks
// that the scope map correctly reads as `plain`.
const PLAIN_IDS = new Set(['txt', 'text', 'plaintext', 'plain', 'ansi']);
const loaded = new Set([...highlighter.getLoadedLanguages(), ...PLAIN_IDS]);

function encode(code, lang) {
  let result;
  try {
    result = highlighter.codeToTokens(code, {
      lang,
      theme: 'github-dark',
      includeExplanation: true,
    });
  } catch {
    return null;
  }

  const runs = [];
  let pos = 0;
  const push = (cls, len) => {
    if (len <= 0) return;
    const last = runs[runs.length - 1];
    if (last && last[0] === cls) last[1] += len;
    else runs.push([cls, len]);
  };

  for (const line of result.tokens) {
    for (const token of line) {
      // Shiki merges adjacent tokens that share a theme colour; the explanation
      // array preserves the underlying grammar tokens and their scope stacks.
      const parts = token.explanation?.length
        ? token.explanation
        : [{ content: token.content, scopes: [] }];
      for (const part of parts) {
        const text = part.content;
        if (!text.length) continue;
        if (!code.startsWith(text, pos)) return null;
        const scopes = part.scopes.map((s) => s.scopeName);
        const cls = /^\s*$/.test(text) ? MASK : classifyScopes(scopes);
        push(cls, text.length);
        pos += text.length;
      }
    }
    // Shiki strips the line terminator; it belongs to no token.
    if (code[pos] === '\r') { push(MASK, 1); pos += 1; }
    if (code[pos] === '\n') { push(MASK, 1); pos += 1; }
  }

  if (pos !== code.length) {
    const tail = code.slice(pos);
    if (!/^\s*$/.test(tail)) return null;
    push(MASK, tail.length);
  }
  return runs.map(([c, n]) => `${c}:${n}`).join(',');
}

const out = fs.createWriteStream(outPath, { encoding: 'utf8' });
const rl = readline.createInterface({
  input: fs.createReadStream(manifestPath, { encoding: 'utf8' }),
  crlfDelay: Infinity,
});

let ok = 0;
let dropped = 0;
const dropsByLang = new Map();
const t0 = Date.now();

for await (const line of rl) {
  if (!line.trim()) continue;
  const { lang, path } = JSON.parse(line);
  if (!loaded.has(lang)) { dropped += 1; continue; }
  let code;
  try {
    code = fs.readFileSync(path, 'utf8');
  } catch {
    dropped += 1;
    continue;
  }
  const rle = encode(code, lang);
  if (rle === null) {
    dropped += 1;
    dropsByLang.set(lang, (dropsByLang.get(lang) || 0) + 1);
    continue;
  }
  out.write(`${path}\t${rle}\n`);
  ok += 1;
  if ((ok + dropped) % 500 === 0) {
    const rate = (ok + dropped) / ((Date.now() - t0) / 1000);
    process.stderr.write(`  ${ok} ok / ${dropped} dropped  (${rate.toFixed(0)} files/s)\r`);
  }
}
await new Promise((r) => out.end(r));
process.stderr.write(`\nlabelled ${ok} files, dropped ${dropped}\n`);
if (dropsByLang.size) {
  process.stderr.write(`drops by language: ${JSON.stringify(Object.fromEntries(dropsByLang))}\n`);
}
