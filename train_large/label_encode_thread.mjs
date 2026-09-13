// Worker-thread body for label_worker.mjs. Runs in its own thread so the
// orchestrator can forcibly terminate() it if a single pathological file
// (catastrophic regex backtracking in a Shiki/oniguruma or Highlight.js
// grammar on generated/repetitive code) blocks for too long -- a plain
// subprocess-per-shard has no way to skip just one file and keep going.

import { parentPort, workerData } from 'node:worker_threads';
import hljs from 'highlight.js';
import { createHighlighter } from 'shiki';
import { classifyScopes } from '../train/scope_map.mjs';

const MASK = 9;
const { shikiLanguages } = workerData;
const highlighter = shikiLanguages.length
  ? await createHighlighter({ themes: ['github-dark'], langs: shikiLanguages })
  : null;

const priority = [
  ['comment', 1], ['quote', 1],
  ['string', 2], ['regexp', 2], ['template-string', 2],
  ['number', 3],
  ['keyword', 4], ['meta-keyword', 4], ['selector-tag', 4],
  ['type', 5], ['class', 5], ['built_in', 5],
  ['title.function', 6], ['function', 6], ['title', 6],
  ['literal', 7], ['symbol', 7], ['variable.constant', 7],
  ['operator', 8], ['punctuation', 8],
];

function highlightClass(stack) {
  const joined = stack.join('.');
  for (const [needle, value] of priority) {
    if (joined.includes(needle)) return value;
  }
  return 0;
}

function encodeRuns(classes) {
  const runs = [];
  for (const value of classes) {
    const last = runs[runs.length - 1];
    if (last && last[0] === value) last[1] += 1;
    else runs.push([value, 1]);
  }
  return runs.map(([value, count]) => `${value}:${count}`).join(',');
}

function encodeShiki(code, language) {
  try {
    const result = highlighter.codeToTokens(code, {
      lang: language,
      theme: 'github-dark',
      includeExplanation: true,
    });
    const classes = [];
    let pos = 0;
    for (const line of result.tokens) {
      for (const token of line) {
        const parts = token.explanation?.length
          ? token.explanation
          : [{ content: token.content, scopes: [] }];
        for (const part of parts) {
          if (!code.startsWith(part.content, pos)) return null;
          const cls = /^\s*$/.test(part.content)
            ? MASK
            : classifyScopes(part.scopes.map((scope) => scope.scopeName));
          classes.push(...Array([...part.content].length).fill(cls));
          pos += part.content.length;
        }
      }
      while (code[pos] === '\r' || code[pos] === '\n') {
        classes.push(MASK);
        pos += 1;
      }
    }
    while (pos < code.length && /\s/.test(code[pos])) {
      classes.push(MASK);
      pos += 1;
    }
    return pos === code.length ? encodeRuns(classes) : null;
  } catch {
    return null;
  }
}

function encodeHighlight(code, language) {
  let result;
  try {
    result = hljs.highlight(code, { language, ignoreIllegals: true });
  } catch {
    return null;
  }
  const classes = [];
  const visit = (node, stack = []) => {
    if (typeof node === 'string') {
      const cls = highlightClass(stack);
      for (const char of node) classes.push(/\s/.test(char) ? MASK : cls);
      return;
    }
    // Highlight.js 11 renamed emitter nodes' `kind` field to `scope`.
    // Keep the fallback for older emitters, but prefer the current field.
    const scope = node.scope || node.kind;
    const next = scope ? [...stack, scope] : stack;
    for (const child of node.children || []) visit(child, next);
  };
  visit(result._emitter.rootNode);
  return classes.length === [...code].length ? encodeRuns(classes) : null;
}

parentPort.on('message', ({ id, code, teacher, language, sourceLanguage }) => {
  let rle = teacher === 'shiki'
    ? encodeShiki(code, language)
    : encodeHighlight(code, language);
  if (rle === null && teacher === 'shiki') {
    rle = encodeHighlight(code, sourceLanguage);
  }
  parentPort.postMessage({ id, rle });
});
