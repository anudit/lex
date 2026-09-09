// Shared measurement code for the correctness benchmark.
//
// Lives apart from the page so it can be run headlessly to regenerate
// results.json (`npm run capture`) without the page depending on it. The page
// renders committed numbers; only this file computes them.

import { createHighlighter } from 'shiki';
import { parse as sugarParse } from 'sugar-high/core';
import { languages as sugarLanguages } from 'sugar-high/lang';
import Prism from 'prismjs';
import 'prismjs/components/prism-clike.js';
import 'prismjs/components/prism-javascript.js';
import 'prismjs/components/prism-typescript.js';
import 'prismjs/components/prism-python.js';
import 'prismjs/components/prism-rust.js';
import 'prismjs/components/prism-go.js';
import 'prismjs/components/prism-c.js';
import 'prismjs/components/prism-cpp.js';
import 'prismjs/components/prism-csharp.js';
import 'prismjs/components/prism-java.js';
import 'prismjs/components/prism-kotlin.js';
import 'prismjs/components/prism-swift.js';
import 'prismjs/components/prism-ruby.js';
import 'prismjs/components/prism-php.js';
import 'prismjs/components/prism-lua.js';
import 'prismjs/components/prism-zig.js';
import 'prismjs/components/prism-sql.js';
import 'prismjs/components/prism-bash.js';
import 'prismjs/components/prism-powershell.js';
import 'prismjs/components/prism-markup.js';
import 'prismjs/components/prism-css.js';
import 'prismjs/components/prism-json.js';
import 'prismjs/components/prism-yaml.js';
import 'prismjs/components/prism-toml.js';
import 'prismjs/components/prism-markdown.js';
import 'prismjs/components/prism-docker.js';
import 'prismjs/components/prism-graphql.js';
import 'prismjs/components/prism-hcl.js';
import 'prismjs/components/prism-diff.js';
import 'prismjs/components/prism-dart.js';
import 'prismjs/components/prism-scala.js';
import 'prismjs/components/prism-perl.js';
import 'prismjs/components/prism-r.js';

import {
  makeShikiAdapter, makeSugarHighAdapter, makePrismAdapter, makeSpanAdapter,
  agreement,
} from './adapters.js';
import { CORPUS, WEIGHTS } from './corpus.js';
import { UNSEEN } from './corpus_unseen.js';

export const PRISM_LANG = {
  javascript: 'javascript', typescript: 'typescript', python: 'python',
  rust: 'rust', go: 'go', c: 'c', cpp: 'cpp', csharp: 'csharp', java: 'java',
  kotlin: 'kotlin', swift: 'swift', ruby: 'ruby', php: 'php', lua: 'lua',
  zig: 'zig', sql: 'sql', shell: 'bash', powershell: 'powershell',
  html: 'markup', css: 'css', json: 'json', yaml: 'yaml', toml: 'toml',
  markdown: 'markdown', dockerfile: 'docker', graphql: 'graphql', hcl: 'hcl',
  diff: 'diff', dart: 'dart', scala: 'scala', perl: 'perl', r: 'r',
  plaintext: 'plain',
};

export async function boot(onStatus = () => {}) {
  onStatus('loading engines…');
  const langs = [...new Set([...CORPUS, ...UNSEEN].map((c) => c.shikiLang))]
    .filter((l) => !['txt', 'text', 'plaintext'].includes(l));
  const highlighter = await createHighlighter({ themes: ['github-dark'], langs });

  const adapters = [
    makeShikiAdapter(highlighter),
    makeSugarHighAdapter(sugarParse, sugarLanguages),
    makePrismAdapter(Prism, PRISM_LANG),
  ];
  if (navigator.gpu) {
    const { createLexer } = await import('lex');
    const lexer = await createLexer();
    adapters.push(makeSpanAdapter('lex (ours)', (code) => lexer.highlight(code)));
    const mod = await import('gpu-lexer');
    const gpuParse = mod.parse ?? mod.highlight;
    adapters.push(makeSpanAdapter('gpu-lexer', (code) => gpuParse(code)));
  }
  return adapters;
}

async function classesOf(adapter, code, lang, shikiLang) {
  if (adapter.classesAsync) return adapter.classesAsync(code);
  if (!adapter.supports(adapter.reference ? shikiLang : lang)) return null;
  return adapter.classes(code, adapter.reference ? shikiLang : lang);
}

/** Score one corpus, returning rows ready to render. */
export async function measure(adapters, corpus, label, onStatus = () => {}) {
  const ref = adapters.find((a) => a.reference);
  const others = adapters.filter((a) => !a.reference);
  const tally = new Map(others.map((a) => [a.name, new Map()]));

  for (let i = 0; i < corpus.length; i++) {
    const { code, lang, shikiLang } = corpus[i];
    if (i % 10 === 0) onStatus(`${label} — ${i}/${corpus.length} files`);
    let refCls;
    try {
      refCls = ref.classes(code, shikiLang);
    } catch { continue; }
    for (const a of others) {
      const t = tally.get(a.name);
      if (!t.has(lang)) t.set(lang, { hit: 0, total: 0 });
      const bucket = t.get(lang);
      let got = null;
      try { got = await classesOf(a, code, lang, shikiLang); } catch { got = null; }
      const { hit, total } = got
        ? agreement(code, refCls, got)
        : { hit: 0, total: [...code].filter((c) => !/\s/.test(c)).length };
      bucket.hit += hit;
      bucket.total += total;
    }
    if (i % 10 === 0) await new Promise((r) => setTimeout(r));
  }

  const present = [...new Set(corpus.map((c) => c.lang))];
  const wsum = present.reduce((acc, l) => acc + (WEIGHTS[l] ?? 0), 0);
  const rows = [{ name: 'Shiki', weighted: 1, micro: 1, reference: true, perLang: {} }];
  for (const a of others) {
    const perLang = {};
    let hit = 0;
    let total = 0;
    for (const [lang, b] of tally.get(a.name)) {
      perLang[lang] = b.total ? b.hit / b.total : 0;
      hit += b.hit;
      total += b.total;
    }
    const weighted = present
      .reduce((acc, l) => acc + (WEIGHTS[l] ?? 0) * (perLang[l] ?? 0), 0) / wsum;
    rows.push({ name: a.name, weighted, micro: total ? hit / total : 0, perLang });
  }
  rows.sort((a, b) => b.weighted - a.weighted);
  return { label, nFiles: corpus.length, present, rows };
}

export async function measureAll(onStatus = () => {}) {
  const adapters = await boot(onStatus);
  return {
    generated: new Date().toISOString().slice(0, 10),
    corpora: [
      await measure(adapters, UNSEEN, 'unseen repos', onStatus),
      await measure(adapters, CORPUS, 'held-out files', onStatus),
    ],
  };
}
