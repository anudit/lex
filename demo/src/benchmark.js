// Shared measurement code for the correctness benchmark.
//
// Lives apart from the page so it can be run headlessly to regenerate
// results.json (`bun run capture`) without the page depending on it. The page
// renders committed numbers; only this file computes them.

import { createHighlighter, bundledLanguages } from 'shiki';
import { parse as sugarParse } from 'sugar-high/core';
import { languages as sugarLanguages } from 'sugar-high/lang';
import hljs from 'highlight.js';
import Prism from 'prismjs';
import 'prismjs/components/prism-abnf.js';
import 'prismjs/components/prism-clike.js';
import 'prismjs/components/prism-javascript.js';
import 'prismjs/components/prism-actionscript.js';
import 'prismjs/components/prism-ada.js';
import 'prismjs/components/prism-applescript.js';
import 'prismjs/components/prism-c.js';
import 'prismjs/components/prism-cpp.js';
import 'prismjs/components/prism-arduino.js';
import 'prismjs/components/prism-armasm.js';
import 'prismjs/components/prism-asciidoc.js';
import 'prismjs/components/prism-autohotkey.js';
import 'prismjs/components/prism-autoit.js';
import 'prismjs/components/prism-awk.js';
import 'prismjs/components/prism-bash.js';
import 'prismjs/components/prism-basic.js';
import 'prismjs/components/prism-bnf.js';
import 'prismjs/components/prism-brainfuck.js';
import 'prismjs/components/prism-clojure.js';
import 'prismjs/components/prism-cmake.js';
import 'prismjs/components/prism-coffeescript.js';
import 'prismjs/components/prism-coq.js';
import 'prismjs/components/prism-ruby.js';
import 'prismjs/components/prism-crystal.js';
import 'prismjs/components/prism-csharp.js';
import 'prismjs/components/prism-csp.js';
import 'prismjs/components/prism-css.js';
import 'prismjs/components/prism-d.js';
import 'prismjs/components/prism-dart.js';
import 'prismjs/components/prism-diff.js';
import 'prismjs/components/prism-markup.js';
import 'prismjs/components/prism-markup-templating.js';
import 'prismjs/components/prism-django.js';
import 'prismjs/components/prism-docker.js';
import 'prismjs/components/prism-ebnf.js';
import 'prismjs/components/prism-elixir.js';
import 'prismjs/components/prism-elm.js';
import 'prismjs/components/prism-erb.js';
import 'prismjs/components/prism-erlang.js';
import 'prismjs/components/prism-fortran.js';
import 'prismjs/components/prism-fsharp.js';
import 'prismjs/components/prism-gcode.js';
import 'prismjs/components/prism-gherkin.js';
import 'prismjs/components/prism-glsl.js';
import 'prismjs/components/prism-gml.js';
import 'prismjs/components/prism-go.js';
import 'prismjs/components/prism-gradle.js';
import 'prismjs/components/prism-graphql.js';
import 'prismjs/components/prism-groovy.js';
import 'prismjs/components/prism-haml.js';
import 'prismjs/components/prism-handlebars.js';
import 'prismjs/components/prism-haskell.js';
import 'prismjs/components/prism-haxe.js';
import 'prismjs/components/prism-http.js';
import 'prismjs/components/prism-inform7.js';
import 'prismjs/components/prism-ini.js';
import 'prismjs/components/prism-java.js';
import 'prismjs/components/prism-json.js';
import 'prismjs/components/prism-julia.js';
import 'prismjs/components/prism-kotlin.js';
import 'prismjs/components/prism-latex.js';
import 'prismjs/components/prism-less.js';
import 'prismjs/components/prism-lisp.js';
import 'prismjs/components/prism-livescript.js';
import 'prismjs/components/prism-llvm.js';
import 'prismjs/components/prism-lua.js';
import 'prismjs/components/prism-makefile.js';
import 'prismjs/components/prism-markdown.js';
import 'prismjs/components/prism-matlab.js';
import 'prismjs/components/prism-mel.js';
import 'prismjs/components/prism-mizar.js';
import 'prismjs/components/prism-monkey.js';
import 'prismjs/components/prism-moonscript.js';
import 'prismjs/components/prism-n1ql.js';
import 'prismjs/components/prism-nasm.js';
import 'prismjs/components/prism-nginx.js';
import 'prismjs/components/prism-nim.js';
import 'prismjs/components/prism-nix.js';
import 'prismjs/components/prism-nsis.js';
import 'prismjs/components/prism-objectivec.js';
import 'prismjs/components/prism-ocaml.js';
import 'prismjs/components/prism-perl.js';
import 'prismjs/components/prism-php.js';
import 'prismjs/components/prism-powershell.js';
import 'prismjs/components/prism-processing.js';
import 'prismjs/components/prism-prolog.js';
import 'prismjs/components/prism-properties.js';
import 'prismjs/components/prism-protobuf.js';
import 'prismjs/components/prism-puppet.js';
import 'prismjs/components/prism-purebasic.js';
import 'prismjs/components/prism-python.js';
import 'prismjs/components/prism-q.js';
import 'prismjs/components/prism-qml.js';
import 'prismjs/components/prism-r.js';
import 'prismjs/components/prism-roboconf.js';
import 'prismjs/components/prism-rust.js';
import 'prismjs/components/prism-sas.js';
import 'prismjs/components/prism-scala.js';
import 'prismjs/components/prism-scheme.js';
import 'prismjs/components/prism-scss.js';
import 'prismjs/components/prism-smali.js';
import 'prismjs/components/prism-smalltalk.js';
import 'prismjs/components/prism-sml.js';
import 'prismjs/components/prism-sqf.js';
import 'prismjs/components/prism-sql.js';
import 'prismjs/components/prism-stan.js';
import 'prismjs/components/prism-mata.js';
import 'prismjs/components/prism-stata.js';
import 'prismjs/components/prism-stylus.js';
import 'prismjs/components/prism-swift.js';
import 'prismjs/components/prism-yaml.js';
import 'prismjs/components/prism-tap.js';
import 'prismjs/components/prism-tcl.js';
import 'prismjs/components/prism-twig.js';
import 'prismjs/components/prism-typescript.js';
import 'prismjs/components/prism-vala.js';
import 'prismjs/components/prism-vbnet.js';
import 'prismjs/components/prism-verilog.js';
import 'prismjs/components/prism-vhdl.js';
import 'prismjs/components/prism-vim.js';
import 'prismjs/components/prism-wasm.js';
import 'prismjs/components/prism-wren.js';
import 'prismjs/components/prism-xquery.js';
import 'prismjs/components/prism-zig.js';
import 'prismjs/components/prism-toml.js';
import 'prismjs/components/prism-hcl.js';

import {
  makeShikiAdapter, makeSugarHighAdapter, makePrismAdapter, makeHljsAdapter, makeSpanAdapter,
  agreement,
} from './adapters.js';
import { CORPUS, EXCLUDED, WEIGHTS } from './corpus.js';
import { UNSEEN } from './corpus_unseen.js';
import { CORPUS_LARGE, WEIGHTS_LARGE } from './corpus_large.js';

export const PRISM_LANG = {
  abnf: 'abnf', actionscript: 'actionscript', ada: 'ada', applescript: 'applescript', arduino: 'arduino',
  armasm: 'armasm', asciidoc: 'asciidoc', autohotkey: 'autohotkey', autoit: 'autoit', awk: 'awk',
  bash: 'bash', basic: 'basic', bnf: 'bnf', brainfuck: 'brainfuck', c: 'c',
  clojure: 'clojure', cmake: 'cmake', coffeescript: 'coffeescript', coq: 'coq', cpp: 'cpp',
  crystal: 'crystal', csharp: 'csharp', csp: 'csp', css: 'css', d: 'd',
  dart: 'dart', diff: 'diff', django: 'django', dockerfile: 'docker', ebnf: 'ebnf',
  elixir: 'elixir', elm: 'elm', erb: 'erb', erlang: 'erlang', fortran: 'fortran',
  fsharp: 'fsharp', gcode: 'gcode', gherkin: 'gherkin', glsl: 'glsl', gml: 'gml',
  go: 'go', gradle: 'gradle', graphql: 'graphql', groovy: 'groovy', haml: 'haml',
  handlebars: 'handlebars', haskell: 'haskell', haxe: 'haxe', hcl: 'hcl', html: 'markup',
  http: 'http', inform7: 'inform7', ini: 'ini', java: 'java', javascript: 'javascript',
  json: 'json', julia: 'julia', kotlin: 'kotlin', latex: 'latex', less: 'less',
  lisp: 'lisp', livescript: 'livescript', llvm: 'llvm', lua: 'lua', makefile: 'makefile',
  markdown: 'markdown', matlab: 'matlab', mel: 'mel', mizar: 'mizar', monkey: 'monkey',
  moonscript: 'moonscript', n1ql: 'n1ql', nginx: 'nginx', nim: 'nim', nix: 'nix',
  nsis: 'nsis', objectivec: 'objectivec', ocaml: 'ocaml', perl: 'perl', php: 'php',
  plaintext: 'plain', powershell: 'powershell', processing: 'processing', prolog: 'prolog', properties: 'properties',
  protobuf: 'protobuf', puppet: 'puppet', purebasic: 'purebasic', python: 'python', q: 'q',
  qml: 'qml', r: 'r', roboconf: 'roboconf', ruby: 'ruby', rust: 'rust',
  sas: 'sas', scala: 'scala', scheme: 'scheme', scss: 'scss', shell: 'bash',
  smali: 'smali', smalltalk: 'smalltalk', sml: 'sml', sqf: 'sqf', sql: 'sql',
  stan: 'stan', stata: 'stata', stylus: 'stylus', swift: 'swift', tap: 'tap',
  tcl: 'tcl', toml: 'toml', twig: 'twig', typescript: 'typescript', vala: 'vala',
  vbnet: 'vbnet', verilog: 'verilog', vhdl: 'vhdl', vim: 'vim', wasm: 'wasm',
  wren: 'wren', x86asm: 'nasm', xml: 'markup', xquery: 'xquery', yaml: 'yaml',
  zig: 'zig',
};

export async function boot(onStatus = () => {}) {
  onStatus('loading engines…');
  // CORPUS_LARGE draws from train_large's 193-grammar target set, whose
  // shiki_id() falls back to the raw Highlight.js id for languages Shiki
  // doesn't actually bundle (mizar, rib, step21, ...) -- passing an unknown
  // id to createHighlighter throws, so only request ids Shiki really has.
  const langs = [...new Set([...CORPUS, ...UNSEEN, ...CORPUS_LARGE].map((c) => c.shikiLang))]
    .filter((l) => !['txt', 'text', 'plaintext'].includes(l))
    .filter((l) => l in bundledLanguages);
  const highlighter = await createHighlighter({ themes: ['github-dark'], langs });

  const adapters = [
    makeShikiAdapter(highlighter),
    makeSugarHighAdapter(sugarParse, sugarLanguages),
    makePrismAdapter(Prism, PRISM_LANG),
    makeHljsAdapter(hljs),
  ];
  if (navigator.gpu) {
    const { createLexer } = await import('lex');
    const lexer = await createLexer();
    adapters.push(makeSpanAdapter('lex-lite (ours)', (code) => lexer.highlight(code)));
    const { createLexer: createLexerLarge } = await import('lex-large');
    const lexerLarge = await createLexerLarge();
    adapters.push(makeSpanAdapter('lex-large (ours)', (code) => lexerLarge.highlight(code)));
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
export async function measure(adapters, corpus, label, onStatus = () => {}, opts = {}) {
  const weights = opts.weights ?? WEIGHTS;
  const excluded = new Set(opts.excluded ?? EXCLUDED);
  corpus = corpus.filter((item) => !excluded.has(item.lang));
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
  const wsum = present.reduce((acc, l) => acc + (weights[l] ?? 0), 0);
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
      .reduce((acc, l) => acc + (weights[l] ?? 0) * (perLang[l] ?? 0), 0) / wsum;
    rows.push({ name: a.name, weighted, micro: total ? hit / total : 0, perLang });
  }
  rows.sort((a, b) => b.weighted - a.weighted);
  return { label, nFiles: corpus.length, present, rows };
}

export async function measureAll(onStatus = () => {}) {
  const adapters = await boot(onStatus);
  // Each tab compares its own model against the shared field (gpu-lexer, Prism,
  // Sugar High, Shiki) -- lex-lite's corpora exclude lex-large and vice versa, so
  // each table is the same shape (5 rows) it would be with only one model to show.
  const liteAdapters = adapters.filter(
    (a) => a.name !== 'lex-large (ours)' && a.name !== 'highlight.js');
  const largeAdapters = adapters.filter((a) => a.name !== 'lex-lite (ours)');
  return {
    generated: new Date().toISOString().slice(0, 10),
    corpora: [
      { ...await measure(liteAdapters, UNSEEN, 'unseen repos', onStatus), group: 'lite' },
      { ...await measure(liteAdapters, CORPUS, 'held-out files', onStatus), group: 'lite' },
      { ...await measure(largeAdapters, CORPUS_LARGE, 'train_large corpus', onStatus,
          { weights: WEIGHTS_LARGE, excluded: [] }), group: 'large' },
    ],
  };
}
