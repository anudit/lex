/** Build the pinned 185-language manifest from installed Highlight.js + Shiki. */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import hljs from 'highlight.js';
import { bundledLanguages } from 'shiki';
import YAML from 'yaml';

const here = path.dirname(fileURLToPath(import.meta.url));
const hlPackage = JSON.parse(fs.readFileSync(
  path.join(here, 'node_modules/highlight.js/package.json'), 'utf8'));
const shikiPackage = JSON.parse(fs.readFileSync(
  path.join(here, 'node_modules/shiki/package.json'), 'utf8'));
const languageDir = path.join(here, 'node_modules/highlight.js/lib/languages');

// Excluded: each of these is fetched and labelled with the SAME extension as
// a plain sibling grammar already in the manifest, so its "corpus" is just
// duplicate or overlapping content under a second label the fetcher cannot
// actually distinguish from the sibling's. Training measured this directly:
// erlang-repl/julia-repl came out byte-identical in token count to
// erlang/julia (same files, same content, two labels); python-repl,
// clojure-repl and node-repl show the same pattern with '.py'/'.clj'/'.js'.
// php-template is fetched via plain '.php' (a strict subset of php's own
// extension list) so every file collected is an ordinary standalone PHP
// file, not the embedded HTML+PHP markup the grammar exists to demarcate.
// xl is dropped separately: highlight.js's own xl grammar mislabels 73-97%
// of every real .xl source file as "comment" regardless of content (checked
// against six files from c3d/xl and c3d/tao3D) -- the teacher itself is
// broken for this grammar, not just under-resourced.
// excel is dropped separately: its only extension is .xlsx, a binary
// zip-based (OOXML) format, not text -- every fetched file necessarily fails
// the pipeline's UTF-8 decode, so the corpus is structurally 0 tokens no
// matter how much fetching is done. Confirmed after a full rebuild: 0/300k.
const EXCLUDED_IDS = new Set([
  'erlang-repl', 'julia-repl', 'python-repl', 'clojure-repl', 'node-repl',
  'php-template', 'xl', 'excel',
]);

const ids = fs.readdirSync(languageDir)
  .filter((name) => name.endsWith('.js'))
  // The npm package ships both foo.js and foo.js.js module variants.
  .map((name) => name.replace(/(?:\.js)+$/, ''));
const canonicalIds = [...new Set(ids)].filter((id) => !EXCLUDED_IDS.has(id)).sort();

const aliases = {
  bash: 'shellscript',
  delphi: 'pascal',
  fortran: 'fortran-free-form',
  makefile: 'make',
  mathematica: 'wolfram',
  objectivec: 'objc',
  protobuf: 'proto',
  shell: 'shellsession',
  vim: 'viml',
  x86asm: 'asm',
  armasm: 'asm',
};

const linguistAliases = {
  '1c': '1C Enterprise',
  armasm: 'Assembly',
  bash: 'Shell',
  csharp: 'C#',
  cpp: 'C++',
  delphi: 'Pascal',
  dockerfile: 'Dockerfile',
  dos: 'Batchfile',
  fortran: 'Fortran Free Form',
  javascript: 'JavaScript',
  makefile: 'Makefile',
  mathematica: 'Mathematica',
  objectivec: 'Objective-C',
  plaintext: 'Text',
  protobuf: 'Protocol Buffer',
  q: 'q',
  shell: 'ShellSession',
  typescript: 'TypeScript',
  vbnet: 'Visual Basic .NET',
  vim: 'Vim Script',
  x86asm: 'Assembly',
};

const fallbackExtensions = {
  abnf: ['.abnf'], accesslog: ['.log'], apache: ['.conf'], bnf: ['.bnf'],
  arcade: ['.arcade'], arduino: ['.ino'], avrasm: ['.asm'], axapta: ['.xpp'],
  brainfuck: ['.bf'], cal: ['.cl'], cos: ['.cls'],
  crmsh: ['.crm'], csp: ['.csp'], dts: ['.dts'], dust: ['.dust'], fix: ['.fix'],
  django: ['.html'], dns: ['.zone'], dsconfig: ['.conf'],
  freedesktop: ['.desktop'],
  gauss: ['.gss'], gcode: ['.gcode'], gml: ['.gml'], hsp: ['.hsp'],
  http: ['.http'], ini: ['.ini'], irpf90: ['.irp.f'], isbl: ['.q'],
  'jboss-cli': ['.cli'], ldif: ['.ldif'], leaf: ['.leaf'],
  livecodeserver: ['.lcs'], maxima: ['.mac'], mel: ['.mel'], mipsasm: ['.s'],
  mizar: ['.miz'], mojolicious: ['.ep'], n1ql: ['.n1ql'], nestedtext: ['.nt'],
  plaintext: ['.txt'],
  parser3: ['.p'], pf: ['.conf'], pgsql: ['.sql'], profile: ['.profile'],
  properties: ['.properties'], reasonml: ['.re'],
  rib: ['.rib'], roboconf: ['.graph'], routeros: ['.rsc'], rsl: ['.rsl'],
  ruleslanguage: ['.rule'], shell: ['.sh'], step21: ['.p21'],
  subunit: ['.subunit'], taggerscript: ['.tag'], tap: ['.tap'], tp: ['.tp'],
  'vbscript-html': ['.asp'], xl: ['.xl'],
};

const normalize = (value) => String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '');

const commitResponse = await fetch('https://api.github.com/repos/github-linguist/linguist/commits/main', {
  headers: { 'User-Agent': 'lex-train-large/1.0' },
});
if (!commitResponse.ok) throw new Error(`GitHub Linguist commit: ${commitResponse.status}`);
const linguistCommit = await commitResponse.json();
const linguistSha = linguistCommit.sha;
const yamlResponse = await fetch(
  `https://raw.githubusercontent.com/github-linguist/linguist/${linguistSha}/lib/linguist/languages.yml`);
if (!yamlResponse.ok) throw new Error(`GitHub Linguist languages.yml: ${yamlResponse.status}`);
const linguist = YAML.parse(await yamlResponse.text());
const linguistIndex = new Map();
for (const [name, meta] of Object.entries(linguist)) {
  for (const candidate of [name, ...(meta.aliases || [])]) {
    linguistIndex.set(normalize(candidate), [name, meta]);
  }
}

const shikiIds = new Set(Object.keys(bundledLanguages));
const languages = canonicalIds.map((id) => {
  const grammar = hljs.getLanguage(id) || {};
  const grammarAliases = [...new Set(grammar.aliases || [])].sort();
  const candidates = [
    linguistAliases[id], grammar.name, id, ...grammarAliases,
  ].filter(Boolean);
  const match = candidates.map((candidate) => linguistIndex.get(normalize(candidate))).find(Boolean);
  const [linguistName, linguistMeta] = match || [null, {}];
  const extensionSet = new Set([
    ...(linguistMeta.extensions || []),
    ...(fallbackExtensions[id] || []),
  ]);
  const shiki = aliases[id] || (shikiIds.has(id) ? id : null);
  return {
    id,
    name: grammar.name || id,
    aliases: grammarAliases,
    teacher: shikiIds.has(shiki)
      ? { engine: 'shiki', language: shiki }
      : { engine: 'highlight.js', language: id },
    linguist: linguistName,
    extensions: [...extensionSet].sort(),
    filenames: [...new Set(linguistMeta.filenames || [])].sort(),
  };
});

const manifest = {
  format: 'lex-large-language-manifest-v1',
  generated: new Date().toISOString(),
  highlightJs: {
    version: hlPackage.version,
    canonicalGrammarCount: languages.length,
    releaseCommit: 'f7f7d3803bd898e37c017ffb881317f0cde04a70',
  },
  shiki: { version: shikiPackage.version },
  githubLinguist: { commit: linguistSha },
  languages,
};

fs.writeFileSync(path.join(here, 'language_manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
const counts = Object.groupBy(languages, (language) => language.teacher.engine);
console.log(`wrote ${languages.length} grammars`);
for (const [engine, values] of Object.entries(counts)) {
  console.log(`  ${engine}: ${values.length}`);
}
