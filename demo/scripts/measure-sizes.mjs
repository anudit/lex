import zlib from 'node:zlib';
import esbuild from 'esbuild';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const demoDir = path.resolve(__dirname, '..');

async function buildAndMeasure(stdinCode) {
  const res = await esbuild.build({
    stdin: { contents: stdinCode, resolveDir: demoDir },
    bundle: true,
    minify: true,
    format: 'esm',
    write: false,
    logLevel: 'silent',
  });
  const code = res.outputFiles[0].contents;
  const brotli = zlib.brotliCompressSync(code, {
    params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 11 },
  });
  return { raw: code.length, brotli: brotli.length };
}

function fmt(bytes) {
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + 'MB';
  return (bytes / 1024).toFixed(1) + 'KB';
}

async function run() {
  console.log('Measuring bundles...');

  // 1. Sugar High (major 6 web languages)
  const sh6 = await buildAndMeasure(`
    import { parse, render } from "sugar-high/core";
    import * as js from "sugar-high/lang/javascript";
    import * as ts from "sugar-high/lang/typescript";
    import * as css from "sugar-high/lang/css";
    import * as html from "sugar-high/lang/html";
    import * as json from "sugar-high/lang/json";
    import * as md from "sugar-high/lang/markdown";
    const langs = { js, ts, css, html, json, md };
    export const highlight = (code, lang) => render(parse(code, langs[lang]));
  `);
  console.log('Sugar High major 6:', sh6.brotli, fmt(sh6.brotli));

  // 2. Prism.js (major 6 web languages)
  const p6 = await buildAndMeasure(`
    import Prism from "prismjs";
    import "prismjs/components/prism-typescript";
    import "prismjs/components/prism-css";
    import "prismjs/components/prism-json";
    import "prismjs/components/prism-markdown";
    export default Prism;
  `);
  console.log('Prism.js major 6:', p6.brotli, fmt(p6.brotli));

  // 3. Sugar High (all 29 languages)
  const shAll = await buildAndMeasure(`
    import { highlight } from "sugar-high";
    export default highlight;
  `);
  console.log('Sugar High all 29:', shAll.brotli, fmt(shAll.brotli));

  // 4. Highlight.js (major 6 web languages)
  const hljs6 = await buildAndMeasure(`
    import hljs from "highlight.js/lib/core";
    import javascript from "highlight.js/lib/languages/javascript";
    import typescript from "highlight.js/lib/languages/typescript";
    import css from "highlight.js/lib/languages/css";
    import xml from "highlight.js/lib/languages/xml";
    import json from "highlight.js/lib/languages/json";
    import markdown from "highlight.js/lib/languages/markdown";
    hljs.registerLanguage("javascript", javascript);
    hljs.registerLanguage("typescript", typescript);
    hljs.registerLanguage("css", css);
    hljs.registerLanguage("xml", xml);
    hljs.registerLanguage("json", json);
    hljs.registerLanguage("markdown", markdown);
    export default hljs;
  `);
  console.log('Highlight.js major 6:', hljs6.brotli, fmt(hljs6.brotli));

  // 5. gpu-lexer
  const gpuLexer = await buildAndMeasure(`
    import { parse } from "gpu-lexer";
    export default parse;
  `);
  console.log('gpu-lexer:', gpuLexer.brotli, fmt(gpuLexer.brotli));

  // 6. lex-lite (ours)
  const lexLite = await buildAndMeasure(`
    import { parse } from "lex";
    export default parse;
  `);
  const lexLiteWeights = await buildAndMeasure(`
    import * as w from "../lex/src/weights.js";
    export default w;
  `);
  const lexLiteCodeBytes = lexLite.brotli - lexLiteWeights.brotli;
  console.log('lex-lite:', lexLite.brotli, fmt(lexLite.brotli), `(weights: ${fmt(lexLiteWeights.brotli)}, code: ${fmt(lexLiteCodeBytes)})`);

  // 7. lex-large (ours)
  const lexLarge = await buildAndMeasure(`
    import { parse } from "lex-large";
    export default parse;
  `);
  const lexLargeWeights = await buildAndMeasure(`
    import * as w from "../lex-large/src/weights.js";
    export default w;
  `);
  const lexLargeCodeBytes = lexLarge.brotli - lexLargeWeights.brotli;
  console.log('lex-large:', lexLarge.brotli, fmt(lexLarge.brotli), `(weights: ${fmt(lexLargeWeights.brotli)}, code: ${fmt(lexLargeCodeBytes)})`);

  // 8. Prism.js (all 297 languages)
  const componentsDir = path.join(demoDir, 'node_modules/prismjs/components');
  const prismFiles = fs.readdirSync(componentsDir).filter((f) => f.startsWith('prism-') && f.endsWith('.min.js'));
  const pAllImports = prismFiles.map((f) => `import "prismjs/components/${f.replace('.min.js', '')}";`).join('\n');
  const pAll = await buildAndMeasure(`
    import Prism from "prismjs";
    ${pAllImports}
    export default Prism;
  `);
  console.log('Prism.js all:', pAll.brotli, fmt(pAll.brotli));

  // 9. Starry Night (major 6 web languages) + WASM
  const sn6 = await buildAndMeasure(`
    import { createStarryNight } from "@wooorm/starry-night";
    import sourceJs from "@wooorm/starry-night/source.js";
    import sourceTs from "@wooorm/starry-night/source.ts";
    import sourceCss from "@wooorm/starry-night/source.css";
    import textHtmlBasic from "@wooorm/starry-night/text.html.basic";
    import sourceJson from "@wooorm/starry-night/source.json";
    import textMd from "@wooorm/starry-night/text.md";
    export default { createStarryNight, sourceJs, sourceTs, sourceCss, textHtmlBasic, sourceJson, textMd };
  `);
  const wasmPath = path.join(demoDir, 'node_modules/vscode-oniguruma/release/onig.wasm');
  const wasmBuf = fs.readFileSync(wasmPath);
  const wasmBrotli = zlib.brotliCompressSync(wasmBuf, {
    params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 11 },
  });
  const sn6Total = sn6.brotli + wasmBrotli.length;
  console.log('Starry Night major 6:', sn6Total, fmt(sn6Total));

  // 10. Shiki (major 6 web languages) + WASM
  const shiki6 = await buildAndMeasure(`
    import { createHighlighterCore } from "shiki/core";
    import js from "@shikijs/langs/javascript";
    import ts from "@shikijs/langs/typescript";
    import css from "@shikijs/langs/css";
    import html from "@shikijs/langs/html";
    import json from "@shikijs/langs/json";
    import md from "@shikijs/langs/markdown";
    import getWasm from "shiki/wasm";
    export default { createHighlighterCore, js, ts, css, html, json, md, getWasm };
  `);
  console.log('Shiki major 6:', shiki6.brotli, fmt(shiki6.brotli));

  // 11. Highlight.js (all 193 languages)
  const hljsAll = await buildAndMeasure(`
    import hljs from "highlight.js";
    export default hljs;
  `);
  console.log('Highlight.js all:', hljsAll.brotli, fmt(hljsAll.brotli));

  // 12. Shiki (all 242 grammars)
  const shikiAll = await buildAndMeasure(`
    import { createHighlighter } from "shiki/bundle/full";
    export default createHighlighter;
  `);
  console.log('Shiki all:', shikiAll.brotli, fmt(shikiAll.brotli));

  // 13. Starry Night (all 710 grammars) + WASM
  const snAll = await buildAndMeasure(`
    import { createStarryNight, all } from "@wooorm/starry-night";
    export default { createStarryNight, all };
  `);
  const snAllTotal = snAll.brotli + wasmBrotli.length;
  console.log('Starry Night all:', snAllTotal, fmt(snAllTotal));

  const data = {
    title: 'Loaded library size',
    subtitle: 'runtime + selected language coverage \u00b7 lower is better',
    footnote:
      'Minified and Brotli-compressed browser bundles. gpu-lexer and lex were measured on September 18, 2026; comparison libraries on September 9\u201318, 2026. Major web includes javascript, typescript, css, html, json, and markdown. gpu-lexer, lex-lite, and lex-large use the same bundle for every language. Starry Night totals include its Oniguruma WASM payload.',
    rows: [
      { name: 'Sugar High (major 6 web languages)', bytes: sh6.brotli, display: '5.4KB' },
      { name: 'Prism.js (major 6 web languages)', bytes: p6.brotli, display: '8.6KB' },
      { name: 'Sugar High (all 29 languages)', bytes: shAll.brotli, display: '8.8KB' },
      { name: 'Highlight.js (major 6 web languages)', bytes: hljs6.brotli, display: '14.9KB' },
      { name: 'gpu-lexer', bytes: gpuLexer.brotli, display: '27.6KB', highlight: true },
      {
        name: 'lex-lite (ours, 52 languages)',
        bytes: lexLite.brotli,
        display: fmt(lexLite.brotli),
        ours: true,
        stacked: {
          weights: { bytes: lexLiteWeights.brotli, display: fmt(lexLiteWeights.brotli) },
          code: { bytes: lexLiteCodeBytes, display: fmt(lexLiteCodeBytes) },
        },
      },
      {
        name: 'lex-large (ours, 185 languages)',
        bytes: lexLarge.brotli,
        display: fmt(lexLarge.brotli),
        ours: true,
        stacked: {
          weights: { bytes: lexLargeWeights.brotli, display: fmt(lexLargeWeights.brotli) },
          code: { bytes: lexLargeCodeBytes, display: fmt(lexLargeCodeBytes) },
        },
      },
      { name: 'Prism.js (all 297 languages)', bytes: pAll.brotli, display: '162.1KB' },
      { name: 'Starry Night (major 6 web languages)', bytes: sn6Total, display: '185.3KB' },
      { name: 'Shiki (major 6 web languages)', bytes: shiki6.brotli, display: '213.9KB' },
      { name: 'Highlight.js (all 193 languages)', bytes: hljsAll.brotli, display: '240.4KB' },
      { name: 'Shiki (all 242 grammars)', bytes: shikiAll.brotli, display: '991.5KB' },
      { name: 'Starry Night (all 710 grammars)', bytes: snAllTotal, display: '1.46MB' },
    ],
  };

  // Sort rows by size ascending
  data.rows.sort((a, b) => a.bytes - b.bytes);

  const outPath = path.join(demoDir, 'src/sizes.json');
  fs.writeFileSync(outPath, JSON.stringify(data, null, 2) + '\n');
  console.log(`Wrote ${outPath}`);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
