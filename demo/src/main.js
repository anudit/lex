// Benchmark: five highlighters, two questions.
//
// Correctness is measured once and committed to results.json, because the answer
// only changes when the model or the corpora change -- recomputing 1,555 files on
// every page load costs ~90 seconds and a GPU device to redraw a static bar
// chart. Regenerate with `npm run capture`. Latency is different: it depends on
// the machine viewing the page, so it stays a live button.
//
//   Correctness -- popularity-weighted agreement with Shiki, over two corpora.
//     Shiki is the normalization reference; every other library's token names are
//     mapped onto the same nine classes and compared per non-whitespace
//     character. Languages a library does not support score zero, which is the
//     honest treatment: an unhighlighted file is a wrong answer, not a missing
//     measurement.
//
//     "held-out files" holds out files but not repositories, so lex has seen the
//     house style; "unseen repos" is drawn from repositories absent from the
//     training corpus, where every engine is equally out of distribution. The
//     second is the number to quote in a head-to-head.
//

import './style.css';
import { WEIGHTS, TOP25 } from './corpus.js';
import RESULTS from './results.json';
import { boot } from './benchmark.js';
import { CLASS_NAMES } from './adapters.js';

const pct = (n) => `${(n * 100).toFixed(2)}%`;

const app = document.getElementById('app');

app.innerHTML = `
  <header>
    <h1>lex vs. the field</h1>
    <p class="sub">
      Five highlighters normalized to the same nine syntax classes and scored
      against <strong>Shiki</strong> per non-whitespace character, weighted by
      GitHub language popularity. A language an engine does not support scores
      zero &mdash; an unhighlighted file is a wrong answer, not a missing
      measurement.
    </p>
  </header>
  <div id="results"></div>
  <div class="controls">
    <button id="run-latency" class="primary" type="button">Run latency</button>
    <span class="hint">measured on your machine &middot; needs WebGPU</span>
  </div>
  <div id="status" class="status">correctness measured ahead of time; press Run for latency.</div>
  <div id="latency"></div>
`;

const status = document.getElementById('status');
const results = document.getElementById('results');
const setStatus = (t, cls = '') => { status.className = `status ${cls}`; status.textContent = t; };

let engines = null;




function renderCorrectness(rows, label, nFiles, present) {
  const best = Math.max(...rows.map((r) => r.weighted));
  const top = rows.reduce((a, b) => (b.weighted > a.weighted ? b : a));
  const langs = Object.keys(WEIGHTS)
    .filter((l) => present.has(l))
    .sort((a, b) => WEIGHTS[b] - WEIGHTS[a]);
  const section = `
    <section data-corpus="${label}">
      <div class="section-head">
        <h2>Correctness — ${label}</h2>
        <span class="meta">popularity-weighted agreement · higher is better</span>
      </div>
      <div class="axis" aria-hidden="true">
        <span></span>
        <div class="axis-ticks"><span>0%</span><span style="left:25%">25%</span><span style="left:50%">50%</span><span style="left:75%">75%</span><span style="left:100%">100%</span></div>
        <span></span>
      </div>
      <div class="bars">
        ${rows.map((r) => `
          <div class="bar-row${r === top ? ' leader' : ''}${r.reference ? ' ref' : ''}">
            <span class="bar-label">${r.name}</span>
            <span class="bar-track"><span class="bar-fill" style="width:${(r.weighted / best) * 100}%"></span></span>
            <span class="bar-value">${pct(r.weighted)}</span>
          </div>`).join('')}
      </div>
      <p class="footnote">Popularity-weighted agreement with Shiki over ${nFiles} files in
        ${present.size} languages, renormalized over the languages present.</p>
      <details>
        <summary>Per-language breakdown</summary>
        <div class="scroll">
          <table>
            <thead><tr><th>language</th><th>weight</th>
              ${rows.filter((r) => !r.reference).map((r) => `<th>${r.name}</th>`).join('')}
            </tr></thead>
            <tbody>
              ${langs.map((l) => `<tr${TOP25.includes(l) ? '' : ' class="tail"'}>
                <td>${l}</td><td class="num">${pct(WEIGHTS[l])}</td>
                ${rows.filter((r) => !r.reference)
                  .map((r) => `<td class="num">${r.perLang[l] === undefined ? '—' : pct(r.perLang[l])}</td>`).join('')}
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </details>
    </section>`;
  // Replace this corpus's own section, keeping the other corpus and latency.
  const existing = results.querySelector(`section[data-corpus="${label}"]`);
  if (existing) {
    existing.outerHTML = section;
  } else {
    results.insertAdjacentHTML('afterbegin', section);
  }
}


// ------------------------------------------------------------------- latency
// Kept live rather than committed: unlike correctness, this depends on the
// machine viewing the page.

const SNIPPETS = [
  'export async function fetchUsers(ids = []) {\n  const MAX = 3.14;\n  return ids.map((id) => `#${id}`); // done\n}',
  'def total(xs):\n    """Sum them."""\n    return sum(xs) + MAX_N',
  'pub fn counts(t: &str) -> HashMap<&str, usize> {\n    let mut m = HashMap::new();  // tally\n    m\n}',
  'SELECT u.id, COUNT(o.id) FROM users u LEFT JOIN orders o ON o.user_id = u.id;',
  ':root { --brand: #0b7285; }\n.card:hover { border-radius: 8px; }',
  'package main\n\nimport "fmt"\n\nfunc main() { fmt.Println(42) }',
  'public final class Cache<K, V> {\n  private static final int MAX = 1024;\n}',
  'apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: 3',
  '# Title\n\nSome **prose** with `code` and a [link](http://x).',
  'const x: Record<string, number> = { a: 1 };',
];
const WARMUP = 5;
const ITERATIONS = 25;
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const fmt = (n) => n.toFixed(2);

async function runLatency() {
  if (!navigator.gpu) throw new Error('WebGPU unavailable');
  setStatus('loading engines\u2026');
  engines ||= await boot((m) => setStatus(m));

  const rows = [];
  for (const a of engines) {
    const call = a.classesAsync
      ? (code) => a.classesAsync(code)
      : (code) => a.classes(code, 'javascript');
    try {
      for (const c of SNIPPETS) for (let i = 0; i < WARMUP; i++) await call(c);
    } catch { continue; }
    const per = [];
    for (const code of SNIPPETS) {
      const t = [];
      for (let i = 0; i < ITERATIONS; i++) {
        const s0 = performance.now();
        await call(code);
        t.push(performance.now() - s0);
      }
      per.push(median(t));
      setStatus(`latency \u2014 ${a.name}`);
    }
    // Every call issued in one turn, which is how a real page highlights a
    // document: the WebGPU engines then share a single submit and readback.
    const t0 = performance.now();
    await Promise.all(SNIPPETS.map(call));
    rows.push({ name: a.name, median: median(per), batch: performance.now() - t0 });
  }
  rows.sort((a, b) => a.median - b.median);

  document.getElementById('latency').innerHTML = `
    <section>
      <div class="section-head"><h2>Latency</h2>
        <span class="meta">median of ${ITERATIONS} iterations &middot; lower is better</span></div>
      <div class="scroll"><table>
        <thead><tr><th>engine</th><th>one call</th>
          <th>${SNIPPETS.length} blocks in one turn</th><th>per block</th></tr></thead>
        <tbody>${rows.map((r) => `<tr>
          <td>${r.name}</td>
          <td class="num">${fmt(r.median)} ms</td>
          <td class="num">${fmt(r.batch)} ms</td>
          <td class="num">${fmt(r.batch / SNIPPETS.length)} ms</td>
        </tr>`).join('')}</tbody>
      </table></div>
      <p class="note">
        About 96% of a call is the GPU readback round-trip, and it barely varies
        with input size &mdash; so calls issued together share one submit and one
        readback.
      </p>
    </section>`;
  setStatus(`latency done \u2014 ${rows.length} engines`);
}

// ---------------------------------------------------------------- page setup

for (const c of RESULTS.corpora) {
  renderCorrectness(c.rows, c.label, c.nFiles, new Set(c.present));
}


document.getElementById('run-latency').addEventListener('click', (e) => {
  e.target.disabled = true;
  runLatency()
    .catch((err) => { setStatus(String(err), 'error'); console.error(err); })
    .finally(() => { e.target.disabled = false; });
});
