// Regenerates demo/src/results.json. Open /capture.html, wait, copy the JSON.
// Correctness only changes when the model or the corpora change, so the page
// itself renders committed numbers rather than recomputing 1,555 files on load.
import { measureAll } from './benchmark.js';

const s = document.getElementById('s');
const out = document.getElementById('out');
try {
  const data = await measureAll((msg) => { s.textContent = msg; });
  const json = JSON.stringify(data, null, 2);
  out.value = json;
  const saved = await fetch('/__save-results', { method: 'POST', body: json })
    .then((r) => r.ok).catch(() => false);
  s.textContent = `done — ${data.corpora.map((c) => `${c.label}: ${c.nFiles} files`).join(', ')}`
    + (saved ? ' — written to src/results.json' : ' — copy the JSON below into src/results.json');
} catch (err) {
  s.textContent = String(err);
  console.error(err);
}
