// Pins the WGSL pipeline against the numpy reference in train/reference.py.
// Both read the same weights.bin, so any disagreement is a shader bug. The cases
// cover the shapes that have actually broken it before: a single token, symbols
// only, a sequence longer than one tile, escapes and template literals, and
// non-ASCII text.
import cases from './refcases.json';
import { createLexer } from 'lex';

const out = document.getElementById('out');
const log = [];

try {
  if (!navigator.gpu) throw new Error('WebGPU unavailable');
  const lexer = await createLexer();

  let allOk = true;
  log.push('— end-to-end class agreement (single job) —');
  for (const [name, c] of Object.entries(cases)) {
    const { classes } = await lexer.classify(c.code);
    const got = Array.from(classes);
    const want = c.classes;
    let diff = -1;
    for (let i = 0; i < want.length; i++) if (got[i] !== want[i]) { diff = i; break; }
    const same = diff === -1 && got.length === want.length;
    allOk &&= same;
    log.push(`  ${same ? 'OK  ' : 'FAIL'} ${name.padEnd(6)} n=${want.length}` +
      (same ? '' : ` first diff @${diff} got=${got[diff]} want=${want[diff]}`));
  }
  // Batched dispatch writes per-job slices of the shared hidden-state and
  // scratch buffers, indexed off workgroup_id.y. A region-offset mistake there
  // would be invisible in the single-job tests above and silently wrong on any
  // real page, which highlights many blocks at once.
  log.push('\n— batched vs sequential (same inputs, one submit) —');
  {
    const codes = Object.values(cases).map((c) => c.code);
    const sequential = [];
    for (const code of codes) sequential.push(Array.from((await lexer.classify(code)).classes));
    const batched = (await Promise.all(codes.map((c) => lexer.classify(c))))
      .map((r) => Array.from(r.classes));
    let bad = 0;
    for (let i = 0; i < codes.length; i++) {
      const same = sequential[i].length === batched[i].length
        && sequential[i].every((v, j) => v === batched[i][j]);
      if (!same) {
        bad += 1;
        const at = sequential[i].findIndex((v, j) => v !== batched[i][j]);
        log.push(`  FAIL job ${i} (n=${sequential[i].length}) first diff @${at}` +
          ` sequential=${sequential[i][at]} batched=${batched[i][at]}`);
      }
    }
    log.push(bad ? `  ${bad}/${codes.length} jobs differ when batched`
                 : `  OK   all ${codes.length} jobs identical batched and sequential`);
    allOk &&= bad === 0;
  }

  log.push(allOk ? '\nSHADER MATCHES REFERENCE' : '\nSHADER MISMATCH');
} catch (err) {
  log.push('ERROR: ' + (err && err.stack ? err.stack : String(err)));
}
out.textContent = log.join('\n');
