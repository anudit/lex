// Hybrid ground-truth worker. Shiki is preferred where it has a grammar;
// Highlight.js 11.12.0 supplies the remaining canonical Highlight.js set.
//
// The actual tokenization runs in a worker thread (label_encode_thread.mjs)
// so a single pathological file -- catastrophic regex backtracking in a
// Shiki/oniguruma or Highlight.js grammar, seen in practice on generated
// boilerplate (e.g. Nethereum's ContractDefinition C# files) -- can be
// force-terminated after PER_FILE_TIMEOUT_MS instead of hanging the whole
// shard (and with it the rest of the pipeline) forever.

import fs from 'node:fs';
import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const WORKER_ENTRY = HERE + 'label_encode_thread.mjs';
const PER_FILE_TIMEOUT_MS = Number(process.env.LABEL_TIMEOUT_MS || 10_000);

const [, , manifestPath, outPath] = process.argv;
const entries = fs.readFileSync(manifestPath, 'utf8')
  .split('\n').filter(Boolean).map(JSON.parse);
const shikiLanguages = [...new Set(entries
  .filter((entry) => entry.teacher === 'shiki')
  .map((entry) => entry.language))];

let worker;
let pending = null; // { id, resolve, timer }
let nextId = 0;
let timeouts = 0;

function onMessage(msg) {
  if (pending && msg.id === pending.id) {
    clearTimeout(pending.timer);
    const resolve = pending.resolve;
    pending = null;
    resolve(msg.rle);
  }
}

function onCrash() {
  if (pending) {
    clearTimeout(pending.timer);
    const resolve = pending.resolve;
    pending = null;
    resolve(null);
  }
  worker = spawnWorker();
}

function spawnWorker() {
  const w = new Worker(WORKER_ENTRY, { workerData: { shikiLanguages } });
  w.on('message', onMessage);
  w.on('error', onCrash);
  w.on('exit', (code) => {
    if (code !== 0 && pending) onCrash();
  });
  return w;
}

worker = spawnWorker();

async function encodeEntry(entry, code) {
  const id = nextId++;
  return new Promise((resolve) => {
    pending = {
      id,
      resolve,
      timer: setTimeout(async () => {
        pending = null;
        timeouts += 1;
        process.stderr.write(
          `  !! timeout after ${PER_FILE_TIMEOUT_MS}ms, dropping: ${entry.path}\n`);
        await worker.terminate();
        worker = spawnWorker();
        resolve(null);
      }, PER_FILE_TIMEOUT_MS),
    };
    worker.postMessage({
      id, code, teacher: entry.teacher,
      language: entry.language, sourceLanguage: entry.sourceLanguage,
    });
  });
}

const out = fs.createWriteStream(outPath, { encoding: 'utf8' });
let ok = 0;
let dropped = 0;
const drops = new Map();
for (const entry of entries) {
  let code;
  try {
    code = fs.readFileSync(entry.path, 'utf8');
  } catch {
    dropped += 1;
    continue;
  }
  const rle = await encodeEntry(entry, code);
  if (rle === null) {
    dropped += 1;
    drops.set(entry.sourceLanguage, (drops.get(entry.sourceLanguage) || 0) + 1);
    continue;
  }
  out.write(`${entry.path}\t${rle}\n`);
  ok += 1;
}
await worker.terminate();
await new Promise((resolve) => out.end(resolve));
process.stderr.write(`labelled ${ok}, dropped ${dropped}`);
if (timeouts) process.stderr.write(`; ${timeouts} dropped by timeout`);
if (drops.size) process.stderr.write(`; drops ${JSON.stringify(Object.fromEntries(drops))}`);
process.stderr.write('\n');
