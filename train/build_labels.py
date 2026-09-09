"""
Runs Shiki over the raw corpus and writes per-character class labels.

Work is sharded across N node processes because a single Shiki process labels
roughly 0.3 MB/s; the corpus is ~120 MB. Each shard gets a disjoint slice of the
manifest and writes its own TSV, so a crashed shard costs one slice rather than
the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from languages import TARGET_LANGUAGES, shiki_id

HERE = Path(__file__).resolve().parent


def run_shard(shard_id: int, manifest: Path, out: Path, langs: list[str]) -> tuple[int, int]:
    proc = subprocess.run(
        ['node', '--max-old-space-size=4096', str(HERE / 'label_worker.mjs'),
         str(manifest), str(out), ','.join(sorted(set(langs)))],
        cwd=HERE, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f'  shard {shard_id} failed:\n{proc.stderr[-2000:]}', file=sys.stderr)
        return 0, 0
    tail = proc.stderr.strip().splitlines()
    print(f'  shard {shard_id}: {tail[-1] if tail else "done"}')
    ok = sum(1 for _ in out.open())
    return ok, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='./corpus/raw')
    ap.add_argument('--out', default='./corpus/labels')
    ap.add_argument('--shards', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument('--languages', default='')
    args = ap.parse_args()

    raw = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    langs = [l.strip() for l in args.languages.split(',') if l.strip()] or list(TARGET_LANGUAGES)

    entries: list[dict[str, str]] = []
    for lang in langs:
        d = raw / lang
        if not d.is_dir():
            print(f'  ! no raw corpus for {lang}', file=sys.stderr)
            continue
        sid = shiki_id(lang)
        for p in sorted(d.iterdir()):
            if p.is_file():
                entries.append({'lang': sid, 'path': str(p)})

    if not entries:
        sys.exit('no files to label')

    # Round-robin so every shard sees a mix of languages and grammar-load cost is
    # spread evenly rather than one shard paying for all of C++.
    shards = min(args.shards, len(entries))
    buckets: list[list[dict[str, str]]] = [[] for _ in range(shards)]
    for i, e in enumerate(entries):
        buckets[i % shards].append(e)

    print(f'Labelling {len(entries):,} files across {shards} Shiki shards...')
    manifests = []
    for i, bucket in enumerate(buckets):
        m = out_dir / f'manifest.{i}.jsonl'
        with m.open('w') as fh:
            for e in bucket:
                fh.write(json.dumps(e) + '\n')
        manifests.append((i, m, out_dir / f'labels.{i}.tsv', [e['lang'] for e in bucket]))

    with ThreadPoolExecutor(max_workers=shards) as pool:
        list(pool.map(lambda a: run_shard(*a), manifests))

    total = sum(1 for _, _, o, _ in manifests if o.exists() for _ in o.open())
    print(f'\nLabelled {total:,} / {len(entries):,} files '
          f'({100 * total / len(entries):.1f}%)')
    for _, m, _, _ in manifests:
        m.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
