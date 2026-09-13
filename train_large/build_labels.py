"""Shard hybrid Shiki/Highlight.js labelling across disjoint language groups."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from languages import TARGET_LANGUAGES, teacher

HERE = Path(__file__).resolve().parent


def run_shard(index: int, entries: list[dict], out_dir: Path) -> tuple[int, int]:
    manifest = out_dir / f'manifest.{index}.jsonl'
    output = out_dir / f'labels.{index}.tsv'
    with manifest.open('w') as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + '\n')
    process = subprocess.run(
        ['node', '--max-old-space-size=6144', str(HERE / 'label_worker.mjs'),
         str(manifest), str(output)],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    manifest.unlink(missing_ok=True)
    if process.returncode:
        raise RuntimeError(f'shard {index} failed:\n{process.stderr[-3000:]}')
    ok = sum(1 for _ in output.open()) if output.exists() else 0
    tail = process.stderr.strip().splitlines()
    print(f'  shard {index}: {tail[-1] if tail else f"{ok} labelled"}')
    return ok, len(entries) - ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='./corpus/raw')
    parser.add_argument('--out', default='./corpus/labels')
    parser.add_argument('--shards', type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    parser.add_argument('--languages', default='')
    args = parser.parse_args()
    selected = ([item.strip() for item in args.languages.split(',') if item.strip()]
                or list(TARGET_LANGUAGES))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Shard names depend on the selected languages and shard count. Leaving
    # files from an older run here makes build_dataset.py ingest both schemas.
    for stale in out_dir.glob('labels.*.tsv'):
        stale.unlink()
    for stale in out_dir.glob('manifest.*.jsonl'):
        stale.unlink()

    by_language: dict[str, list[dict]] = {}
    for language in selected:
        directory = Path(args.raw) / language
        if not directory.is_dir():
            print(f'  ! no raw corpus for {language}', file=sys.stderr)
            continue
        engine, teacher_language = teacher(language)
        by_language[language] = [
            {
                'sourceLanguage': language,
                'teacher': engine,
                'language': teacher_language,
                'path': str(path.resolve()),
            }
            for path in sorted(directory.iterdir()) if path.is_file()
        ]

    if not by_language:
        raise SystemExit('no files to label')

    # Keep each grammar in one shard. This loads each Shiki grammar once rather
    # than loading nearly all 185 grammars into every worker.
    buckets: list[list[dict]] = [[] for _ in range(min(args.shards, len(by_language)))]
    loads = [0] * len(buckets)
    for _, entries in sorted(by_language.items(), key=lambda item: -len(item[1])):
        target = min(range(len(buckets)), key=loads.__getitem__)
        buckets[target].extend(entries)
        loads[target] += len(entries)

    total = sum(loads)
    print(f'Labelling {total:,} files across {len(buckets)} grammar-partitioned shards...')
    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        results = list(pool.map(lambda pair: run_shard(pair[0], pair[1], out_dir), enumerate(buckets)))
    ok = sum(result[0] for result in results)
    print(f'\nLabelled {ok:,} / {total:,} files ({100 * ok / total:.1f}%)')


if __name__ == '__main__':
    main()
