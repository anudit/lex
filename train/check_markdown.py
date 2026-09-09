"""
Why markdown scores far below every other language.

Runs the shipped model against Shiki on markdown from the unseen-repo corpus and
prints the confusion, so the cause is visible rather than inferred.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

import data_pipeline as dp
import reference
import tokenizer
from labels import CLASS_NAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='markdown')
    ap.add_argument('--raw', default='./corpus/holdout')
    ap.add_argument('--weights', default='./checkpoints')
    ap.add_argument('--files', type=int, default=12)
    args = ap.parse_args()

    files = sorted((Path(args.raw) / args.lang).iterdir())[:args.files]
    manifest = Path('/tmp/md_manifest.jsonl')
    manifest.write_text(''.join(
        json.dumps({'lang': args.lang, 'path': str(p)}) + '\n' for p in files))
    subprocess.run(['node', 'label_worker.mjs', str(manifest), '/tmp/md_labels.tsv',
                    args.lang], check=True, capture_output=True)

    ref = reference.Reference(args.weights)
    conf = np.zeros((9, 9), dtype=np.int64)
    train_dist = Counter()

    for line in open('/tmp/md_labels.tsv'):
        path, _, rle = line.rstrip('\n').partition('\t')
        code = Path(path).read_text(encoding='utf-8')
        char_cls = dp.decode_rle(rle, len(code))
        toks = tokenizer.tokenize(code)
        gold = dp.align(toks, char_cls)
        feats = tokenizer.tokens_to_arrays(toks)
        pred = ref.forward(feats).argmax(-1)
        for g, p in zip(gold, pred):
            if g >= 0:
                conf[g, p] += 1
                train_dist[int(g)] += 1

    total = conf.sum()
    print(f'\n{args.lang}: {total:,} supervised tokens from {len(files)} unseen files')
    print(f'accuracy {100 * np.trace(conf) / max(1, total):.2f}%\n')
    print(f'{"truth":9s} {"n":>8s} {"share":>7s} {"recall":>7s}   predicted as')
    for i in np.argsort(-conf.sum(1)):
        n = conf[i].sum()
        if n == 0:
            continue
        row = conf[i].copy()
        rec = row[i] / n
        row[i] = 0
        top = np.argsort(-row)[:3]
        s = ', '.join(f'{CLASS_NAMES[j]} {100 * row[j] / n:.0f}%' for j in top if row[j])
        print(f'{CLASS_NAMES[i]:9s} {n:8,d} {100 * n / total:6.1f}% {100 * rec:6.1f}%   {s}')

    # Compare against what the training corpus contained for this language.
    ds, meta = dp.build(cache='./corpus/dataset')
    print(f'\ntraining tokens for {args.lang}: '
          f'{meta["tokens_per_lang"].get(args.lang, 0):,}')


if __name__ == '__main__':
    main()
