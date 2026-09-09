"""
How much of the remaining error is the model's fault, and how much is the
feature set's?

The model only ever sees what the CPU pre-tokenizer hands it: kind, a length
bucket, first and last character, two hashes of the word, eight flags, and the
bigram transitions on either side. It never sees the token's characters. So two
different identifiers that land in the same hash bucket are, to the model, the
same input -- and if they carry different labels, no amount of parameters or data
can separate them.

That gives a computable ceiling: group the corpus by exact feature vector and
count how often the majority label is wrong. That is the Bayes error of this
feature set, and it bounds every model built on it.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

import data_pipeline as dp
from labels import CLASS_NAMES, MASK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='./corpus/dataset')
    ap.add_argument('--split', default='val')
    args = ap.parse_args()

    ds, meta = dp.build(cache=args.dataset)
    d = ds[args.split]
    flat = d.flat
    labels = flat['label']
    keep = labels != MASK
    n = int(keep.sum())

    # Exact feature identity, as the embedding sees it.
    full = np.stack([flat[k][keep] for k in dp.FEATURE_KEYS], axis=1)
    y = labels[keep]

    def ceiling(cols, name):
        sub = np.ascontiguousarray(full[:, cols])
        _, inv = np.unique(sub, axis=0, return_inverse=True)
        inv = inv.ravel()
        # Majority label per feature group, via a group x class count matrix.
        n_groups = int(inv.max()) + 1
        counts = np.zeros((n_groups, len(CLASS_NAMES)), dtype=np.int64)
        np.add.at(counts, (inv, y), 1)
        correct = int(counts.max(axis=1).sum())
        groups = counts
        print(f'  {name:44s} {len(groups):8,d} distinct  ceiling {100 * correct / n:6.2f}%')
        return correct / n

    idx = {k: i for i, k in enumerate(dp.FEATURE_KEYS)}
    print(f'\n{args.split} split: {n:,} supervised tokens\n')
    print('Best achievable accuracy for a model that sees only these features:')
    ceiling(list(range(len(dp.FEATURE_KEYS))), 'all features (current model input)')
    ceiling([idx['kind'], idx['hash1'], idx['hash2']], 'kind + both word hashes')
    ceiling([idx['kind'], idx['first_char'], idx['last_char'], idx['len_bucket']],
            'kind + first/last char + length')
    ceiling([idx['kind']], 'kind alone')

    # What the hash actually costs: how many distinct words share a bucket.
    words = full[:, idx['hash1']]
    per_bucket = Counter(words.tolist())
    print(f'\n  hash1 buckets used: {len(per_bucket)} / 1024')

    # Per-class share of the residual.
    print('\nLabel distribution in this split:')
    c = Counter(y.tolist())
    for k, v in sorted(c.items()):
        print(f'  {CLASS_NAMES[k]:9s} {v:8,d}  {100 * v / n:5.2f}%')


if __name__ == '__main__':
    main()
