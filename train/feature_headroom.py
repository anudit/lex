"""
How much accuracy is locked behind the feature set, and what would unlock it.

The ceiling of a feature set is the accuracy of a perfect classifier over it:
group every token by its exact feature vector and take the majority label. No
model, no amount of data, and no training objective can beat it.

This compares the current features against candidate additions, so the question
"what gets us to 95%" is answered with a number rather than a preference.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import data_pipeline as dp
import tokenizer
from languages import TARGET_LANGUAGES as LANGS
from labels import CLASS_NAMES, MASK


def fnv(s: str, mod: int) -> int:
    h = 2166136261
    for ch in s:
        h = ((h ^ (ord(ch) & 0x7F)) * 16777619) & 0xFFFFFFFF
    return (h ^ (h >> 16)) % mod


def ceiling(keys: list[tuple], y: np.ndarray) -> tuple[float, int]:
    _, inv = np.unique(np.array(keys, dtype=np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    counts = np.zeros((int(inv.max()) + 1, 9), dtype=np.int64)
    np.add.at(counts, (inv, y), 1)
    return counts.max(axis=1).sum() / len(y), counts.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='./corpus/labels')
    ap.add_argument('--max-files', type=int, default=900)
    args = ap.parse_args()

    rows: list[dict] = []
    labs: list[int] = []
    seen = 0
    for shard in sorted(Path(args.labels).glob('labels.*.tsv')):
        for line in shard.open():
            if seen >= args.max_files:
                break
            path, _, rle = line.rstrip('\n').partition('\t')
            if not rle:
                continue
            try:
                code = Path(path).read_text(encoding='utf-8')
            except OSError:
                continue
            seen += 1
            cc = dp.decode_rle(rle, len(code))
            toks = tokenizer.tokenize(code)
            lab = dp.align(toks, cc)
            lang_id = LANGS.index(Path(path).parent.name) \
                if Path(path).parent.name in LANGS else len(LANGS)
            for t, l in zip(toks, lab):
                if l == MASK:
                    continue
                txt = t.text
                rows.append({
                    'kind': t.kind, 'len': t.len_bucket, 'first': t.first_char,
                    'last': t.last_char, 'h1': t.hash1, 'h2': t.hash2,
                    'flags': t.flags, 'tp': t.trans_prev, 'tn': t.trans_next,
                    'sp': t.sym_prev, 'sn': t.sym_next,
                    # Candidate additions: hashed character n-grams. These are what
                    # separate `HashMap` from `hashmap` from a colliding identifier.
                    'pre3': fnv(txt[:3].lower(), 512),
                    'suf3': fnv(txt[-3:].lower(), 512),
                    'pre4': fnv(txt[:4].lower(), 1024),
                    'suf4': fnv(txt[-4:].lower(), 1024),
                    'exact': fnv(txt, 1 << 20),   # stand-in for "sees the word"
                    'lang': lang_id,             # a language hint the model does not get
                })
                labs.append(int(l))
        if seen >= args.max_files:
            break

    y = np.array(labs)
    n = len(y)
    base = ['kind', 'len', 'first', 'last', 'h1', 'h2', 'flags', 'tp', 'tn', 'sp', 'sn']

    def show(name: str, cols: list[str]) -> float:
        keys = [tuple(r[c] for c in cols) for r in rows]
        acc, groups = ceiling(keys, y)
        print(f'  {name:46s} {groups:8,d} groups   ceiling {100 * acc:6.2f}%')
        return acc

    print(f'\n{n:,} tokens from {seen} files\n')
    print('Context-free ceiling (best possible per-token classifier):')
    b = show('current features (hash1 = 1024 buckets)', base)
    # Would a smaller word-hash table cost anything? The embedding is the largest
    # tensor in the model, so halving it is the cheapest way to buy depth.
    e = show('+ exact word identity', base + ['exact'])
    # The model is language-agnostic by design and never sees which language it
    # is reading. Several remaining errors look like language-identity errors --
    # punctuation that is an operator in C and prose in Markdown -- so this
    # measures what a language hint would actually be worth.
    g = show('+ language id (a hint the model does not get)', base + ['lang'])
    h = show('+ language id AND exact word identity', base + ['lang', 'exact'])
    print(f'\n  headroom from perfect word identity: {100 * (e - b):+.2f} pts')
    print(f'  headroom from a language hint:       {100 * (g - b):+.2f} pts')
    print(f'  headroom from both:                  {100 * (h - b):+.2f} pts')

    print('\nWhere the current ceiling bites, by class:')
    keys = [tuple(r[c] for c in base) for r in rows]
    _, inv = np.unique(np.array(keys, dtype=np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    counts = np.zeros((int(inv.max()) + 1, 9), dtype=np.int64)
    np.add.at(counts, (inv, y), 1)
    winner = counts.argmax(axis=1)
    reachable = (winner[inv] == y)
    for c in range(9):
        sel = y == c
        if not sel.any():
            continue
        print(f'  {CLASS_NAMES[c]:9s} {sel.sum():8,d} tokens   '
              f'max reachable {100 * reachable[sel].mean():6.2f}%')


if __name__ == '__main__':
    main()
