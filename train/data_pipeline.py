"""
Dataset assembly for the neural lexer.

Ground truth comes from Shiki, which is the normalization reference for the
benchmark this model is scored on. Nothing in this file invents a label: the
Shiki pass (build_labels.py) produces per-character classes, and the only work
here is aligning them onto the lexer's own token grid and dropping anything
ambiguous rather than guessing.

Three properties the previous pipeline lacked, and the reasons they matter:

  * Splits are by *file*, assigned from a hash of the path. Splitting randomly
    over chunks leaks near-identical blocks between train and val and makes the
    validation number meaningless.
  * Near-duplicate chunks are capped. Real corpora contain generated files,
    vendored copies and licence headers by the thousand; left alone they dominate
    the gradient.
  * Every sample carries its language id, so evaluation can reproduce the
    popularity-weighted metric rather than a corpus-shaped average.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import tokenizer
from labels import MASK, NUM_CLASSES, CLASS_NAMES
from languages import TARGET_LANGUAGES, token_budgets, weights

FEATURE_KEYS = (
    'kind', 'len_bucket', 'first_char', 'last_char', 'hash1', 'hash2',
    'flags', 'trans_prev', 'trans_next', 'sym_prev', 'sym_next',
)

_RLE = re.compile(r'(\d):(\d+)')


def decode_rle(rle: str, length: int) -> np.ndarray:
    """Expand "4:3,9:1,6:5" into a per-character class array (9 = masked)."""
    out = np.full(length, 9, dtype=np.uint8)
    pos = 0
    for cls, count in _RLE.findall(rle):
        n = int(count)
        end = min(pos + n, length)
        if end > pos:
            out[pos:end] = int(cls)
        pos += n
    return out


def align(tokens, char_cls: np.ndarray) -> np.ndarray:
    """Label each lexer token by the class of its first character.

    Whitespace and newline tokens are masked. So is any token whose first
    character Shiki left uncovered -- those are boundary disagreements between the
    two tokenizers, and a guess there is exactly the kind of systematic label
    error that caps accuracy.
    """
    labels = np.full(len(tokens), MASK, dtype=np.int64)
    n = len(char_cls)
    for i, tok in enumerate(tokens):
        if tok.kind in (1, 2) or tok.start >= n:
            continue
        c = int(char_cls[tok.start])
        if c != 9:
            labels[i] = c
    return labels


def split_of(text: str, val_pct: int = 5, test_pct: int = 5) -> str:
    """Deterministic per-content split. Hashing the file *content* -- not the
    path -- guarantees a byte-identical file always lands in the same split no
    matter how many repo paths it was fetched under. Hashing the path basename
    let ~0.6% of files (vendored copies, generated boilerplate, license
    headers fetched under different repo paths) leak byte-identical content
    across train/val/test, silently inflating val/test accuracy."""
    h = int(hashlib.sha1(text.encode('utf-8', errors='ignore')).hexdigest()[:8], 16) % 100
    if h < test_pct:
        return 'test'
    if h < test_pct + val_pct:
        return 'val'
    return 'train'


class LexerDataset(Dataset):
    """Padded fixed-length windows with a validity mask and a language id.

    Windows are stored as one flat array per feature plus an offset index, and
    sliced on access. Materializing 45k per-window dicts costs minutes to load and
    holds the whole corpus as Python objects; this holds it as 12 arrays.
    """

    def __init__(self, flat: dict[str, np.ndarray], offsets: np.ndarray,
                 langs: np.ndarray, seq_len: int):
        self.flat = flat
        self.offsets = offsets
        self.langs = langs
        self.seq_len = seq_len
        self.teacher_logits: np.ndarray | None = None

    @classmethod
    def from_samples(cls, samples: list[dict], seq_len: int) -> 'LexerDataset':
        if not samples:
            empty = {k: np.zeros(0, dtype=np.int64) for k in FEATURE_KEYS}
            empty['label'] = np.zeros(0, dtype=np.int64)
            return cls(empty, np.zeros(1, dtype=np.int64),
                       np.zeros(0, dtype=np.int64), seq_len)
        offsets = np.zeros(len(samples) + 1, dtype=np.int64)
        for i, s in enumerate(samples):
            offsets[i + 1] = offsets[i] + len(s['label'])
        flat = {k: np.concatenate([s[k] for s in samples]).astype(np.int64)
                for k in FEATURE_KEYS}
        flat['label'] = np.concatenate([s['label'] for s in samples]).astype(np.int64)
        langs = np.array([s['lang'] for s in samples], dtype=np.int64)
        return cls(flat, offsets, langs, seq_len)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, b = int(self.offsets[idx]), int(self.offsets[idx + 1])
        n = b - a
        pad = self.seq_len - n
        item = {}
        for k in FEATURE_KEYS:
            v = torch.from_numpy(self.flat[k][a:b])
            item[k] = torch.cat([v, v.new_zeros(pad)]) if pad > 0 else v
        lab = torch.from_numpy(self.flat['label'][a:b])
        item['label'] = torch.cat([lab, lab.new_full((pad,), MASK)]) if pad > 0 else lab
        # Explicit validity mask: padding must not enter the pooled global context.
        valid = torch.zeros(self.seq_len, dtype=torch.bool)
        valid[:n] = True
        item['valid'] = valid
        item['lang'] = torch.tensor(int(self.langs[idx]), dtype=torch.long)
        item['index'] = torch.tensor(idx, dtype=torch.long)
        if self.teacher_logits is not None:
            # Copy one small row out of the read-only mmap so DataLoader can pin it.
            item['teacher_logits'] = torch.from_numpy(
                np.array(self.teacher_logits[idx], copy=True))
        return item


def _chunk(arrays: dict[str, np.ndarray], labels: np.ndarray, lang_id: int,
           seq_len: int, min_len: int) -> list[dict]:
    T = len(labels)
    out = []
    for start in range(0, T, seq_len):
        end = min(start + seq_len, T)
        if end - start < min_len:
            continue
        piece = {k: arrays[k][start:end] for k in FEATURE_KEYS}
        piece['label'] = labels[start:end]
        piece['lang'] = lang_id
        if int((piece['label'] >= 0).sum()) < min_len // 2:
            continue  # almost all whitespace; nothing to learn
        out.append(piece)
    return out


def build(
    label_dir: str = './corpus/labels',
    seq_len: int = 512,
    min_len: int = 32,
    total_tokens: int | None = None,
    max_dup: int = 3,
    cache: str | None = './corpus/dataset',
    verbose: bool = True,
) -> tuple[dict[str, LexerDataset], dict]:
    """Assemble train/val/test datasets from the Shiki-labelled corpus.

    total_tokens=None (the default) means "whatever is already cached at
    `cache`, or 24M for a fresh build" -- a caller that doesn't care about
    corpus size should never be able to silently resize and overwrite a cache
    another process (or a full training run) is relying on.
    """
    cache_path = Path(cache) if cache else None
    cached_meta = None
    if cache_path and (cache_path / 'meta.json').exists():
        cached_meta = json.loads((cache_path / 'meta.json').read_text())

    if total_tokens is None:
        total_tokens = cached_meta.get('total_tokens_requested', 24_000_000) if cached_meta else 24_000_000

    lang_index = {l: i for i, l in enumerate(TARGET_LANGUAGES)}
    budgets = token_budgets(total_tokens)

    if cache_path and cached_meta is not None:
        label_mtime = max((f.stat().st_mtime for f in Path(label_dir).glob('labels.*.tsv')), default=0)
        cache_mtime = (cache_path / 'meta.json').stat().st_mtime
        if cached_meta.get('total_tokens_requested') == total_tokens and cache_mtime >= label_mtime:
            return _load_cache(cache_path, seq_len)
        if verbose:
            print(f'>> cache at {cache_path} is stale (requested tokens or labels changed) -- rebuilding')

    label_files = sorted(Path(label_dir).glob('labels.*.tsv'))
    if not label_files:
        raise FileNotFoundError(
            f'no Shiki label shards in {label_dir}; run build_labels.py first')

    per_split: dict[str, list[dict]] = {'train': [], 'val': [], 'test': []}
    used_tokens: Counter[str] = Counter()
    dup_counts: Counter[bytes] = Counter()
    stats = {
        'files': 0, 'files_skipped_budget': 0, 'chunks_dropped_dup': 0,
        'tokens_per_lang': Counter(), 'labels': Counter(),
    }

    for shard in label_files:
        with shard.open() as fh:
            for line in fh:
                path, _, rle = line.rstrip('\n').partition('\t')
                if not rle:
                    continue
                lang = Path(path).parent.name
                lang_id = lang_index.get(lang)
                if lang_id is None:
                    continue
                if used_tokens[lang] >= budgets.get(lang, 0):
                    stats['files_skipped_budget'] += 1
                    continue
                try:
                    code = Path(path).read_text(encoding='utf-8')
                except OSError:
                    continue
                char_cls = decode_rle(rle, len(code))
                toks = tokenizer.tokenize(code)
                if len(toks) < min_len:
                    continue
                labels = align(toks, char_cls)
                if int((labels >= 0).sum()) < min_len // 2:
                    continue

                arrays = tokenizer.tokens_to_arrays(toks)
                split = split_of(code)
                chunks = _chunk(arrays, labels, lang_id, seq_len, min_len)

                kept = []
                for ch in chunks:
                    # Structural fingerprint: token kinds plus labels. Identifier
                    # names differ across generated files that are otherwise the
                    # same shape, so hashing the text would not catch them.
                    key = hashlib.blake2b(
                        ch['kind'].astype(np.uint8).tobytes()
                        + ch['label'].astype(np.int8).tobytes(),
                        digest_size=16).digest()
                    dup_counts[key] += 1
                    if dup_counts[key] > max_dup:
                        stats['chunks_dropped_dup'] += 1
                        continue
                    kept.append(ch)

                if not kept:
                    continue
                per_split[split].extend(kept)
                n_tok = sum(len(c['label']) for c in kept)
                used_tokens[lang] += n_tok
                stats['tokens_per_lang'][lang] += n_tok
                stats['files'] += 1
                for c in kept:
                    lab = c['label']
                    stats['labels'].update(lab[lab >= 0].tolist())

    _guarantee_split_coverage(per_split)

    datasets = {k: LexerDataset.from_samples(v, seq_len)
                for k, v in per_split.items()}
    meta = _summarize(stats, used_tokens, seq_len, verbose)
    meta['total_tokens_requested'] = total_tokens
    if cache_path:
        _save_cache(cache_path, per_split, meta)
    return datasets, meta


def _guarantee_split_coverage(per_split: dict[str, list[dict]]) -> None:
    """A low-file-count language can land zero chunks in val/test purely from
    the 5%/5% hash-of-basename split's luck of the draw -- it then scores a
    silent 0% forever. Move a few of its train chunks over so every language
    that has any data at all shows up in every split."""
    train_by_lang: dict[int, list[dict]] = defaultdict(list)
    for c in per_split['train']:
        train_by_lang[int(c['lang'])].append(c)

    for split_name in ('val', 'test'):
        present = {int(c['lang']) for c in per_split[split_name]}
        moved_ids: set[int] = set()
        for lang_id, pool in train_by_lang.items():
            if lang_id in present or not pool:
                continue
            n_move = max(1, len(pool) // 20)
            moved = pool[:n_move]
            moved_ids.update(id(c) for c in moved)
            per_split[split_name].extend(moved)
            train_by_lang[lang_id] = pool[n_move:]
        if moved_ids:
            per_split['train'] = [c for c in per_split['train'] if id(c) not in moved_ids]


def _summarize(stats, used_tokens, seq_len, verbose) -> dict:
    w = weights()
    total = sum(used_tokens.values())
    covered = [l for l in TARGET_LANGUAGES if used_tokens[l] > 0]
    meta = {
        'seq_len': seq_len,
        'total_tokens': total,
        'files': stats['files'],
        'languages_covered': len(covered),
        'languages_target': len(TARGET_LANGUAGES),
        'weight_covered': sum(w[l] for l in covered),
        'tokens_per_lang': dict(stats['tokens_per_lang']),
        'label_counts': {CLASS_NAMES[k]: v for k, v in sorted(stats['labels'].items())},
        'chunks_dropped_dup': stats['chunks_dropped_dup'],
    }
    if verbose:
        print(f"\nCorpus: {total:,} labelled tokens from {stats['files']:,} files")
        print(f"Coverage: {len(covered)}/{len(TARGET_LANGUAGES)} languages "
              f"= {100 * meta['weight_covered']:.1f}% of eval weight")
        print(f"Near-duplicate chunks dropped: {stats['chunks_dropped_dup']:,}")
        print('\nLabel distribution:')
        lt = sum(stats['labels'].values())
        for k, v in sorted(stats['labels'].items()):
            print(f'  {CLASS_NAMES[k]:9s} {v:10,d}  {100 * v / max(1, lt):5.2f}%')
    return meta


def _save_cache(path: Path, per_split: dict[str, list[dict]], meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for split, samples in per_split.items():
        if not samples:
            continue
        # Ragged windows are stored flat with an offset index; padding them to
        # seq_len on disk would roughly double the cache for no benefit.
        ds = LexerDataset.from_samples(samples, 0)
        blob = dict(ds.flat)
        blob['offsets'] = ds.offsets
        blob['lang'] = ds.langs
        np.savez(path / f'{split}.npz', **blob)
    (path / 'meta.json').write_text(json.dumps(meta, indent=2))


def _load_cache(path: Path, seq_len: int) -> tuple[dict[str, LexerDataset], dict]:
    meta = json.loads((path / 'meta.json').read_text())
    datasets = {}
    for split in ('train', 'val', 'test'):
        f = path / f'{split}.npz'
        if not f.exists():
            datasets[split] = LexerDataset.from_samples([], seq_len)
            continue
        z = np.load(f)
        flat = {k: z[k] for k in (*FEATURE_KEYS, 'label')}
        datasets[split] = LexerDataset(flat, z['offsets'], z['lang'], seq_len)
    return datasets, meta


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='./corpus/labels')
    ap.add_argument('--seq-len', type=int, default=512)
    ap.add_argument('--total-tokens', type=int, default=24_000_000)
    ap.add_argument('--cache', default='./corpus/dataset')
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()
    ds, meta = build(args.labels, seq_len=args.seq_len, total_tokens=args.total_tokens,
                     cache=None if args.no_cache else args.cache)
    for k, v in ds.items():
        print(f'{k:6s} {len(v):7,d} windows')
