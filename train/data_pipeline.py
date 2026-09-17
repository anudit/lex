"""
Dataset assembly for the neural lexer.

Ground truth comes from Shiki, which is the normalization reference for the
benchmark this model is scored on. Nothing in this file invents a label: the
Shiki pass (build_labels.py) produces per-character classes, and the only work
here is aligning them onto the lexer's own token grid and dropping anything
ambiguous rather than guessing.

Properties that matter for the result:

  * Splits are by *file*, assigned from a hash of the file content, so
    near-identical chunks and byte-identical copies cannot straddle train and
    validation.
  * Files whose extension is a different dialect from their directory's grammar
    (languages.grammar_matches_file) are skipped rather than mislabelled.
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

try:
    from languages import grammar_matches_file
except ImportError:  # lex-large's languages module has no dialect filter
    def grammar_matches_file(lang: str, path: str) -> bool:
        return True

FEATURE_KEYS = (
    'kind', 'len_bucket', 'first_char', 'last_char', 'hash1', 'hash2',
    'flags', 'trans_prev', 'trans_next', 'sym_prev', 'sym_next',
)
# lex-large imports this module with its own `tokenizer`, so the v2 field
# names are spelled out here rather than read from tokenizer.V2_FIELDS.
FEATURE_KEYS_V2 = FEATURE_KEYS + (
    'gap_prev', 'gap_next', 'indent', 'line_first', 'brace_depth', 'paren_depth',
)


def feature_keys(version: int) -> tuple[str, ...]:
    return FEATURE_KEYS if version == 1 else FEATURE_KEYS_V2

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
    """Deterministic per-content split: a byte-identical file (vendored copy,
    generated boilerplate, licence header) always lands in the same split no
    matter how many repo paths it was fetched under."""
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
        self.keys = tuple(k for k in flat if k != 'label')

    @classmethod
    def from_samples(cls, samples: list[dict], seq_len: int,
                     keys: tuple[str, ...] = FEATURE_KEYS) -> 'LexerDataset':
        if not samples:
            empty = {k: np.zeros(0, dtype=np.int64) for k in keys}
            empty['label'] = np.zeros(0, dtype=np.int64)
            return cls(empty, np.zeros(1, dtype=np.int64),
                       np.zeros(0, dtype=np.int64), seq_len)
        offsets = np.zeros(len(samples) + 1, dtype=np.int64)
        for i, s in enumerate(samples):
            offsets[i + 1] = offsets[i] + len(s['label'])
        keys = tuple(k for k in samples[0] if k not in ('label', 'lang'))
        flat = {k: np.concatenate([s[k] for s in samples]).astype(np.int64)
                for k in keys}
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
        for k in self.keys:
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


def file_groups(ds: LexerDataset, seq_len: int = 512) -> np.ndarray:
    """Group id per window, where every group is one or more whole source files.

    The cache stores no file ids, but build() appends each file's chunks
    contiguously and only a file's final chunk can be shorter than seq_len. So a
    short window or a language change always ends a file. A file whose last
    chunk is exactly seq_len long merges with the next file of the same
    language -- groups can over-merge but never split a file, which is the
    property cross-fitting needs.
    """
    lengths = np.diff(ds.offsets)
    new = np.ones(len(ds), dtype=bool)
    if len(ds) > 1:
        new[1:] = (lengths[:-1] < seq_len) | (ds.langs[1:] != ds.langs[:-1])
    return np.cumsum(new) - 1


def teacher_folds(ds: LexerDataset, n_folds: int = 2, min_groups: int = 4,
                  seq_len: int = 512) -> np.ndarray:
    """Fold id per training window for cross-fitted teachers; -1 means shared.

    Whole file groups are assigned greedily to the fold holding the fewest of
    that language's windows so far, so every fold sees every language in about
    equal measure. A language with fewer than `min_groups` groups cannot be
    split without starving one fold of it entirely, so its windows are shared:
    every fold trains on them and their cached logits are not held out.
    """
    groups = file_groups(ds, seq_len)
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    sizes = np.diff(np.r_[starts, len(ds)])
    group_langs = ds.langs[starts]
    group_fold = np.full(len(starts), -1, dtype=np.int8)
    n_groups_per_lang = np.bincount(group_langs)
    load = np.zeros((len(n_groups_per_lang), n_folds), dtype=np.int64)
    # Largest groups first, so the greedy balance is not decided by the tail.
    for g in np.argsort(-sizes, kind='stable'):
        lang = group_langs[g]
        if n_groups_per_lang[lang] < min_groups:
            continue
        fold = int(np.argmin(load[lang]))
        group_fold[g] = fold
        load[lang, fold] += sizes[g]
    return np.repeat(group_fold, sizes)


def _chunk(arrays: dict[str, np.ndarray], labels: np.ndarray, lang_id: int,
           seq_len: int, min_len: int) -> list[dict]:
    keys = tuple(arrays)
    T = len(labels)
    out = []
    for start in range(0, T, seq_len):
        end = min(start + seq_len, T)
        if end - start < min_len:
            continue
        piece = {k: arrays[k][start:end] for k in keys}
        piece['label'] = labels[start:end]
        piece['lang'] = lang_id
        if int((piece['label'] >= 0).sum()) < min_len // 2:
            continue  # almost all whitespace; nothing to learn
        out.append(piece)
    return out


def _prepare_file(job: tuple[str, str, int, int, int, int]):
    """Tokenize, align and chunk one labelled file (runs in a worker process)."""
    path, rle, lang_id, seq_len, min_len, version = job
    try:
        code = Path(path).read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None
    char_cls = decode_rle(rle, len(code))
    if version == 1:
        # lex-large's tokenizer module only has the version-1 entry points.
        toks = tokenizer.tokenize(code)
    else:
        toks = tokenizer.tokenize_version(code, version)
    if len(toks) < min_len:
        return None
    labels = align(toks, char_cls)
    if int((labels >= 0).sum()) < min_len // 2:
        return None
    arrays = (tokenizer.tokens_to_arrays(toks) if version == 1
              else tokenizer.tokens_to_arrays(toks, version))
    return split_of(code), _chunk(arrays, labels, lang_id, seq_len, min_len)


def _label_jobs(label_files, lang_index, seq_len, min_len, version, stats):
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
                if not grammar_matches_file(lang, path):
                    stats['files_skipped_grammar'] += 1
                    continue
                yield lang, (path, rle, lang_id, seq_len, min_len, version)


def build(
    label_dir: str = './corpus/labels',
    seq_len: int = 512,
    min_len: int = 32,
    total_tokens: int | None = None,
    max_dup: int = 3,
    cache: str | None = './corpus/dataset_v2',
    verbose: bool = True,
    feature_version: int = 2,
    workers: int = 0,
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

    if cached_meta is not None and cached_meta.get('feature_version', 1) != feature_version:
        # Never rebuild over a cache of another feature layout: other runs and
        # checkpoints depend on it. Pick a different --dataset directory.
        raise ValueError(
            f"{cache_path} holds feature version {cached_meta.get('feature_version', 1)}, "
            f'not {feature_version}; use a separate cache directory')

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
        'files_skipped_grammar': 0,
        'tokens_per_lang': Counter(), 'labels': Counter(),
    }

    import multiprocessing as mp
    n_workers = workers or max(1, (os.cpu_count() or 2) - 2)
    jobs = _label_jobs(label_files, lang_index, seq_len, min_len, feature_version, stats)
    pending_langs: list[str] = []

    def job_stream():
        # Budget check before dispatch, as the sequential build did; the check is
        # slightly optimistic under parallelism (in-flight files are not yet
        # counted), so the same check is repeated when a result is consumed.
        for lang, job in jobs:
            if used_tokens[lang] >= budgets.get(lang, 0):
                stats['files_skipped_budget'] += 1
                continue
            pending_langs.append(lang)
            yield job

    with mp.get_context('fork').Pool(n_workers) as pool:
        # imap keeps file order, so the dedup and budget decisions below are
        # deterministic regardless of worker count.
        for i, result in enumerate(pool.imap(_prepare_file, job_stream(), chunksize=16)):
            lang = pending_langs[i]
            if result is None:
                continue
            if used_tokens[lang] >= budgets.get(lang, 0):
                stats['files_skipped_budget'] += 1
                continue
            split, chunks = result
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

    keys = feature_keys(feature_version)
    datasets = {k: LexerDataset.from_samples(v, seq_len, keys)
                for k, v in per_split.items()}
    meta = _summarize(stats, used_tokens, seq_len, verbose)
    meta['total_tokens_requested'] = total_tokens
    meta['feature_version'] = feature_version
    meta['files_skipped_grammar'] = stats['files_skipped_grammar']
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
        flat = {k: z[k] for k in (*feature_keys(meta.get('feature_version', 1)), 'label')}
        datasets[split] = LexerDataset(flat, z['offsets'], z['lang'], seq_len)
    return datasets, meta


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='./corpus/labels')
    ap.add_argument('--seq-len', type=int, default=512)
    ap.add_argument('--total-tokens', type=int, default=42_000_000)
    ap.add_argument('--cache', default='./corpus/dataset_v2')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--feature-version', type=int, choices=(1, 2), default=2)
    ap.add_argument('--workers', type=int, default=0)
    args = ap.parse_args()
    ds, meta = build(args.labels, seq_len=args.seq_len, total_tokens=args.total_tokens,
                     cache=None if args.no_cache else args.cache,
                     feature_version=args.feature_version, workers=args.workers)
    for k, v in ds.items():
        print(f'{k:6s} {len(v):7,d} windows')
