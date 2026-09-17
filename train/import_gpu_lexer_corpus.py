"""
Import gpu-lexer's own training corpus into lex's training data, labelled by
gpu-lexer's own methodology rather than re-run through Shiki + scope_map.mjs.

Why label these files gpu-lexer's way instead of lex's way: the whole point is
to give lex direct exposure to the exact ground-truth rubric its accuracy is
now being measured against (eval_real_bench.py, and the "gpu-lexer verification
corpus" benchmark in demo/) -- training on scope_map.mjs labels and evaluating
on classes.js labels bakes in a taxonomy mismatch no amount of extra data can
close (see summary.md #15). gpu-lexer's own shards already carry sourceLabels
computed from classes.js (class name + confidence), so no re-labelling pass is
needed -- this script only needs to expand those spans into lex's per-character
RLE format and gate out confidence<1 spans, exactly as gpu-lexer's own
correctness.js does (its Uint8Array supervisionWeights truncates confidence 0.5
to 0 and excludes it from scoring).

Pulls only the `train` and `mining` splits. The `verification` split is never
touched here -- it must stay held-out for eval_real_bench.py, which scores
checkpoints against it directly from gpu-lexer's shard. Importing verification
files into the training corpus would leak the eval set into training and
invalidate every real-bench number this pipeline reports.

Usage:
    python import_gpu_lexer_corpus.py \
        --gpu-lexer-root ../../gpu-lexer --raw ./corpus/raw --labels ./corpus/labels

Output:
    corpus/raw/<lex-lang>/gpu_<owner>__<repo>__<path>   -- one file per import
    corpus/labels/labels.gpu.tsv                        -- <path>\\t<rle> lines,
                                                             same format build_labels.py emits
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

from labels import CLASS_NAMES
from languages import TARGET_LANGUAGES, BENCHMARK_FAMILY_TO_LANGUAGE

MASK = 9
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Extends languages.BENCHMARK_FAMILY_TO_LANGUAGE (which only covers the top-25
# benchmark families) with aliases needed for the wider train/mining corpus.
# Families with no entry here, and not already equal to a TARGET_LANGUAGES id,
# are skipped -- they'd need a new language slot in the model's language
# embedding table (languages.TARGET_LANGUAGES / model.py), which is a bigger,
# separate change from importing more data for languages lex already trains.
EXTRA_FAMILY_ALIAS = {
    'emacs-lisp': 'elisp',
    'vim-script': 'viml',
    'objective-cpp': 'objcpp',
}


def lex_lang_for(family: str) -> str | None:
    if family in TARGET_LANGUAGES:
        return family
    mapped = BENCHMARK_FAMILY_TO_LANGUAGE.get(family) or EXTRA_FAMILY_ALIAS.get(family)
    if mapped in TARGET_LANGUAGES:
        return mapped
    return None


def sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name)


def rle_encode(classes) -> str:
    runs = []
    for c in classes:
        if runs and runs[-1][0] == c:
            runs[-1][1] += 1
        else:
            runs.append([c, 1])
    return ','.join(f'{c}:{n}' for c, n in runs)


def char_classes_confidence_gated(source: str, source_labels: list[dict]):
    """Per-character class array, gpu-lexer's way: confidence<1 spans stay
    masked (9), matching correctness.js's confidence-gated denominator."""
    out = [MASK] * len(source)
    for label in source_labels:
        if label.get('confidence', 1) < 1:
            continue
        cls = CLASS_INDEX.get(label['class'])
        if cls is None:
            continue
        for i in range(label['from'], min(label['to'], len(source))):
            out[i] = cls
    return out


def load_shard(path: Path, wanted_splits: set[str]):
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get('split') in wanted_splits:
                yield item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu-lexer-root', default='../../gpu-lexer')
    ap.add_argument('--raw', default='./corpus/raw')
    ap.add_argument('--labels', default='./corpus/labels')
    ap.add_argument('--shard-name', default='labels.gpu.tsv')
    args = ap.parse_args()

    shards_dir = Path(args.gpu_lexer_root) / 'packages/training/data/generated/shards'
    raw_dir = Path(args.raw)
    labels_dir = Path(args.labels)
    labels_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = labels_dir / args.shard_name

    written = Counter()
    skipped_lang = Counter()
    skipped_bad = 0
    seen_paths = set()

    with out_tsv.open('w', encoding='utf-8') as tsv:
        for shard_file, splits in (('train.jsonl.gz', {'train'}), ('mining.jsonl.gz', {'mining'})):
            shard_path = shards_dir / shard_file
            if not shard_path.exists():
                print(f'!! missing {shard_path}, skipping')
                continue
            for item in load_shard(shard_path, splits):
                assert item['split'] != 'verification'  # never import the eval set
                family = item.get('family')
                lex_lang = lex_lang_for(family) if family else None
                if lex_lang is None:
                    skipped_lang[family] += 1
                    continue
                source = item.get('source', '')
                labels = item.get('sourceLabels')
                if not source or not labels:
                    skipped_bad += 1
                    continue

                fname = 'gpu_' + sanitize(f"{item.get('sourceName', 'unknown')}__{item.get('path', '')}")
                lang_dir = raw_dir / lex_lang
                lang_dir.mkdir(parents=True, exist_ok=True)
                file_path = lang_dir / fname
                stem, suffix = file_path.stem, file_path.suffix
                n = 1
                while str(file_path) in seen_paths or file_path.exists():
                    n += 1
                    file_path = lang_dir / f'{stem}_{n}{suffix}'
                seen_paths.add(str(file_path))

                file_path.write_text(source, encoding='utf-8')
                classes = char_classes_confidence_gated(source, labels)
                rle = rle_encode(classes)
                tsv.write(f'{file_path}\t{rle}\n')
                written[lex_lang] += 1

    total = sum(written.values())
    print(f'imported {total:,} files across {len(written)} languages -> {out_tsv}')
    for lang, n in sorted(written.items(), key=lambda kv: -kv[1]):
        print(f'  {lang:12s} {n:5,d}')
    if skipped_lang:
        total_skipped = sum(skipped_lang.values())
        print(f'\nskipped {total_skipped:,} files (no target-language mapping):')
        for fam, n in sorted(skipped_lang.items(), key=lambda kv: -kv[1])[:20]:
            print(f'  {fam:20s} {n:5,d}')
    if skipped_bad:
        print(f'skipped {skipped_bad:,} files with no source/labels')
    print('\nnext: python data_pipeline.py --labels ./corpus/labels --cache ./corpus/dataset '
          '--total-tokens <N>   # rebuild the cached dataset including this shard')


if __name__ == '__main__':
    main()
