"""Import gpu-lexer's train/mining corpus with its native class labels.

The verification split is deliberately excluded: it is the held-out benchmark.
The output shard sorts before the general teacher shards so these examples are
seen before per-language token budgets fill.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / 'train'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from labels import CLASS_NAMES
from languages import TARGET_LANGUAGES, benchmark_family_language

MASK = 9
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
PREFIX = 'gpu_lexer__'

# Canonical equivalents whose names differ between gpu-lexer families and the
# pinned Highlight.js grammar ids. Avoid approximate mappings that would train
# examples under the wrong language-conditioning row.
FAMILY_ALIASES = {
    'assembly': 'x86asm',
    'batchfile': 'dos',
    'emacs-lisp': 'lisp',
    'html': 'xml',
    'objective-c': 'objectivec',
    'objective-cpp': 'objectivec',
    'plpgsql': 'pgsql',
    'shell': 'bash',
    'systemverilog': 'verilog',
    'tsql': 'sql',
    'vb': 'vbnet',
    'vim-script': 'vim',
    'webassembly': 'wasm',
    'xslt': 'xml',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def target_language(family: str) -> str | None:
    targets = set(TARGET_LANGUAGES)
    candidate = FAMILY_ALIASES.get(family, benchmark_family_language(family))
    return candidate if candidate in targets else None


def sanitize(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('._') or 'unknown'


def utf16_boundaries(source: str) -> list[int]:
    """Map JavaScript UTF-16 offsets in sourceLabels to Python indices."""
    out = [0]
    for index, char in enumerate(source, 1):
        out.append(index)
        if ord(char) > 0xFFFF:
            out.append(index)
    return out


def char_classes(source: str, labels: list[dict]) -> list[int]:
    result = [MASK] * len(source)
    boundaries = utf16_boundaries(source)
    last = len(boundaries) - 1
    for label in labels:
        if label.get('confidence', 1) < 1:
            continue
        cls = CLASS_INDEX.get(label.get('class'))
        if cls is None:
            continue
        start = boundaries[min(max(0, int(label['from'])), last)]
        end = boundaries[min(max(0, int(label['to'])), last)]
        result[start:end] = [cls] * max(0, end - start)
    return result


def rle_encode(classes: list[int]) -> str:
    runs: list[list[int]] = []
    for cls in classes:
        if runs and runs[-1][0] == cls:
            runs[-1][1] += 1
        else:
            runs.append([cls, 1])
    return ','.join(f'{cls}:{count}' for cls, count in runs)


def items(path: Path, split: str):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            item = json.loads(line)
            if item.get('split') == split:
                yield item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-lexer-root', default='../../gpu-lexer')
    parser.add_argument('--raw', default='./corpus/raw')
    parser.add_argument('--labels', default='./corpus/labels')
    parser.add_argument('--shard-name', default='labels.-gpu.tsv')
    args = parser.parse_args()

    shards = Path(args.gpu_lexer_root).resolve() / 'packages/training/data/generated/shards'
    inputs = [('train.jsonl.gz', 'train'), ('mining.jsonl.gz', 'mining')]
    verification_path = shards / 'verification.jsonl.gz'
    missing = ([str(shards / name) for name, _ in inputs if not (shards / name).is_file()]
               + ([] if verification_path.is_file() else [str(verification_path)]))
    if missing:
        raise SystemExit(f'missing gpu-lexer shards: {", ".join(missing)}')

    raw = Path(args.raw).resolve()
    labels_dir = Path(args.labels).resolve()
    labels_dir.mkdir(parents=True, exist_ok=True)
    for path in raw.glob(f'*/{PREFIX}*'):
        if path.is_file():
            path.unlink()

    written: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    supervised = source_chars = 0
    label_path = labels_dir / args.shard_name
    temporary = label_path.with_suffix(label_path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as output:
        for shard_name, split in inputs:
            for item in items(shards / shard_name, split):
                family = item.get('family')
                language = target_language(family) if family else None
                source = item.get('source', '')
                source_labels = item.get('sourceLabels')
                if language is None:
                    skipped[str(family)] += 1
                    continue
                if not source or not source_labels:
                    skipped['<missing-source-or-labels>'] += 1
                    continue
                identity = '\0'.join((split, str(item.get('sourceName', '')),
                                      str(item.get('path', '')), source))
                digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
                basename = sanitize(Path(str(item.get('path', 'source'))).name)
                filename = f'{PREFIX}{sanitize(str(item.get("sourceName", "unknown")))}__{digest}__{basename}'
                path = raw / language / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding='utf-8')
                classes = char_classes(source, source_labels)
                output.write(f'{path}\t{rle_encode(classes)}\n')
                written[language] += 1
                source_chars += len(source)
                supervised += sum(cls != MASK for cls in classes)
    temporary.replace(label_path)

    verification_hashes: set[str] = set()
    for item in items(verification_path, 'verification'):
        source = item.get('source', '')
        if source:
            verification_hashes.add(hashlib.sha256(source.encode()).hexdigest())
    exclusion_path = labels_dir / 'exclude.gpu-verification.sha256'
    exclusion_path.write_text(''.join(f'{digest}\n' for digest in sorted(verification_hashes)))

    provenance = {
        'format': 'lex-large-gpu-lexer-import-v1',
        'verification_excluded': True,
        'source_shards': [
            {'name': name, 'split': split, 'bytes': (shards / name).stat().st_size,
             'sha256': sha256(shards / name)}
            for name, split in inputs
        ],
        'verification_shard': {
            'name': verification_path.name,
            'bytes': verification_path.stat().st_size,
            'sha256': sha256(verification_path),
            'unique_source_hashes_excluded': len(verification_hashes),
            'exclusion_file': exclusion_path.name,
        },
        'label_shard': args.shard_name,
        'files': sum(written.values()),
        'source_characters': source_chars,
        'supervised_characters': supervised,
        'files_per_language': dict(sorted(written.items())),
        'skipped_families': dict(sorted(skipped.items())),
        'family_aliases': FAMILY_ALIASES,
    }
    provenance_path = labels_dir / 'gpu_lexer_import.json'
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'imported {provenance["files"]:,} gpu-lexer train/mining files '
          f'across {len(written)} languages')
    print(f'supervised {supervised:,} / {source_chars:,} source characters')
    print(f'labels: {label_path}')
    print(f'provenance: {provenance_path}')
    print(f'excluded {len(verification_hashes):,} unique verification-source hashes')
    if skipped:
        print(f'skipped {sum(skipped.values()):,} files without an exact large-language mapping')


if __name__ == '__main__':
    main()
