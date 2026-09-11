"""Publish the prepared numeric cache as a resumable Hugging Face dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_card(repo_id: str, meta: dict, files: list[dict]) -> str:
    return f"""---
pretty_name: lex-large
task_categories:
- token-classification
tags:
- source-code
- syntax-highlighting
---

# lex-large

Prepared numeric training cache for the 99.97 KiB, 193-grammar neural lexer.
It contains token features and nine-class syntax labels, not the original source
text. File-level hashing fixes the train/validation/test split before upload.

## Contents

- `train.npz`, `val.npz`, `test.npz`: ragged token windows and split indices
- `meta.json`: counts, label distribution, and per-language coverage
- `language_manifest.json`: pinned Highlight.js grammar mapping
- `sources.json`: GitHub repository provenance used by corpus preparation
- `checksums.json`: byte sizes and SHA-256 checksums

Total labelled tokens: {meta.get('total_tokens', 0):,}

Languages covered: {meta.get('languages_covered', 0)}/{meta.get('languages_target', 193)}

Sequence length: {meta.get('seq_len', 512)}

## Download

```bash
cd /path/to/lexer/train_large
hf download {repo_id} --repo-type dataset --local-dir ./corpus/dataset
.venv/bin/python test_setup.py
```

The cache is consumed directly by `train_teacher.py`, `train.py`, and
`eval_coverage.py`. See `train_large/note.txt` in the lexer repository for the
complete commands.

## Integrity

This revision contains {len(files)} payload files. Verify them against
`checksums.json` after download.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='./corpus/dataset')
    parser.add_argument('--repo-id', default='lex-large')
    parser.add_argument('--private', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--allow-underfilled', action='store_true',
                        help='skip the 193-language / 95%%-of-target-tokens checks, '
                             'for a smoke-test upload of a known-underfilled cache')
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    meta_path = folder / 'meta.json'
    required = [folder / name for name in ('train.npz', 'val.npz', 'test.npz', 'meta.json')]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f'dataset cache is incomplete: {", ".join(missing)}')

    meta = json.loads(meta_path.read_text())
    if meta.get('languages_covered') != 193 and not args.allow_underfilled:
        raise SystemExit(f'expected 193 covered languages, found {meta.get("languages_covered")}')
    if (meta.get('total_tokens', 0) < int(meta.get('total_tokens_requested', 160_000_000) * 0.95)
            and not args.allow_underfilled):
        raise SystemExit('dataset is below 95 percent of its requested token count')

    for source, name in (
        (HERE / 'language_manifest.json', 'language_manifest.json'),
        (HERE / 'repos.json', 'sources.json'),
    ):
        if source.exists():
            shutil.copyfile(source, folder / name)

    api = HfApi()
    if '/' not in args.repo_id:
        owner = api.whoami()['name']
        repo_id = f'{owner}/{args.repo_id}'
    else:
        repo_id = args.repo_id

    payload_names = (
        'train.npz', 'val.npz', 'test.npz', 'meta.json',
        'language_manifest.json', 'sources.json',
    )
    files = [
        {'path': name, 'bytes': (folder / name).stat().st_size, 'sha256': sha256(folder / name)}
        for name in payload_names if (folder / name).exists()
    ]
    (folder / 'checksums.json').write_text(json.dumps({
        'format': 'lex-large-checksums-v1',
        'repo_id': repo_id,
        'files': files,
    }, indent=2) + '\n')
    (folder / 'README.md').write_text(dataset_card(repo_id, meta, files))

    api.create_repo(repo_id, repo_type='dataset', private=args.private, exist_ok=True)
    print(f'Uploading {folder} to https://huggingface.co/datasets/{repo_id}', flush=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type='dataset',
        folder_path=folder,
        num_workers=args.workers,
        print_report=True,
    )
    print(f'UPLOAD COMPLETE: https://huggingface.co/datasets/{repo_id}', flush=True)


if __name__ == '__main__':
    main()
