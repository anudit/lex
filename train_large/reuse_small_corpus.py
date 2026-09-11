"""Seed the large raw corpus from the completed compact run without extra disk use."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from languages import TARGET_LANGUAGES

HERE = Path(__file__).resolve().parent
ALIASES = {
    'asm': 'x86asm',
    'bat': 'dos',
    'elisp': 'lisp',
    'html': 'xml',
    'make': 'makefile',
    'objc': 'objectivec',
    'objcpp': 'objectivec',
    'plsql': 'pgsql',
    'shell': 'bash',
    'tsql': 'sql',
    'viml': 'vim',
    'xslt': 'xml',
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=str(HERE.parent / 'train' / 'corpus' / 'raw'))
    parser.add_argument('--out', default='./corpus/raw')
    args = parser.parse_args()

    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    if not source.is_dir():
        raise SystemExit(f'compact raw corpus not found: {source}')

    targets = set(TARGET_LANGUAGES)
    linked = copied = existing = files = total_bytes = 0
    languages = set()
    for directory in sorted(path for path in source.iterdir() if path.is_dir()):
        target_language = ALIASES.get(directory.name, directory.name)
        if target_language not in targets:
            continue
        target_dir = out / target_language
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if not path.is_file():
                continue
            target = target_dir / f'small__{directory.name}__{path.name}'
            if target.exists():
                existing += 1
            else:
                try:
                    os.link(path, target)
                    linked += 1
                except OSError:
                    shutil.copy2(path, target)
                    copied += 1
            files += 1
            total_bytes += path.stat().st_size
            languages.add(target_language)

    print(f'Reused {files:,} compact files ({total_bytes / 1024**2:.1f} MiB) '
          f'across {len(languages)} large grammars')
    print(f'Hardlinked {linked:,}, copied {copied:,}, already present {existing:,}')


if __name__ == '__main__':
    main()
