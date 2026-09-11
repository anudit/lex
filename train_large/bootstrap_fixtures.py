"""Seed every grammar from pinned Highlight.js and GitHub Linguist fixtures.

These fixtures guarantee representation and exercise rare constructs, but they
are not repeated to satisfy token budgets. Repository source must provide the
bulk of every language's 300k-token minimum.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from languages import LANGUAGE_META

HERE = Path(__file__).resolve().parent
UA = {'User-Agent': 'lex-train-large/1.0'}


def archive(url: str) -> tarfile.TarFile:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    return tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz')


def clean(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')


def write_member(tar: tarfile.TarFile, member: tarfile.TarInfo,
                 target: Path) -> bool:
    if not member.isfile() or member.size <= 0 or member.size > 400_000:
        return False
    handle = tar.extractfile(member)
    if handle is None:
        return False
    raw = handle.read()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        return False
    if '\x00' in text or not text.strip():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return True


def seed_highlight(root: Path, commit: str) -> Counter:
    counts = Counter()
    url = f'https://github.com/highlightjs/highlight.js/archive/{commit}.tar.gz'
    with archive(url) as tar:
        for member in tar:
            parts = member.name.split('/')
            if len(parts) < 5 or parts[1] != 'test' or parts[2] not in ('detect', 'markup'):
                continue
            language = parts[3]
            if language not in LANGUAGE_META or not parts[-1].endswith('.txt'):
                continue
            if parts[-1].endswith('.expect.txt'):
                continue
            name = clean('__'.join(parts[2:]))
            if write_member(tar, member, root / language / f'highlightjs__{name}'):
                counts[language] += 1
    return counts


def seed_linguist(root: Path, commit: str) -> Counter:
    by_name = {
        meta['linguist']: language
        for language, meta in LANGUAGE_META.items() if meta.get('linguist')
    }
    counts = Counter()
    url = f'https://github.com/github-linguist/linguist/archive/{commit}.tar.gz'
    with archive(url) as tar:
        for member in tar:
            parts = member.name.split('/')
            if len(parts) < 4 or parts[1] != 'samples':
                continue
            language = by_name.get(parts[2])
            if not language:
                continue
            name = clean('__'.join(parts[2:]))
            if write_member(tar, member, root / language / f'linguist__{name}'):
                counts[language] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='./corpus/raw')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'language_manifest.json').read_text())
    root = Path(args.out)
    highlight = seed_highlight(root, manifest['highlightJs']['releaseCommit'])
    linguist = seed_linguist(root, manifest['githubLinguist']['commit'])
    covered = {language for language in LANGUAGE_META if highlight[language] or linguist[language]}
    print(f'Highlight.js fixtures: {sum(highlight.values()):,}')
    print(f'GitHub Linguist samples: {sum(linguist.values()):,}')
    print(f'Coverage: {len(covered)}/{len(LANGUAGE_META)} grammars')
    missing = sorted(set(LANGUAGE_META) - covered)
    if missing:
        print(f'Missing fixtures: {", ".join(missing)}')


if __name__ == '__main__':
    main()
