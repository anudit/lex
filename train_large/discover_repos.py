"""Discover diverse high-signal GitHub repositories for every Linguist language."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from languages import LANGUAGE_META, TARGET_LANGUAGES
from repos import REPOS

HERE = Path(__file__).resolve().parent

# Highlight.js includes REPLs, templates, logs, and dialects that GitHub
# Linguist does not count independently. Search their closest source ecosystem;
# extension filtering still decides which files enter each corpus directory.
RELATED_LINGUIST = {
    'arduino': 'C++', 'avrasm': 'Assembly', 'clojure-repl': 'Clojure',
    'crmsh': 'Shell', 'dts': 'Device Tree', 'dust': 'HTML',
    'erlang-repl': 'Erlang', 'excel': 'VBA', 'irpf90': 'Fortran',
    'jboss-cli': 'Shell', 'leaf': 'HTML', 'mipsasm': 'Assembly',
    'mojolicious': 'Perl', 'n1ql': 'SQL', 'node-repl': 'JavaScript',
    'parser3': 'HTML', 'pgsql': 'PLpgSQL', 'php-template': 'PHP',
    'profile': 'Shell', 'python-repl': 'Python', 'reasonml': 'Reason',
    'subunit': 'Python', 'vbscript-html': 'VBScript',
}


def _request(url: str, token: str | None) -> dict:
    headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'lex-train-large/1.0'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def search(language: str, limit: int, min_stars: int, token: str | None) -> list[str]:
    query = f'language:"{language}" stars:>={min_stars} archived:false mirror:false'
    url = 'https://api.github.com/search/repositories?' + urllib.parse.urlencode({
        'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': min(100, limit * 3),
    })
    payload = _request(url, token)
    out = []
    owners = set()
    for item in payload.get('items', []):
        owner = item['owner']['login'].lower()
        if item.get('fork') or owner in owners:
            continue
        owners.add(owner)
        out.append(item['full_name'])
        if len(out) == limit:
            break
    return out


def search_extension(extension: str, limit: int, token: str) -> list[str]:
    suffix = extension.rsplit('.', 1)[-1]
    url = 'https://api.github.com/search/code?' + urllib.parse.urlencode({
        'q': f'extension:{suffix}', 'per_page': min(100, limit * 4),
    })
    payload = _request(url, token)
    out = []
    seen = set()
    for item in payload.get('items', []):
        name = item['repository']['full_name']
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(name)
        if len(out) == limit:
            break
    return out


def discover(language: str, limit: int, min_stars: int, token: str | None) -> list[str]:
    """Use Linguist first, then extension search for sparse or mismatched names."""
    meta = LANGUAGE_META[language]
    linguist = meta.get('linguist') or RELATED_LINGUIST.get(language)
    found: list[str] = []
    if linguist:
        found.extend(search(linguist, limit, min_stars, token))
        if len(found) < limit and min_stars > 0:
            found.extend(search(linguist, limit, 0, token))
    if token and len(set(found)) < limit:
        # GitHub and Linguist occasionally disagree on a display name (for
        # example Fortran Free Form). Code search by pinned extension recovers
        # those repositories, at a lower API limit, so use it only as fallback.
        for extension in meta.get('extensions', ())[:3]:
            found.extend(search_extension(extension, limit, token))
            if len(set(found)) >= limit:
                break
            time.sleep(6.2)
    return list(dict.fromkeys(found))[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(HERE / 'repos.json'))
    parser.add_argument('--languages', default='')
    parser.add_argument('--per-language', type=int, default=24)
    parser.add_argument('--min-stars', type=int, default=5)
    args = parser.parse_args()
    selected = ([item.strip() for item in args.languages.split(',') if item.strip()]
                or list(TARGET_LANGUAGES))
    out_path = Path(args.out)
    discovered = json.loads(out_path.read_text()) if out_path.exists() else {}
    token = os.environ.get('GITHUB_TOKEN')
    pause = 2.2 if token else 6.2

    for index, language in enumerate(selected, 1):
        if len(set(REPOS.get(language, ())) | set(discovered.get(language, ()))) >= args.per_language:
            continue
        linguist = LANGUAGE_META[language].get('linguist') or RELATED_LINGUIST.get(language)
        extension = next(iter(LANGUAGE_META[language].get('extensions', ())), None)
        if not linguist and not (token and extension):
            print(f'[{index:3d}/{len(selected)}] {language}: no Linguist mapping; '
                  f'use GITHUB_TOKEN for extension search or seed manually')
            continue
        try:
            found = discover(language, args.per_language, args.min_stars, token)
        except urllib.error.HTTPError as error:
            if error.code in (403, 429):
                reset = int(error.headers.get('X-RateLimit-Reset', time.time() + 60))
                wait = max(5, reset - int(time.time()) + 2)
                print(f'  rate limited; waiting {wait}s')
                time.sleep(wait)
                found = discover(language, args.per_language, args.min_stars, token)
            else:
                raise
        discovered[language] = list(dict.fromkeys([*discovered.get(language, ()), *found]))
        out_path.write_text(json.dumps(discovered, indent=2, sort_keys=True) + '\n')
        print(f'[{index:3d}/{len(selected)}] {language}: {len(discovered[language])} repos')
        time.sleep(pause)


if __name__ == '__main__':
    main()
