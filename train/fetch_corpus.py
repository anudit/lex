"""
Corpus fetcher: downloads real production source from top GitHub repositories.

Tarballs are streamed from codeload and decompressed on the fly; extraction stops
as soon as a repo's byte budget is met, so the download aborts early instead of
pulling multi-gigabyte archives for repos like torvalds/linux or microsoft/vscode.

Unified diffs cannot be harvested this way -- almost no repo checks in .diff files
-- so the `diff` language is built from real commit patches off the GitHub API.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from languages import TARGET_LANGUAGES, token_budgets
from repos import REPOS, FILENAME_MATCH

# Extensions per language. Kept deliberately tight: a loose mapping is how a
# corpus ends up with .h files labelled as C++ or .ts files labelled JavaScript.
LANG_EXTENSIONS: dict[str, tuple[str, ...]] = {
    'javascript': ('.js', '.mjs', '.cjs', '.jsx'),
    'typescript': ('.ts', '.mts', '.cts', '.tsx'),
    'python': ('.py', '.pyi'),
    'rust': ('.rs',),
    'go': ('.go',),
    'c': ('.c',),
    'cpp': ('.cpp', '.cc', '.cxx', '.hpp', '.hh'),
    'csharp': ('.cs',),
    'java': ('.java',),
    'kotlin': ('.kt', '.kts'),
    'swift': ('.swift',),
    'ruby': ('.rb',),
    'php': ('.php',),
    'lua': ('.lua',),
    'zig': ('.zig',),
    'sql': ('.sql',),
    'shell': ('.sh', '.bash', '.zsh'),
    'powershell': ('.ps1', '.psm1', '.psd1'),
    'html': ('.html', '.htm'),
    'css': ('.css', '.scss'),
    'json': ('.json',),
    'yaml': ('.yaml', '.yml'),
    'toml': ('.toml',),
    'markdown': ('.md', '.mdx'),
    'dockerfile': ('.dockerfile',),
    'graphql': ('.graphql', '.gql'),
    'hcl': ('.tf', '.hcl', '.tfvars'),
    'diff': ('.diff', '.patch'),
    'plaintext': ('.txt',),
    'dart': ('.dart',),
    'scala': ('.scala', '.sc'),
    'perl': ('.pl', '.pm', '.t'),
    'r': ('.R', '.r'),
}

# Paths that add bulk without adding syntax: vendored trees, minified bundles,
# generated code, fixtures and lockfiles.
SKIP_PATH = re.compile(
    r'(^|/)(node_modules|vendor|third_party|thirdparty|external|deps|dist|build|'
    r'target|\.git|testdata|fixtures?|__snapshots__|generated|gen|godeps)(/|$)',
    re.IGNORECASE,
)
SKIP_NAME = re.compile(
    r'(\.min\.(js|css)$|\.pb\.go$|_pb2\.py$|\.generated\.|package-lock\.json$|'
    r'yarn\.lock$|pnpm-lock\.yaml$|Cargo\.lock$|composer\.lock$|\.d\.ts$)',
    re.IGNORECASE,
)

MIN_FILE_BYTES = 200
MAX_FILE_BYTES = 400_000
# Long files are truncated rather than taken whole. A byte budget spent on a few
# enormous documents buys tokens but not diversity: the markdown corpus reached
# its budget from 35 files at 23k tokens each -- six times the tokens-per-file of
# any other language -- and the model learned one house style well enough to read
# all prose as comment text, scoring 15% on unseen markdown.
TRUNCATE_BYTES = 30_000
# A single repository may contribute at most this multiple of its equal share.
MAX_REPO_SHARE = 2
UA = {'User-Agent': 'lexer-corpus-builder/1.0'}


def _matches(lang: str, member_path: str) -> bool:
    name = os.path.basename(member_path)
    lower = name.lower()
    if SKIP_PATH.search(member_path) or SKIP_NAME.search(name):
        return False
    for stem in FILENAME_MATCH.get(lang, ()):
        if lower == stem or lower.startswith(stem + '.'):
            return True
    return lower.endswith(LANG_EXTENSIONS[lang])


def harvest_repo(lang: str, repo: str, out_dir: Path, budget: int,
                 max_download: int = 400_000_000) -> int:
    """Stream one repo tarball, writing matching files until `budget` bytes are met.

    Returns bytes written. The response is closed as soon as the budget is hit,
    which aborts the remainder of the transfer.
    """
    url = f'https://codeload.github.com/{repo}/tar.gz/HEAD'
    written = 0
    files = 0
    slug = repo.replace('/', '__')
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            counting = _CountingReader(resp, max_download)
            with tarfile.open(fileobj=counting, mode='r|gz') as tar:
                for member in tar:
                    if written >= budget or counting.read_bytes >= max_download:
                        break
                    if not member.isfile() or member.size < MIN_FILE_BYTES:
                        continue
                    if member.size > MAX_FILE_BYTES:
                        continue
                    # Strip the leading "<repo>-HEAD/" component.
                    rel = member.name.split('/', 1)[1] if '/' in member.name else member.name
                    if not _matches(lang, rel):
                        continue
                    fh = tar.extractfile(member)
                    if fh is None:
                        continue
                    raw = fh.read()
                    try:
                        text = raw.decode('utf-8')
                    except UnicodeDecodeError:
                        continue
                    if '\x00' in text or len(text.strip()) < MIN_FILE_BYTES:
                        continue
                    if len(text) > TRUNCATE_BYTES:
                        # Cut on a line boundary so the tail is not a half token.
                        cut = text.rfind('\n', 0, TRUNCATE_BYTES)
                        text = text[:cut if cut > MIN_FILE_BYTES else TRUNCATE_BYTES]
                    flat = f'{slug}__{rel.replace("/", "_")}'
                    if not flat.lower().endswith(LANG_EXTENSIONS[lang]):
                        flat += '.txt' if lang == 'plaintext' else '.dockerfile'
                    (out_dir / flat).write_text(text, encoding='utf-8')
                    written += len(text)
                    files += 1
    except _BudgetReached:
        pass
    except (urllib.error.URLError, tarfile.TarError, OSError, EOFError) as exc:
        print(f'    ! {repo}: {type(exc).__name__}: {exc}', file=sys.stderr)
    return written


class _BudgetReached(Exception):
    pass


class _CountingReader(io.RawIOBase):
    """Wraps a response so we can cap total compressed bytes pulled per repo."""

    def __init__(self, raw, limit: int):
        self._raw = raw
        self._limit = limit
        self.read_bytes = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self.read_bytes >= self._limit:
            raise _BudgetReached()
        chunk = self._raw.read(size)
        self.read_bytes += len(chunk)
        return chunk


def harvest_diffs(out_dir: Path, budget: int, token: str | None) -> int:
    """Build the `diff` corpus from real commit patches on high-traffic repos."""
    written = 0
    headers = dict(UA)
    if token:
        headers['Authorization'] = f'Bearer {token}'
    for repo in REPOS['diff']:
        if written >= budget:
            break
        api = f'https://api.github.com/repos/{repo}/commits?per_page=100'
        try:
            req = urllib.request.Request(api, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                commits = json.load(resp)
        except (urllib.error.URLError, ValueError) as exc:
            print(f'    ! {repo} commit list: {exc}', file=sys.stderr)
            continue
        slug = repo.replace('/', '__')
        for commit in commits:
            if written >= budget:
                break
            sha = commit.get('sha')
            if not sha:
                continue
            patch_url = f'https://github.com/{repo}/commit/{sha}.patch'
            try:
                req = urllib.request.Request(patch_url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read(MAX_FILE_BYTES)
            except (urllib.error.URLError, OSError):
                continue
            try:
                text = raw.decode('utf-8')
            except UnicodeDecodeError:
                continue
            if len(text) < MIN_FILE_BYTES:
                continue
            (out_dir / f'{slug}__{sha[:12]}.diff').write_text(text, encoding='utf-8')
            written += len(raw)
            time.sleep(0.05)  # stay polite on the unauthenticated rate limit
    return written


def fetch_language(lang: str, root: Path, budget: int, token: str | None) -> tuple[str, int, int]:
    out_dir = root / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file())
    if existing >= budget:
        return lang, existing, len(list(out_dir.iterdir()))

    written = existing
    if lang == 'diff':
        written += harvest_diffs(out_dir, budget - written, token)
    else:
        # Give every repository a share of the budget rather than draining them
        # in order. Taking the budget from the first two repos is how a language
        # ends up with one house style, and adding repos to the end of the list
        # then changes nothing because the budget is already met.
        repos = REPOS[lang]
        share = max(1, (budget - written) // max(1, len(repos)))
        taken: dict[str, int] = {}
        for repo in repos:
            if written >= budget:
                break
            got = harvest_repo(lang, repo, out_dir, share)
            taken[repo] = got
            written += got
        # Second pass for repos that had less than a share, but capped: a project
        # with thousands of tiny files (SPDX licence templates, say) would
        # otherwise absorb everyone else's leftover budget and end up as 91% of
        # the language. Many files from one project is still one distribution,
        # and validation drawn from the same source will not reveal the gap.
        cap = share * MAX_REPO_SHARE
        for repo in repos:
            if written >= budget:
                break
            room = min(budget - written, cap - taken.get(repo, 0))
            if room > 0:
                written += harvest_repo(lang, repo, out_dir, room)
    return lang, written, len(list(out_dir.iterdir()))


def main() -> None:
    ap = argparse.ArgumentParser(description='Download real source corpus from GitHub')
    ap.add_argument('--out', default='./corpus/raw', help='output root directory')
    ap.add_argument('--total-tokens', type=int, default=24_000_000,
                    help='token budget the corpus must support, split by language weight')
    ap.add_argument('--headroom', type=float, default=2.0,
                    help='raw bytes to fetch per token budget, above the ~2.5 B/token floor')
    ap.add_argument('--languages', default='', help='comma-separated subset (default: all 29)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--holdout', action='store_true',
                    help='fetch from repos_holdout.py instead of repos.py')
    args = ap.parse_args()

    if args.holdout:
        # holdout_repos() strips anything that also appears in training, so a
        # repository in both lists cannot silently contaminate the comparison.
        from repos_holdout import holdout_repos
        replacement = holdout_repos()
        REPOS.clear()
        REPOS.update(replacement)

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    langs = ([l.strip() for l in args.languages.split(',') if l.strip()]
             or [l for l in TARGET_LANGUAGES if l in REPOS])
    token = os.environ.get('GITHUB_TOKEN')

    # ~2.5 source bytes per lexer token, times headroom for files that get
    # dropped by the labeller or by near-duplicate filtering.
    budgets = token_budgets(args.total_tokens)
    byte_budget = {l: int(budgets.get(l, 250_000) * 2.5 * args.headroom) for l in langs}

    print(f'Fetching {len(langs)} languages for a {args.total_tokens:,}-token corpus -> {root}')
    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_language, l, root, byte_budget[l], token): l
                   for l in langs}
        for fut in as_completed(futures):
            lang, written, nfiles = fut.result()
            total += written
            print(f'  {lang:12s} {written / 1e6:7.2f} MB  {nfiles:5d} files')
    print(f'\nTotal raw corpus: {total / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
