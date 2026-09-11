"""
Fast corpus fetch for lexer, run on Google Colab.

Paste this whole file into a single Colab cell (or split at the "

Why this is faster than train_large/fetch_corpus.py:
  fetch_corpus.py downloads a tarball per GitHub repo via codeload.github.com
  and hits api.github.com for the `diff` language -- both are rate limited
  (secondary abuse-detection limits hit hard on popular repos like
  torvalds/linux). This script instead streams the pre-scraped, deduplicated,
  permissively-licensed `bigcode/the-stack-dedup` dataset from Hugging Face --
  no GitHub involved, no rate limits, and HF's CDN is much faster than
  per-repo git tarball streaming.

Coverage note:
  the-stack-dedup covers ~300 mainstream languages (Linguist-style names). Many
  of this project's 193 highlight.js grammar ids are obscure DSLs/configs
  (mizar, rib, xl, step21, ruleslanguage, dns, dsconfig, ...) that the-stack
  simply does not have. This script maps what it can (verified live against
  the dataset's actual config list, not guessed), fetches those, and prints a
  clear list of unmapped languages at the end -- leave the existing
  GitHub-based train_large/fetch_corpus.py running locally to mop up that
  long tail; it already resumes from whatever's on disk.

Output: corpus/raw/<lang>/<repo>__<path> files, same layout fetch_corpus.py
produces, then everything is zipped and pushed to a Hugging Face dataset repo
so you can pull it down to the Mac and drop it into train_large/corpus/raw/.

Total target across all languages is only ~1.2 GB (see BYTE_BUDGET below), so
Colab's disk is never the bottleneck -- streaming mode means we only ever
touch the shard bytes we actually keep.

Before running:
  1. Accept the dataset terms at https://huggingface.co/datasets/bigcode/the-stack-dedup
  2. Create a Hugging Face token (read is enough for fetching; write if you
     want this script to push the zip back to a dataset repo of your own)
  3. In Colab: Runtime > Change runtime type > any (CPU is fine, no GPU needed)
"""

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                 'datasets', 'huggingface_hub', 'tqdm'], check=True)

from huggingface_hub import login
login()  # paste your HF token when prompted

import re, json
from pathlib import Path
from difflib import get_close_matches

OUT = Path('/content/corpus/raw')
OUT.mkdir(parents=True, exist_ok=True)

# int(token_budgets(160_000_000)[lang] * 2.5 bytes/token * 3.0 headroom)
BYTE_BUDGET = {"1c": 5111850, "abnf": 5111850, "accesslog": 5111850, "actionscript": 5111850, "ada": 5111850, "angelscript": 5111850, "apache": 5111850, "applescript": 5111850, "arcade": 5111850, "arduino": 5111850, "armasm": 5111850, "asciidoc": 5111850, "aspectj": 5111850, "autohotkey": 5111850, "autoit": 5111850, "avrasm": 5111850, "awk": 6114367, "axapta": 5111850, "bash": 14139787, "basic": 5111850, "bnf": 5111850, "brainfuck": 5111850, "c": 13558440, "cal": 5111850, "capnproto": 5111850, "ceylon": 5111850, "clean": 5111850, "clojure": 5111850, "clojure-repl": 5111850, "cmake": 7420477, "coffeescript": 5111850, "coq": 5111850, "cos": 5111850, "cpp": 14962057, "crmsh": 5111850, "crystal": 5111850, "csharp": 15481882, "csp": 5111850, "css": 11778570, "d": 5111850, "dart": 9102975, "delphi": 5111850, "diff": 5111850, "django": 5111850, "dns": 5111850, "dockerfile": 7083450, "dos": 5111850, "dsconfig": 5111850, "dts": 5111850, "dust": 5111850, "ebnf": 5111850, "elixir": 5111850, "elm": 5111850, "erb": 5111850, "erlang": 5111850, "erlang-repl": 5111850, "excel": 5111850, "fix": 5111850, "flix": 5111850, "fortran": 5111850, "freedesktop": 5111850, "fsharp": 5111850, "gams": 5111850, "gauss": 5111850, "gcode": 5111850, "gherkin": 6326655, "glsl": 6326655, "gml": 5111850, "go": 13256257, "golo": 5111850, "gradle": 5111850, "graphql": 5111850, "groovy": 7083450, "haml": 5111850, "handlebars": 5111850, "haskell": 5111850, "haxe": 5111850, "hsp": 5111850, "http": 5111850, "hy": 5111850, "inform7": 5111850, "ini": 5111842, "irpf90": 5111842, "isbl": 5111842, "java": 18870157, "javascript": 26326350, "jboss-cli": 5111842, "json": 9344700, "julia": 5111842, "julia-repl": 5111842, "kotlin": 10238805, "lasso": 5111842, "latex": 5111842, "ldif": 5111842, "leaf": 5111842, "less": 5111842, "lisp": 5111842, "livecodeserver": 5111842, "livescript": 5111842, "llvm": 5111842, "lsl": 5111842, "lua": 7736835, "makefile": 8035927, "markdown": 11599972, "mathematica": 5111842, "matlab": 5111842, "maxima": 5111842, "mel": 5111842, "mercury": 5111842, "mipsasm": 5111842, "mizar": 5111842, "mojolicious": 5111842, "monkey": 5111842, "moonscript": 5111842, "n1ql": 5111842, "nestedtext": 5111842, "nginx": 5111842, "nim": 5111842, "nix": 5111842, "node-repl": 5111842, "nsis": 5111842, "objectivec": 5111842, "ocaml": 5111842, "openscad": 5111842, "oxygene": 5111842, "parser3": 5111842, "perl": 7420477, "pf": 5111842, "pgsql": 5111842, "php": 14420055, "php-template": 5111842, "plaintext": 5111842, "pony": 5111842, "powershell": 8035927, "processing": 5111842, "profile": 5111842, "prolog": 5111842, "properties": 5111842, "protobuf": 5111842, "puppet": 5111842, "purebasic": 5111842, "python": 25252110, "python-repl": 5111842, "q": 5111842, "qml": 5111842, "r": 7420477, "reasonml": 5111842, "rib": 5111842, "roboconf": 5111842, "routeros": 5111842, "rsl": 5111842, "ruby": 11953875, "ruleslanguage": 5111842, "rust": 11232142, "sas": 5111842, "scala": 8320297, "scheme": 5111842, "scilab": 5111842, "scss": 5111842, "shell": 5111842, "smali": 5111842, "smalltalk": 5111842, "sml": 5111842, "sqf": 5111842, "sql": 7083450, "stan": 5111842, "stata": 5111842, "step21": 5111842, "stylus": 5111842, "subunit": 5111842, "swift": 9804990, "taggerscript": 5111842, "tap": 5111842, "tcl": 6114367, "thrift": 5111842, "tp": 5111842, "twig": 5111842, "typescript": 21237337, "vala": 5111842, "vbnet": 5111842, "vbscript": 5111842, "vbscript-html": 5111842, "verilog": 5111842, "vhdl": 5111842, "vim": 6721095, "wasm": 5111842, "wren": 5111842, "x86asm": 5111842, "xl": 5111842, "xml": 12945540, "xquery": 5111842, "yaml": 9578460, "zephir": 5111842}

# hljs id -> known the-stack-dedup directory name, for cases the normalizer
# below would get wrong. Anything not listed here is resolved by normalizing
# (lowercase, spaces/underscores -> '-') and fuzzy-matching against the
# dataset's real live config list fetched in step 3.
ALIASES = {
    'cpp': 'c++', 'csharp': 'c-sharp', 'fsharp': 'f-sharp',
    'objectivec': 'objective-c', 'vim': 'viml', 'bash': 'shell',
    'shell': 'shell', 'elisp': 'emacs-lisp', 'asm': 'assembly',
    'x86asm': 'assembly', 'armasm': 'assembly', 'avrasm': 'assembly',
    'mipsasm': 'assembly', 'make': 'makefile', 'lisp': 'common-lisp',
    'erlang-repl': 'erlang', 'julia-repl': 'julia', 'python-repl': 'python',
    'node-repl': 'javascript', 'clojure-repl': 'clojure',
    'protobuf': 'protocol-buffer', 'latex': 'tex', 'scss': 'scss',
    'objc': 'objective-c', 'objcpp': 'objective-c++',
}

MIN_FILE_BYTES = 200
MAX_FILE_BYTES = 400_000
TRUNCATE_BYTES = 30_000
SKIP_PATH = re.compile(
    r'(^|/)(node_modules|vendor|third_party|thirdparty|external|deps|dist|build|'
    r'target|\.git|testdata|fixtures?|__snapshots__|generated|gen|godeps)(/|$)',
    re.IGNORECASE)
SKIP_NAME = re.compile(
    r'(\.min\.(js|css)$|\.pb\.go$|_pb2\.py$|\.generated\.|package-lock\.json$|'
    r'yarn\.lock$|pnpm-lock\.yaml$|Cargo\.lock$|composer\.lock$|\.d\.ts$)',
    re.IGNORECASE)

# the-stack-dedup is data_dir-driven, not config-driven, so
# get_dataset_config_names() only returns 'default' -- list the actual
# top-level folders under data/ instead.
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFolder

print('Listing the-stack-dedup data/ directories...')
api_ = HfApi()
tree = api_.list_repo_tree('bigcode/the-stack-dedup', repo_type='dataset',
                            path_in_repo='data', recursive=False)
live_configs = sorted(e.path.rsplit('/', 1)[-1] for e in tree
                       if isinstance(e, RepoFolder))
live_set = set(live_configs)
print(f'Found {len(live_configs)} language directories.')
assert live_configs, 'no directories found under data/ -- dataset layout may have changed'

def normalize(name: str) -> str:
    return re.sub(r'[\s_]+', '-', name.strip().lower())

resolved: dict[str, str] = {}
unmapped: list[str] = []
for lang in BYTE_BUDGET:
    candidate = ALIASES.get(lang, normalize(lang))
    if candidate in live_set:
        resolved[lang] = candidate
        continue
    close = get_close_matches(candidate, live_configs, n=1, cutoff=0.9)
    if close:
        resolved[lang] = close[0]
    else:
        unmapped.append(lang)

print(f'Resolved {len(resolved)}/{len(BYTE_BUDGET)} languages against the-stack-dedup.')
print(f'Unmapped ({len(unmapped)}), leave these to the local GitHub fetcher:')
print(', '.join(sorted(unmapped)))
assert resolved, 'nothing resolved -- fix ALIASES/normalize() before continuing, do not proceed to fetch/zip/upload'

from datasets import load_dataset
from tqdm.auto import tqdm

def harvest_language(lang: str, stack_name: str, budget: int) -> tuple[int, int]:
    out_dir = OUT / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    written = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file())
    files = len(list(out_dir.iterdir()))
    if written >= budget:
        return written, files

    ds = load_dataset('bigcode/the-stack-dedup', data_dir=f'data/{stack_name}',
                       split='train', streaming=True)
    for row in ds:
        if written >= budget:
            break
        path = row.get('max_stars_repo_path', '') or ''
        name = Path(path).name
        if SKIP_PATH.search(path) or SKIP_NAME.search(name):
            continue
        text = row.get('content', '') or ''
        size = row.get('size', len(text))
        if size < MIN_FILE_BYTES or size > MAX_FILE_BYTES:
            continue
        if '\x00' in text or len(text.strip()) < MIN_FILE_BYTES:
            continue
        if len(text) > TRUNCATE_BYTES:
            cut = text.rfind('\n', 0, TRUNCATE_BYTES)
            text = text[:cut if cut > MIN_FILE_BYTES else TRUNCATE_BYTES]
        repo = (row.get('max_stars_repo_name', '') or 'unknown').replace('/', '__')
        flat = f'{repo}__{path.replace("/", "_")}'
        target = out_dir / flat
        if target.exists():
            continue
        try:
            target.write_text(text, encoding='utf-8')
        except (OSError, UnicodeEncodeError):
            continue
        written += len(text.encode('utf-8'))
        files += 1
    return written, files

from concurrent.futures import ThreadPoolExecutor, as_completed

# I/O-bound (network streaming), not CPU-bound, so a wide thread pool is safe
# and this is the actual lever for wall-clock time -- HF's CDN has no
# per-request rate limit the way GitHub's API does. 16 is a reasonable
# default for Colab's shared vCPUs; raise it if Colab's network looks
# underused (check with `!cat /proc/net/dev` or just watch the tqdm rate).
WORKERS = 16

results = {}
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(harvest_language, lang, stack_name, BYTE_BUDGET[lang]): (lang, stack_name)
               for lang, stack_name in resolved.items()}
    for fut in tqdm(as_completed(futures), total=len(futures), desc='languages'):
        lang, stack_name = futures[fut]
        budget = BYTE_BUDGET[lang]
        try:
            written, files = fut.result()
        except Exception as exc:
            tqdm.write(f'  [FAIL] {lang:15s} <- {stack_name:20s} {type(exc).__name__}: {exc}')
            continue
        results[lang] = {'stack_name': stack_name, 'written': written,
                          'files': files, 'budget': budget}
        tqdm.write(f'  [DONE] {lang:15s} <- {stack_name:20s} '
                   f'{written/1e6:7.2f} MB  {files:5d} files (target: {budget/1e6:.2f} MB)')

total = sum(r['written'] for r in results.values())
print(f'\nTotal raw corpus: {total/1e6:.1f} MB across {len(results)} languages')

import shutil
from huggingface_hub import HfApi, create_repo

ZIP_PATH = '/content/lex_stack_corpus'
shutil.make_archive(ZIP_PATH, 'zip', OUT)
print(f'Zipped to {ZIP_PATH}.zip ({Path(ZIP_PATH + ".zip").stat().st_size/1e6:.1f} MB)')

REPO_ID = 'anudit/lex-large-stack-corpus'  # change if you want a different name
create_repo(REPO_ID, repo_type='dataset', exist_ok=True)
api = HfApi()
api.upload_file(
    path_or_fileobj=f'{ZIP_PATH}.zip',
    path_in_repo='corpus_raw.zip',
    repo_id=REPO_ID,
    repo_type='dataset',
)
print(f'Uploaded. On the Mac run:')
print(f'  huggingface-cli download {REPO_ID} corpus_raw.zip --repo-type dataset '
      f'--local-dir train_large/_stack_download')
print(f'  unzip -n train_large/_stack_download/corpus_raw.zip -d train_large/corpus/raw')
print('(unzip -n will not overwrite files fetch_corpus.py already has, since '
      'filenames are namespaced by repo already)')
