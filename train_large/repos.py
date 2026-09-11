"""Repository manifest, seeded from the compact corpus and extended by discovery."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'train'
_spec = importlib.util.spec_from_file_location('_lex_compact_repos', BASE_DIR / 'repos.py')
assert _spec and _spec.loader
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)

ALIASES = {
    'shell': 'bash', 'make': 'makefile', 'objc': 'objectivec', 'viml': 'vim',
    'asm': 'x86asm', 'bat': 'dos', 'html': 'xml', 'elisp': 'lisp',
    'plsql': 'pgsql', 'xslt': 'xml',
}

REPOS: dict[str, list[str]] = {}
for language, values in _base.REPOS.items():
    target = ALIASES.get(language, language)
    REPOS.setdefault(target, []).extend(values)

generated = HERE / 'repos.json'
if generated.exists():
    for language, values in json.loads(generated.read_text()).items():
        REPOS.setdefault(language, []).extend(values)

for language, values in list(REPOS.items()):
    REPOS[language] = list(dict.fromkeys(values))

FILENAME_MATCH = dict(getattr(_base, 'FILENAME_MATCH', {}))
FILENAME_MATCH.update({
    'dockerfile': ('dockerfile', 'containerfile'),
    'makefile': ('makefile', 'gnumakefile'),
    'plaintext': ('readme', 'license', 'authors', 'changelog'),
    'profile': ('.profile', '.bash_profile', '.zprofile'),
})

