"""Pinned Highlight.js coverage, hybrid teachers, weights, and token budgets."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / 'language_manifest.json'
if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f'{MANIFEST_PATH} is missing; run `bun install && bun run sync-languages` in {HERE}')

MANIFEST = json.loads(MANIFEST_PATH.read_text())
LANGUAGE_META = {item['id']: item for item in MANIFEST['languages']}
TARGET_LANGUAGES: tuple[str, ...] = tuple(sorted(LANGUAGE_META))
TRAIN_EXCLUDED_LANGUAGES: frozenset[str] = frozenset()
TRAIN_LANGUAGES = TARGET_LANGUAGES

# Approximate GitHub popularity shares carried over from the measured compact
# benchmark. Canonical Highlight.js ids are used (for example makefile and
# objectivec); the remaining grammars receive the coverage prior below.
POPULARITY: dict[str, float] = {
    'javascript': 17.0, 'python': 15.5, 'typescript': 10.5, 'java': 8.0,
    'csharp': 5.0, 'cpp': 4.6, 'php': 4.2, 'bash': 4.0, 'c': 3.6,
    'go': 3.4, 'xml': 3.2, 'ruby': 2.6, 'css': 2.5, 'markdown': 2.4,
    'rust': 2.2, 'kotlin': 1.7, 'swift': 1.5, 'yaml': 1.4, 'json': 1.3,
    'dart': 1.2, 'scala': 0.9, 'powershell': 0.8, 'makefile': 0.8,
    'lua': 0.7, 'perl': 0.6, 'r': 0.6, 'cmake': 0.6, 'groovy': 0.5,
    'sql': 0.5, 'dockerfile': 0.5, 'vim': 0.4, 'glsl': 0.3,
    'gherkin': 0.3, 'awk': 0.25, 'tcl': 0.25,
}

POPULARITY_MIX = 0.75
UNIFORM_MIX = 0.25
DEFAULT_TOTAL_TOKENS = 160_000_000
MIN_TOKENS_PER_LANG = 300_000
BUDGET_EXPONENT = 0.5


def weights() -> dict[str, float]:
    """75% usage proxy plus 25% uniform coverage, normalized over 193 grammars."""
    natural = {lang: POPULARITY.get(lang, 0.05) for lang in TARGET_LANGUAGES}
    total = sum(natural.values())
    n = len(TARGET_LANGUAGES)
    return {
        lang: POPULARITY_MIX * natural[lang] / total + UNIFORM_MIX / n
        for lang in TARGET_LANGUAGES
    }


def token_budgets(total_tokens: int = DEFAULT_TOTAL_TOKENS) -> dict[str, int]:
    """Exact-size water-filled allocation with a hard per-language floor."""
    floor_total = MIN_TOKENS_PER_LANG * len(TARGET_LANGUAGES)
    if total_tokens < floor_total:
        raise ValueError(
            f'{total_tokens:,} tokens cannot provide the {MIN_TOKENS_PER_LANG:,} '
            f'floor for {len(TARGET_LANGUAGES)} languages ({floor_total:,} required)')
    w = weights()
    shaped = {lang: value ** BUDGET_EXPONENT for lang, value in w.items()}
    remainder = total_tokens - floor_total
    denom = sum(shaped.values())
    budgets = {
        lang: MIN_TOKENS_PER_LANG + int(remainder * shaped[lang] / denom)
        for lang in TARGET_LANGUAGES
    }
    # Make integer rounding exact and deterministic.
    missing = total_tokens - sum(budgets.values())
    for lang in sorted(TARGET_LANGUAGES, key=lambda key: (-shaped[key], key))[:missing]:
        budgets[lang] += 1
    return budgets


def benchmark_family_language(family: str) -> str:
    aliases = {
        'shell': 'bash', 'shellscript': 'bash', 'make': 'makefile',
        'objective-c': 'objectivec', 'objc': 'objectivec', 'assembly': 'x86asm',
        'html': 'xml', 'batchfile': 'dos', 'protobuf': 'protobuf',
    }
    return aliases.get(family, family)


def teacher(lang: str) -> tuple[str, str]:
    item = LANGUAGE_META[lang]['teacher']
    return item['engine'], item['language']


def shiki_id(lang: str) -> str:
    engine, language = teacher(lang)
    return language if engine == 'shiki' else lang


def extensions(lang: str) -> tuple[str, ...]:
    return tuple(LANGUAGE_META[lang].get('extensions', ()))
