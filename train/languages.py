"""
Target language set and evaluation weights.

Two constituencies have to be satisfied at once:

  * Sugar High's 29 languages -- the coverage claim we inherited from gpu-lexer.
  * The GitHub Innovation Graph top 25 by pusher count -- the set the headline
    "popularity-weighted agreement with Shiki" benchmark actually scores, where
    an unsupported language scores zero.

TARGET_LANGUAGES is the union, so both are covered by one model.

WEIGHTS are pusher-count shares used for (a) corpus budget allocation and
(b) the weighted eval metric. The values below are approximate Innovation Graph
shares and are deliberately kept in one editable table: swap in the exact
2026-Q1 pusher counts and every downstream budget and score follows.
Languages outside the top 25 carry a small floor weight so they still get
trained and still get measured, without distorting the headline number.
"""

from __future__ import annotations

# Approximate GitHub Innovation Graph top-50 pusher shares, normalized below.
TOP50_PUSHER_SHARE: dict[str, float] = {
    'javascript': 17.0,
    'python': 15.5,
    'typescript': 10.5,
    'java': 8.0,
    'csharp': 5.0,
    'cpp': 4.6,
    'php': 4.2,
    'shell': 4.0,
    'c': 3.6,
    'go': 3.4,
    'html': 3.2,
    'ruby': 2.6,
    'css': 2.5,
    'markdown': 2.4,
    'rust': 2.2,
    'kotlin': 1.7,
    'swift': 1.5,
    'yaml': 1.4,
    'json': 1.3,
    'dart': 1.2,
    'scala': 0.9,
    'powershell': 0.8,
    'make': 0.8,
    'lua': 0.7,
    'objc': 0.7,
    'perl': 0.6,
    'r': 0.6,
    'bat': 0.6,
    'cmake': 0.6,
    'groovy': 0.5,
    'sql': 0.5,
    'asm': 0.5,
    'dockerfile': 0.5,
    'viml': 0.4,
    'hcl': 0.4,
    'objcpp': 0.4,
    'elisp': 0.3,
    'plsql': 0.3,
    'hlsl': 0.3,
    'glsl': 0.3,
    'gherkin': 0.3,
    'xslt': 0.3,
    'starlark': 0.25,
    'awk': 0.25,
    'tcl': 0.25,
    'hack': 0.25,
    'shaderlab': 0.25,
    'smarty': 0.25,
    'm4': 0.25,
    'lex': 0.25,
    'yacc': 0.25,
    'qmake': 0.25,
}

# Top 25 pusher shares (preserved for headroom and baseline comparisons).
TOP25_PUSHER_SHARE: dict[str, float] = {
    k: v for k, v in list(TOP50_PUSHER_SHARE.items())[:25]
}

# Sugar High's full language list -- the coverage claim.
SUGAR_HIGH_29: tuple[str, ...] = (
    'javascript', 'typescript', 'css', 'python', 'c', 'go', 'java', 'rust',
    'json', 'diff', 'shell', 'cpp', 'csharp', 'sql', 'html', 'yaml', 'markdown',
    'plaintext', 'ruby', 'kotlin', 'swift', 'php', 'toml', 'powershell',
    'dockerfile', 'graphql', 'hcl', 'zig', 'lua',
)

# Everything the model is trained and evaluated on (50 top languages + Sugar High).
TARGET_LANGUAGES: tuple[str, ...] = tuple(sorted(
    set(TOP50_PUSHER_SHARE) | set(SUGAR_HIGH_29)
))

# Low-popularity languages excluded from the primary training recipe. Keep them
# in TARGET_LANGUAGES so existing dataset ids stay stable and the frozen
# 57-language coverage evaluation continues to expose regressions.
TRAIN_EXCLUDED_LANGUAGES: frozenset[str] = frozenset({
    'gherkin', 'groovy', 'm4', 'qmake', 'starlark',
})
TRAIN_LANGUAGES: tuple[str, ...] = tuple(
    lang for lang in TARGET_LANGUAGES if lang not in TRAIN_EXCLUDED_LANGUAGES
)

# gpu-lexer's verification benchmark uses family names from Linguist rather
# than this corpus's directory names. Procfile is lexed as shell here, so its
# pusher weight is intentionally added to shell's sampling probability.
BENCHMARK_FAMILY_TO_LANGUAGE: dict[str, str] = {
    'makefile': 'make',
    'batchfile': 'bat',
    'plpgsql': 'sql',
    'objective-c': 'objc',
    'assembly': 'asm',
    'procfile': 'shell',
}


def benchmark_family_language(family: str) -> str:
    return BENCHMARK_FAMILY_TO_LANGUAGE.get(family, family)

# Weight floor for languages outside the top 50, as a share point.
TAIL_SHARE = 0.25


def weights() -> dict[str, float]:
    """Normalized evaluation/budget weight per target language (sums to 1.0)."""
    raw = {lang: TOP50_PUSHER_SHARE.get(lang, TAIL_SHARE) for lang in TARGET_LANGUAGES}
    total = sum(raw.values())
    return {lang: w / total for lang, w in raw.items()}


# Corpus budgets are flattened relative to the raw weights: allocating strictly
# in proportion would give Perl and R a few thousand tokens while JavaScript took
# a sixth of the corpus, and a class the model never sees cannot be learned at
# any weight. The exponent trades headline-weight alignment against tail
# coverage; 0.5 keeps JavaScript ~5x Perl instead of ~28x.
BUDGET_EXPONENT = 0.5
MIN_TOKENS_PER_LANG = 400_000


def token_budgets(total_tokens: int) -> dict[str, int]:
    """Tokens to draw per language for a corpus of `total_tokens`."""
    w = weights()
    flat = {lang: v ** BUDGET_EXPONENT for lang, v in w.items()}
    scale = total_tokens / sum(flat.values())
    budgets = {lang: max(MIN_TOKENS_PER_LANG, int(v * scale)) for lang, v in flat.items()}
    return budgets


# Corpus directory name -> Shiki language id. Only entries that differ are listed.
SHIKI_ID: dict[str, str] = {
    'shell': 'shellscript',
    'plaintext': 'txt',
    'make': 'make',
    'bat': 'bat',
    'elisp': 'elisp',
    'viml': 'viml',
    'objc': 'objc',
    'objcpp': 'objective-cpp',
    'tsql': 'sql',
    'plsql': 'plsql',
    'asm': 'asm',
    'cmake': 'cmake',
    'groovy': 'groovy',
    'hack': 'hack',
    'hlsl': 'hlsl',
    'glsl': 'glsl',
    'shaderlab': 'shaderlab',
    'gherkin': 'gherkin',
    'tcl': 'tcl',
    'awk': 'awk',
    'starlark': 'python',
    'smarty': 'html',
    'xslt': 'xml',
    'm4': 'shellscript',
    'lex': 'c',
    'yacc': 'c',
    'qmake': 'make',
}


def shiki_id(lang: str) -> str:
    return SHIKI_ID.get(lang, lang)


if __name__ == '__main__':
    w = weights()
    b = token_budgets(24_000_000)
    print(f'{len(TARGET_LANGUAGES)} target languages '
          f'({len(SUGAR_HIGH_29)} sugar-high + {len(TOP25_PUSHER_SHARE)} top-25)')
    print(f'{"language":14s} {"eval weight":>12s} {"token budget":>13s}')
    for lang in sorted(TARGET_LANGUAGES, key=lambda l: -w[l]):
        top = '*' if lang in TOP25_PUSHER_SHARE else ' '
        print(f'{top}{lang:13s} {w[lang]*100:11.2f}% {b[lang]:13,d}')
    print(f'{"":14s} {sum(w.values())*100:11.2f}% {sum(b.values()):13,d}')
