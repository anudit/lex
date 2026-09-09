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

# Approximate GitHub Innovation Graph top-25 pusher shares, normalized below.
TOP25_PUSHER_SHARE: dict[str, float] = {
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
    'lua': 0.7,
    'perl': 0.6,
    'r': 0.6,
}

# Sugar High's full language list -- the coverage claim.
SUGAR_HIGH_29: tuple[str, ...] = (
    'javascript', 'typescript', 'css', 'python', 'c', 'go', 'java', 'rust',
    'json', 'diff', 'shell', 'cpp', 'csharp', 'sql', 'html', 'yaml', 'markdown',
    'plaintext', 'ruby', 'kotlin', 'swift', 'php', 'toml', 'powershell',
    'dockerfile', 'graphql', 'hcl', 'zig', 'lua',
)

# Everything the model is trained and evaluated on.
TARGET_LANGUAGES: tuple[str, ...] = tuple(sorted(
    set(TOP25_PUSHER_SHARE) | set(SUGAR_HIGH_29)
))

# Weight floor for languages outside the top 25, as a share point. Keeps the tail
# (zig, hcl, graphql, toml, diff, ...) in the corpus and in the metric without
# letting it move the popularity-weighted headline.
TAIL_SHARE = 0.25


def weights() -> dict[str, float]:
    """Normalized evaluation/budget weight per target language (sums to 1.0)."""
    raw = {lang: TOP25_PUSHER_SHARE.get(lang, TAIL_SHARE) for lang in TARGET_LANGUAGES}
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
