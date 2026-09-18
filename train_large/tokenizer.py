"""Large-model tokenizer with wider hashes and zero-parameter structure hints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'train'
_spec = importlib.util.spec_from_file_location('_lex_compact_tokenizer', BASE_DIR / 'tokenizer.py')
assert _spec and _spec.loader
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)

Token = _base.Token
TRANSITIONS = _base.TRANSITIONS
is_space = _base.is_space
is_word = _base.is_word
char_bucket = _base.char_bucket
len_bucket = _base.len_bucket
symbol_hash = _base.symbol_hash


def _word_hash(text: str, seed: int, multiplier: int, modulus: int) -> int:
    value = seed
    for char in text:
        value = ((value ^ char_bucket(ord(char))) * multiplier) & 0xFFFFFFFF
    return (value ^ (value >> 16)) % modulus


def tokenize(text: str) -> list[Token]:
    tokens = _base.tokenize(text)
    for token in tokens:
        if token.kind == 0:
            token.hash1 = _word_hash(token.text, 2166136261, 16777619, 1024)
            token.hash2 = _word_hash(token.text, 2654435769, 2246822519, 256)
    return tokens


def _tokens_to_arrays_v1(tokens: list[Token]) -> dict[str, np.ndarray]:
    arrays = _base.tokens_to_arrays(tokens)
    count = len(tokens)
    structural = {
        name: np.zeros(count, dtype=np.int64)
        for name in (
            'paren_depth', 'brace_depth', 'bracket_depth', 'line_pos',
            'indent_bucket', 'quote_state',
        )
    }
    paren = brace = bracket = line_pos = indent = 0
    quote = 0
    escaped = False
    at_line_start = True

    for i, token in enumerate(tokens):
        structural['paren_depth'][i] = min(paren, 7)
        structural['brace_depth'][i] = min(brace, 7)
        structural['bracket_depth'][i] = min(bracket, 7)
        structural['line_pos'][i] = min(line_pos, 7)
        structural['indent_bucket'][i] = min(indent.bit_length(), 7)
        structural['quote_state'][i] = quote

        if token.kind == 2:
            line_pos = indent = 0
            at_line_start = True
            escaped = False
            continue
        if token.kind == 1:
            if at_line_start:
                indent += sum(4 if char == '\t' else 1 for char in token.text)
            continue

        text = token.text
        if token.kind == 3 and text in ("'", '"', '`') and not escaped:
            state = {"'": 1, '"': 2, '`': 3}[text]
            quote = 0 if quote == state else (state if quote == 0 else quote)
        elif quote == 0 and token.kind == 3:
            if text == '(':
                paren += 1
            elif text == ')':
                paren = max(0, paren - 1)
            elif text == '{':
                brace += 1
            elif text == '}':
                brace = max(0, brace - 1)
            elif text == '[':
                bracket += 1
            elif text == ']':
                bracket = max(0, bracket - 1)

        escaped = token.kind == 3 and text == '\\' and not escaped
        at_line_start = False
        line_pos += 1

    arrays.update(structural)
    return arrays


LINE_FIRST_SYMBOLS = _base.LINE_FIRST_SYMBOLS
_LINE_FIRST = {ord(ch): i + 1 for i, ch in enumerate(LINE_FIRST_SYMBOLS)}


def tokenize_v2(text: str) -> list[Token]:
    """Whitespace-free large tokens with wide hashes and rich structure.

    This mirrors lex-lite v2's gap semantics while retaining lex-large's
    independent bracket depths, line position and quote state.
    """
    raw = tokenize(text)
    out: list[Token] = []
    spaces = False
    newlines = 1
    indent_width = 0
    indent_tab = False
    line_first = -1
    paren = brace = bracket = line_pos = 0
    quote = 0
    escaped = False

    for token in raw:
        if token.kind == 2:
            newlines += 1
            indent_width = 0
            indent_tab = False
            line_first = -1
            line_pos = 0
            escaped = False
            continue
        if token.kind == 1:
            spaces = True
            if line_first < 0:
                for char in token.text:
                    indent_width += 4 if char == '\t' else 1
                    indent_tab = indent_tab or char == '\t'
            continue

        if line_first < 0:
            line_first = 0 if token.kind == 0 else _LINE_FIRST.get(token.first_char, 0)
        token.gap_prev = 3 if newlines >= 2 else 2 if newlines == 1 else 1 if spaces else 0
        token.gap_next = 2
        token.indent = _base.indent_bucket(indent_width)
        token.line_first = line_first
        token.paren_depth = min(paren, 7)
        token.brace_depth = min(brace, 7)
        token.bracket_depth = min(bracket, 7)
        token.line_pos = min(line_pos, 7)
        token.quote_state = quote
        if indent_tab:
            token.flags |= 32

        value = token.text if token.kind == 3 else ''
        if value in ("'", '"', '`') and not escaped:
            state = {"'": 1, '"': 2, '`': 3}[value]
            quote = 0 if quote == state else (state if quote == 0 else quote)
        elif quote == 0:
            if value == '(':
                paren += 1
            elif value == ')':
                paren = max(0, paren - 1)
            elif value == '{':
                brace += 1
            elif value == '}':
                brace = max(0, brace - 1)
            elif value == '[':
                bracket += 1
            elif value == ']':
                bracket = max(0, bracket - 1)
        escaped = value == '\\' and not escaped

        token.trans_prev = token.trans_next = 0
        token.sym_prev = token.sym_next = 0
        if out:
            previous = out[-1]
            previous.gap_next = token.gap_prev
            if token.gap_prev == 0:
                transition = TRANSITIONS.get((previous.last_char, token.first_char), 0)
                token.trans_prev = transition
                previous.trans_next = transition
            if previous.kind == 3 or token.kind == 3:
                pair = symbol_hash(previous.last_char, token.first_char)
                token.sym_prev = pair
                previous.sym_next = pair
        out.append(token)
        spaces = False
        newlines = 0
        line_pos += 1
    return out


V2_FIELDS = (
    'gap_prev', 'gap_next', 'indent', 'line_first', 'brace_depth',
    'paren_depth', 'bracket_depth', 'line_pos', 'quote_state',
)


def tokenize_version(text: str, version: int) -> list[Token]:
    if version == 1:
        return tokenize(text)
    if version == 2:
        return tokenize_v2(text)
    raise ValueError(f'unknown feature version {version}')


def tokens_to_arrays(tokens: list[Token], version: int = 1) -> dict[str, np.ndarray]:
    if version == 1:
        return _tokens_to_arrays_v1(tokens)
    if version != 2:
        raise ValueError(f'unknown feature version {version}')
    arrays = _base.tokens_to_arrays(tokens)
    for name in V2_FIELDS:
        arrays[name] = np.fromiter((getattr(token, name) for token in tokens),
                                   dtype=np.int64, count=len(tokens))
    return arrays
