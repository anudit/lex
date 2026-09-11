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


def tokens_to_arrays(tokens: list[Token]) -> dict[str, np.ndarray]:
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

