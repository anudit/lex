"""
Language-agnostic CPU pre-tokenizer.

`tokenize_v2` produces lex-lite's input: word and symbol tokens only, with
whitespace folded into per-token fields (see the section below). `tokenize` is
the underlying scanner -- runs of word / space / newline / symbol with length,
edge-character, hash, flag and transition features -- and is also the input
layout lex-large trains on. lex/src/tokenizer.js mirrors `tokenize_v2` exactly.
"""

import numpy as np

# Transition patterns table (_A in gpu-lexer)
TRANSITIONS = {
    (47, 47): 1,   # //
    (47, 42): 2,   # /*
    (42, 47): 3,   # */
    (45, 45): 4,   # --
    (61, 62): 5,   # =>
    (58, 58): 6,   # ::
    (60, 47): 7,   # </
    (123, 123): 8, # {{
    (36, 123): 9,  # ${
    (125, 125): 10,# }}
    (45, 62): 11,  # ->
    (63, 63): 12,  # ??
    (63, 46): 13,  # ?.
    (60, 62): 14,  # <>
}

def is_space(c: int) -> bool:
    return c == 9 or c == 11 or c == 12 or c == 32

def is_word(c: int) -> bool:
    return c > 127 or c == 95 or (48 <= c <= 57) or (65 <= c <= 90) or (97 <= c <= 122)

def char_bucket(c: int) -> int:
    return 95 if c > 127 else (c & 127)

def len_bucket(length: int) -> int:
    if length <= 1:
        return 0
    return min(7, length.bit_length() - 1)

def symbol_hash(a: int, b: int) -> int:
    val = (((a + 1) * 131) & 0xFFFFFFFF) ^ b
    return 1 + (val % 31)

class Token:
    __slots__ = (
        'kind', 'start', 'end', 'first_char', 'last_char',
        'len_bucket', 'hash1', 'hash2', 'flags',
        'trans_prev', 'trans_next', 'sym_prev', 'sym_next', 'text',
        # Feature-version-2 fields; unused (and unset) in version 1.
        'gap_prev', 'gap_next', 'indent', 'line_first', 'brace_depth', 'paren_depth',
        # Extended structure fields used by lex-large v2.
        'bracket_depth', 'line_pos', 'indent_bucket', 'quote_state',
    )
    def __init__(self, kind, start, end, first_char, last_char, len_b, h1, h2, flags, text):
        self.kind = kind
        self.start = start
        self.end = end
        self.first_char = first_char
        self.last_char = last_char
        self.len_bucket = len_b
        self.hash1 = h1
        self.hash2 = h2
        self.flags = flags
        self.trans_prev = 0
        self.trans_next = 0
        self.sym_prev = 0
        self.sym_next = 0
        self.text = text

def tokenize(text: str) -> list[Token]:
    tokens = []
    n = len(text)
    pos = 0
    is_line_start = True

    while pos < n:
        start = pos
        c = ord(text[pos])

        if c == 10 or c == 13:
            kind = 2 # newline
        elif is_space(c):
            kind = 1 # space
        elif is_word(c):
            kind = 0 # word
        else:
            kind = 3 # symbol

        flags = 16 if is_line_start else 0
        first_c = char_bucket(c)
        last_c = first_c
        h1 = 0
        h2 = 0

        if kind == 0:
            h1 = 2166136261
            h2 = 2654435769
            while pos < n and is_word(ord(text[pos])):
                cc = ord(text[pos])
                cb = char_bucket(cc)
                last_c = cb
                h1 = (((h1 ^ cb) * 16777619) & 0xFFFFFFFF)
                h2 = (((h2 ^ cb) * 2246822519) & 0xFFFFFFFF)
                if 97 <= cb <= 122:
                    flags |= 1
                elif 65 <= cb <= 90:
                    flags |= 2
                elif 48 <= cb <= 57:
                    flags |= 4
                elif cb == 95:
                    flags |= 8
                pos += 1
            h1 = (h1 ^ (h1 >> 16)) & 511
            h2 = (h2 ^ (h2 >> 16)) & 127
            if (flags & 2) and not (flags & 1):
                flags |= 128
        elif kind == 1:
            while pos < n and is_space(ord(text[pos])):
                cc = ord(text[pos])
                last_c = char_bucket(cc)
                if cc == 9:
                    flags |= 32
                pos += 1
        elif kind == 2:
            pos += 1
            if c == 13 and pos < n and ord(text[pos]) == 10:
                last_c = 10
                pos += 1
        else:
            if c == 92:
                flags |= 64
            pos += 1

        end = pos
        length = end - start
        len_b = len_bucket(length)
        tok_text = text[start:end]

        tok = Token(kind, start, end, first_c, last_c, len_b, h1, h2, flags, tok_text)
        tokens.append(tok)

        if kind == 2:
            is_line_start = True
        elif kind != 1:
            is_line_start = False

    for i in range(1, len(tokens)):
        prev_tok = tokens[i - 1]
        curr_tok = tokens[i]
        trans = TRANSITIONS.get((prev_tok.last_char, curr_tok.first_char), 0)
        curr_tok.trans_prev = trans
        prev_tok.trans_next = trans

        if prev_tok.kind == 3 or curr_tok.kind == 3:
            sym_h = symbol_hash(prev_tok.last_char, curr_tok.first_char)
            curr_tok.sym_prev = sym_h
            prev_tok.sym_next = sym_h

    return tokens

# ---------------------------------------------------------------------------
# lex-lite features (feature_version=2)
#
# Whitespace and newline runs leave the sequence, so every depthwise-conv tap
# and recurrent step is spent on a real token. Their information survives as
# per-token fields:
#
#   gap_prev / gap_next  what separates this token from its non-blank neighbour:
#                        0 nothing, 1 horizontal space, 2 one newline, 3 blank line
#   indent               leading-whitespace width of the token's line, bucketed
#   line_first           first non-blank character of the token's line (0 = word)
#   brace_depth          `{}` nesting, 0..3+
#   paren_depth          `()` + `[]` nesting, 0..3+
#
# Transition ids still require the two characters to be physically adjacent;
# symbol-pair hashes pair nearest non-blank neighbours. hash1 uses 256 buckets.
# ---------------------------------------------------------------------------

LINE_FIRST_SYMBOLS = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'
_LINE_FIRST = {ord(ch): i + 1 for i, ch in enumerate(LINE_FIRST_SYMBOLS)}
assert len(_LINE_FIRST) == 31
HASH1_BUCKETS_V2 = 256


def indent_bucket(width: int) -> int:
    if width <= 2:
        return width
    if width <= 4:
        return 3
    if width <= 16:
        return 4 + (width - 5) // 4
    return 7


def tokenize_v2(text: str) -> list[Token]:
    raw = tokenize(text)
    out: list[Token] = []
    spaces = False
    newlines = 1          # the file start behaves like a line start
    indent_width = 0
    indent_tab = False
    line_first = -1
    braces = 0
    parens = 0
    for tok in raw:
        if tok.kind == 2:
            newlines += 1
            indent_width = 0
            indent_tab = False
            line_first = -1
            continue
        if tok.kind == 1:
            spaces = True
            if line_first < 0:
                for ch in tok.text:
                    indent_width += 4 if ch == '\t' else 1
                    indent_tab = indent_tab or ch == '\t'
            continue
        if line_first < 0:
            line_first = 0 if tok.kind == 0 else _LINE_FIRST.get(tok.first_char, 0)
        tok.hash1 &= HASH1_BUCKETS_V2 - 1
        tok.gap_prev = 3 if newlines >= 2 else 2 if newlines == 1 else 1 if spaces else 0
        tok.gap_next = 2
        tok.indent = indent_bucket(indent_width)
        tok.line_first = line_first
        if indent_tab:
            tok.flags |= 32
        c = tok.first_char if tok.kind == 3 else -1
        if c == 123:
            tok.brace_depth = min(3, braces)
            braces += 1
        elif c == 125:
            braces = max(0, braces - 1)
            tok.brace_depth = min(3, braces)
        else:
            tok.brace_depth = min(3, braces)
        if c in (40, 91):
            tok.paren_depth = min(3, parens)
            parens += 1
        elif c in (41, 93):
            parens = max(0, parens - 1)
            tok.paren_depth = min(3, parens)
        else:
            tok.paren_depth = min(3, parens)
        tok.trans_prev = tok.trans_next = 0
        tok.sym_prev = tok.sym_next = 0
        if out:
            prev = out[-1]
            prev.gap_next = tok.gap_prev
            if tok.gap_prev == 0:
                trans = TRANSITIONS.get((prev.last_char, tok.first_char), 0)
                tok.trans_prev = trans
                prev.trans_next = trans
            if prev.kind == 3 or tok.kind == 3:
                sym_h = symbol_hash(prev.last_char, tok.first_char)
                tok.sym_prev = sym_h
                prev.sym_next = sym_h
        out.append(tok)
        spaces = False
        newlines = 0
    return out


V2_FIELDS = ('gap_prev', 'gap_next', 'indent', 'line_first', 'brace_depth', 'paren_depth')


def tokenize_version(text: str, version: int) -> list[Token]:
    if version == 1:
        return tokenize(text)
    if version == 2:
        return tokenize_v2(text)
    raise ValueError(f'unknown feature version {version}')


def tokens_to_arrays(tokens: list[Token], version: int = 1):
    if version == 2:
        base = tokens_to_arrays(tokens, 1)
        for name in V2_FIELDS:
            base[name] = np.fromiter((getattr(t, name) for t in tokens),
                                     dtype=np.int64, count=len(tokens))
        return base
    T = len(tokens)
    kinds = np.empty(T, dtype=np.int64)
    len_buckets = np.empty(T, dtype=np.int64)
    first_chars = np.empty(T, dtype=np.int64)
    last_chars = np.empty(T, dtype=np.int64)
    hash1s = np.empty(T, dtype=np.int64)
    hash2s = np.empty(T, dtype=np.int64)
    flags = np.empty(T, dtype=np.int64)
    trans_prevs = np.empty(T, dtype=np.int64)
    trans_nexts = np.empty(T, dtype=np.int64)
    sym_prevs = np.empty(T, dtype=np.int64)
    sym_nexts = np.empty(T, dtype=np.int64)

    for i, t in enumerate(tokens):
        kinds[i] = t.kind
        len_buckets[i] = t.len_bucket
        first_chars[i] = t.first_char
        last_chars[i] = t.last_char
        hash1s[i] = t.hash1
        hash2s[i] = t.hash2
        flags[i] = t.flags
        trans_prevs[i] = t.trans_prev
        trans_nexts[i] = t.trans_next
        sym_prevs[i] = t.sym_prev
        sym_nexts[i] = t.sym_next

    return {
        'kind': kinds,
        'len_bucket': len_buckets,
        'first_char': first_chars,
        'last_char': last_chars,
        'hash1': hash1s,
        'hash2': hash2s,
        'flags': flags,
        'trans_prev': trans_prevs,
        'trans_next': trans_nexts,
        'sym_prev': sym_prevs,
        'sym_next': sym_nexts,
    }
