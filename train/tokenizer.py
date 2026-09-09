"""
Universal Code Tokenizer for Neural Lexer.
Matches the language-agnostic fast CPU scanner semantics of gpu-lexer/sugar-high
with expanded hash capacity: 512 primary buckets against gpu-lexer's 256, which is
where keyword/identifier collisions come from, and 128 secondary to match it.
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
        'trans_prev', 'trans_next', 'sym_prev', 'sym_next', 'text'
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

def tokens_to_arrays(tokens: list[Token]):
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
