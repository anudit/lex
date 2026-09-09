"""
Projection from sugar-high's token vocabulary onto gpu-lexer's 9 syntax classes.

sugar-high emits 11 types (identifier, keyword, string, class, property, entity,
jsxliterals, sign, comment, break, space); gpu-lexer's public API emits 9
(plain, comment, string, number, keyword, type, function, constant, operator).
The two spaces are not the same, so the mapping below is an explicit, testable
projection rather than an identity:

  comment                  -> comment
  string | jsxliterals     -> string
  sign                     -> operator
  entity                   -> function
  keyword                  -> constant  if the text is a boolean/null literal
                              keyword   otherwise
  class                    -> number    if the text is a numeric literal
                              constant  if the text is SCREAMING_CASE
                              type      otherwise
  identifier | property    -> function  if immediately applied (next char is '(')
                              constant  if SCREAMING_CASE
                              number    if a numeric literal
                              plain     otherwise
  space | break            -> masked (-100), never contributes to the loss

sugar-high folds numeric literals into `class`, which is why the numeric test
runs before the type test rather than relying on the token type alone.
"""

from __future__ import annotations

import re

CLASS_NAMES = (
    'plain', 'comment', 'string', 'number', 'keyword',
    'type', 'function', 'constant', 'operator',
)
PLAIN, COMMENT, STRING, NUMBER, KEYWORD, TYPE, FUNCTION, CONSTANT, OPERATOR = range(9)
MASK = -100
NUM_CLASSES = 9

# Covers decimal, float, exponent, hex, binary, octal, and the digit separators
# and width suffixes used by Rust, C++, Zig, Java, C# and JavaScript.
_NUMERIC = re.compile(
    r'^[+-]?('
    r'0[xX][0-9a-fA-F_]+|'
    r'0[bB][01_]+|'
    r'0[oO][0-7_]+|'
    r'\d[\d_]*(\.[\d_]*)?([eE][+-]?\d+)?'
    r')'
    r'[uUlLfFdDnN]*'
    r'(i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize|f32|f64)?$'
)

_BOOL_NULL = frozenset({
    'true', 'false', 'null', 'nil', 'none', 'undefined', 'nan', 'inf',
    'nullptr', 'void 0', 'yes', 'no', 'on', 'off', 'null_t',
})

_SCREAMING = re.compile(r'^[A-Z][A-Z0-9_]*$')


def is_numeric(text: str) -> bool:
    return bool(text) and bool(_NUMERIC.match(text))


def is_screaming(text: str) -> bool:
    return len(text) > 1 and bool(_SCREAMING.match(text)) and any(c.isalpha() for c in text)


def project(sh_type: str, text: str, applied: bool) -> int:
    """Map one sugar-high token to a gpu-lexer class id.

    `applied` is True when the very next non-space character in the source is '(',
    which is how both gpu-lexer and every editor theme distinguish a call site
    from a bare identifier.
    """
    if sh_type == 'comment':
        return COMMENT
    if sh_type in ('string', 'jsxliterals'):
        return STRING
    if sh_type == 'sign':
        return OPERATOR
    if sh_type == 'entity':
        return FUNCTION
    if sh_type == 'keyword':
        return CONSTANT if text.lower() in _BOOL_NULL else KEYWORD
    if sh_type == 'class':
        if is_numeric(text):
            return NUMBER
        if is_screaming(text):
            return CONSTANT
        return TYPE
    if sh_type in ('identifier', 'property'):
        if applied:
            return FUNCTION
        if text.lower() in _BOOL_NULL:
            return CONSTANT
        if is_numeric(text):
            return NUMBER
        if is_screaming(text):
            return CONSTANT
        return PLAIN
    return PLAIN


# Single-character codes emitted by label_worker.mjs.
RLE_TYPES = {
    'i': 'identifier', 'k': 'keyword', 's': 'string', 'c': 'class',
    'p': 'property', 'e': 'entity', 'j': 'jsxliterals', 'g': 'sign',
    'm': 'comment', 'b': 'break', 'w': 'space',
}

_RLE = re.compile(r'([ikscpejgmbw])(\d+)')


def decode_rle(rle: str) -> list[tuple[int, int, str]]:
    """Expand "k3w1i5" into [(start, end, sugar-high type), ...]."""
    spans = []
    pos = 0
    for code, count in _RLE.findall(rle):
        n = int(count)
        spans.append((pos, pos + n, RLE_TYPES[code]))
        pos += n
    return spans


def char_classes(code: str, rle: str) -> bytearray:
    """Per-character class id for the whole file; 255 marks whitespace/breaks."""
    out = bytearray([255]) * 0
    out = bytearray(255 for _ in range(len(code)))
    for start, end, sh_type in decode_rle(rle):
        if sh_type in ('space', 'break'):
            continue
        text = code[start:end]
        # Look past horizontal whitespace for an immediate call.
        probe = end
        while probe < len(code) and code[probe] in ' \t':
            probe += 1
        applied = probe < len(code) and code[probe] == '('
        cls = project(sh_type, text, applied)
        for i in range(start, min(end, len(code))):
            out[i] = cls
    return out


def align_to_tokens(tokens, char_cls: bytearray):
    """Assign each lexer token the class of its first character.

    Whitespace and newline tokens are masked. A token whose span carries no
    sugar-high class (possible only if the two tokenizers disagree on a boundary)
    is masked too, so ambiguous positions never enter the loss.
    """
    labels = []
    for tok in tokens:
        if tok.kind in (1, 2):
            labels.append(MASK)
            continue
        c = char_cls[tok.start] if tok.start < len(char_cls) else 255
        labels.append(MASK if c == 255 else int(c))
    return labels
