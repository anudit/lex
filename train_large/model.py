"""The approximately 100 KiB, 193-language neural lexer architecture.

The proven quantized blocks live in ../train/model.py. This module gives them a
larger, word-aligned configuration and extends the language-agnostic feature
table. Keeping the block implementation shared prevents training and export
math from quietly diverging between the compact and large targets.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'train'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

_spec = importlib.util.spec_from_file_location('_lex_compact_model', BASE_DIR / 'model.py')
assert _spec and _spec.loader
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)

# 1024/256 cuts word-hash collision pressure across four times as many grammars.
# The six structural fields cost only 1.03 KiB at 3 bits but give the recurrence
# exact local state that is otherwise expensive to rediscover from long context.
FIELD_SIZES: dict[str, int] = dict(_base.FIELD_SIZES)
FIELD_SIZES.update({
    'hash1': 1024,
    'hash2': 256,
    'paren_depth': 8,
    'brace_depth': 8,
    'bracket_depth': 8,
    'line_pos': 8,
    'indent_bucket': 8,
    'quote_state': 4,
})
N_FLAG_BITS = _base.N_FLAG_BITS
_base.FIELD_SIZES = FIELD_SIZES


@dataclass
class LexerConfig(_base.LexerConfig):
    """Shipping student configuration: exactly 102,368 packed bytes."""

    dim: int = 96
    embed_dim: int = 64
    n_layers: int = 4
    head_hidden: int = 192
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    film_rank: int = 64
    erase_rank: int = 16


class NeuralLexer(_base.NeuralLexer):
    def __init__(self, cfg: LexerConfig | None = None):
        super().__init__(cfg or LexerConfig())


FeatureEmbedding = _base.FeatureEmbedding
GlobalContext = _base.GlobalContext
DocSignature = _base.DocSignature
FiLM = _base.FiLM
BidiGLUBlock = _base.BidiGLUBlock
assoc_scan = _base.assoc_scan
