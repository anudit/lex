"""185-language view of the proven split, deduplication, and cache pipeline."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from languages import DEFAULT_TOTAL_TOKENS

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'train'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

_spec = importlib.util.spec_from_file_location('_lex_compact_data_pipeline', BASE_DIR / 'data_pipeline.py')
assert _spec and _spec.loader
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)

FEATURE_KEYS = _base.FEATURE_KEYS + (
    'paren_depth', 'brace_depth', 'bracket_depth', 'line_pos',
    'indent_bucket', 'quote_state',
)
_base.FEATURE_KEYS = FEATURE_KEYS

LexerDataset = _base.LexerDataset
decode_rle = _base.decode_rle
align = _base.align
split_of = _base.split_of


def build(*args, total_tokens: int | None = None, **kwargs):
    if total_tokens is None:
        cache = kwargs.get('cache', './corpus/dataset')
        meta_path = Path(cache) / 'meta.json' if cache else None
        total_tokens = (
            json.loads(meta_path.read_text()).get('total_tokens_requested', DEFAULT_TOTAL_TOKENS)
            if meta_path and meta_path.exists() else DEFAULT_TOTAL_TOKENS
        )
    return _base.build(
        *args,
        total_tokens=total_tokens,
        **kwargs,
    )
