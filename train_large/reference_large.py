"""
Numpy forward-pass reference for the lex-large checkpoint, to pin down
wgsl_large.py's bespoke shader the same way train/reference.py pins down
lex's. The forward pass itself (train/reference.py's Reference class) is
already fully parametric off the checkpoint's own meta/config, so this file
only swaps in train_large's tokenizer (which adds the paren/brace/bracket
depth, line-position, indent, and quote-state fields lex-large's schema
needs and train's tokenizer doesn't produce).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / 'train'))

from reference import Reference  # noqa: E402
import tokenizer  # train_large's own, resolved via this file's directory being first on sys.path


def features_from_code(code: str):
    toks = tokenizer.tokenize(code)
    return tokenizer.tokens_to_arrays(toks), toks


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='./_lex_large_download')
    ap.add_argument('--code', default='const MAX = 3.14; // hi\nfunction go(a) { return a; }')
    ap.add_argument('--json-out', default='')
    args = ap.parse_args()
    ref = Reference(args.weights)
    feats, toks = features_from_code(args.code)
    logits = ref.forward(feats)
    cls = logits.argmax(-1)
    from labels import CLASS_NAMES
    for t, c in zip(toks, cls):
        if t.kind not in (1, 2):
            print(f'  {t.text!r:16s} {CLASS_NAMES[c]}')
    if args.json_out:
        json.dump({'code': args.code, 'classes': cls.tolist()}, open(args.json_out, 'w'))
