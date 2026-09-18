"""
Numpy forward-pass reference for the lex-large checkpoint, to pin down
wgsl_large.py's bespoke shader the same way train/reference.py pins down
lex's. The forward pass itself is parametric off the exported config; this
file selects the matching large tokenizer version, including the wider hashes
and extended structural state.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / 'train'))

from reference import Reference  # noqa: E402
import tokenizer  # train_large's own, resolved via this file's directory being first on sys.path


def features_from_code(code: str, version: int = 2):
    toks = tokenizer.tokenize_version(code, version)
    return tokenizer.tokens_to_arrays(toks, version), toks


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='./_lex_large_download')
    ap.add_argument('--code', default='const MAX = 3.14; // hi\nfunction go(a) { return a; }')
    ap.add_argument('--json-out', default='')
    args = ap.parse_args()
    ref = Reference(args.weights)
    version = ref.meta.get('feature_version', ref.meta.get('config', {}).get('feature_version', 1))
    feats, toks = features_from_code(args.code, version)
    logits = ref.forward(feats)
    cls = logits.argmax(-1)
    from labels import CLASS_NAMES
    for t, c in zip(toks, cls):
        if t.kind not in (1, 2):
            print(f'  {t.text!r:16s} {CLASS_NAMES[c]}')
    if args.json_out:
        json.dump({'code': args.code, 'classes': cls.tolist()}, open(args.json_out, 'w'))
