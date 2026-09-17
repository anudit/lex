"""
Regenerates the shader test fixture `demo/src/refcases.json` from the shipped
export: end-to-end classes from the numpy reference for a handful of inputs
that have broken the kernel before. demo/validate.html checks the WGSL
pipeline against them, single-job and batched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import reference

CASES = {
    'js': 'export async function fetchUsers(ids = []) {\n  const MAX = 3.14;\n'
          '  return ids; // done\n}',
    'py': 'def total(node: Node) -> int:\n    # comment\n    x = 0xFF\n'
          '    return node.value + MAX_N',
    'short': 'a',
    'sym': '=>{}[]();::',
    'long': 'let x = 1; // c\n' * 40,
    'strings': 'const s = "a\\"b" + \'c\' + `d${e}f`; /* block */ // line',
    'unicode': 'const élève = "café"; // naïve\n',
}



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='./checkpoints_v2')
    ap.add_argument('--out', default='../demo/src')
    args = ap.parse_args()

    ref = reference.Reference(args.weights)
    cases = {}
    for name, code in CASES.items():
        feats, _ = reference.features_from_code(code)
        cases[name] = {'code': code,
                       'classes': ref.forward(feats).argmax(-1).tolist()}
    (Path(args.out) / 'refcases.json').write_text(json.dumps(cases))
    print(f'refcases: {len(cases)} cases '
          f'({", ".join(f"{k}={len(v["classes"])}" for k, v in cases.items())})')


if __name__ == '__main__':
    main()
