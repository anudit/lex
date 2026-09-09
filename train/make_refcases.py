"""
Regenerates the shader test fixtures from the current export.

Both files come from one run so they cannot drift apart: `refcases.json` holds
end-to-end classes, `refstages.json` holds the hidden state after each stage of
the model. The WGSL kernel is checked against both, which localizes any
mismatch to a single stage instead of leaving a wrong colour to explain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _rms(z, g, eps=1e-6):
    return z / np.sqrt((z * z).mean(-1, keepdims=True) + eps) * g


def stages_for(ref: reference.Reference, code: str) -> dict[str, list]:
    """Hidden state after embedding, each layer, the global context, and the head."""
    feats, toks = reference.features_from_code(code)
    cfg = ref.cfg
    D, ED = cfg['dim'], cfg['embed_dim']
    T = len(feats['kind'])
    fo = ref.meta['field_offsets']
    order = list(ref.meta['field_sizes'].keys())

    table = ref.W('embedding.table')
    e = np.zeros((T, ED), dtype=np.float32)
    word = feats['kind'] == 0
    for name in order:
        rows = table[fo[name] + feats[name]]
        if name in ('hash1', 'hash2'):
            rows = rows * word[:, None]
        e += rows
    fr = table[ref.meta['flag_offset']:ref.meta['flag_offset'] + ref.meta['n_flag_bits']]
    bits = (feats['flags'][:, None] >> np.arange(ref.meta['n_flag_bits'])) & 1
    e += bits.astype(np.float32) @ fr
    x = e @ ref.W('embedding.up').T + ref.B('embedding.up')
    out = {'1': x.copy()}

    for L in range(cfg['n_layers']):
        h = _rms(x, ref.F(f'layers.{L}.norm.weight'))
        k = cfg['kernel_size']
        pad = k // 2
        hp = np.pad(h, ((pad, pad), (0, 0)))
        dw = ref.W(f'layers.{L}.dw')
        conv = np.zeros_like(h)
        for j in range(k):
            conv += hp[j:j + T] * dw[:, j]
        conv += ref.B(f'layers.{L}.dw')
        proj = conv @ ref.W(f'layers.{L}.proj_in').T + ref.B(f'layers.{L}.proj_in')
        b = np.tanh(proj[:, :D]) * _sigmoid(proj[:, D:])
        af = _sigmoid(ref.F(f'layers.{L}.decay_f'))
        ab = _sigmoid(ref.F(f'layers.{L}.decay_b'))
        fwd = np.zeros_like(b)
        acc = np.zeros(D, dtype=np.float32)
        for t in range(T):
            acc = af * acc + b[t]
            fwd[t] = acc
        bwd = np.zeros_like(b)
        acc = np.zeros(D, dtype=np.float32)
        for t in range(T - 1, -1, -1):
            acc = ab * acc + b[t]
            bwd[t] = acc
        y = (np.concatenate([fwd, bwd], -1) @ ref.W(f'layers.{L}.proj_out').T
             + ref.B(f'layers.{L}.proj_out'))
        x = x + y * _sigmoid(ref.F(f'layers.{L}.out_gate'))
        out[str(2 + L)] = x.copy()

    counts = np.arange(1, T + 1, dtype=np.float32)[:, None]
    pooled = np.concatenate([
        np.broadcast_to(x.mean(0), x.shape), np.broadcast_to(x.max(0), x.shape),
        np.cumsum(x, axis=0) / counts,
        np.cumsum(x[::-1], axis=0)[::-1] / counts[::-1],
    ], axis=-1)
    ctx = np.tanh(pooled @ ref.W('global_ctx.summary').T + ref.B('global_ctx.summary'))
    g = 1.0 / (1.0 + np.exp(-(x @ ref.W('global_ctx.gate').T + ref.B('global_ctx.gate'))))
    out[str(2 + cfg['n_layers'])] = ctx * g

    logits = ref.forward(feats)
    padded = np.zeros((T, D), dtype=np.float32)
    padded[:, :logits.shape[1]] = logits
    out[str(3 + cfg['n_layers'])] = padded
    return {k: v.ravel().tolist() for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='./checkpoints')
    ap.add_argument('--out', default='../demo/src')
    ap.add_argument('--stage-case', default='js')
    args = ap.parse_args()

    ref = reference.Reference(args.weights)
    out = Path(args.out)

    cases = {}
    for name, code in CASES.items():
        feats, _ = reference.features_from_code(code)
        cases[name] = {'code': code,
                       'classes': ref.forward(feats).argmax(-1).tolist()}
    (out / 'refcases.json').write_text(json.dumps(cases))

    code = CASES[args.stage_case]
    (out / 'refstages.json').write_text(
        json.dumps({'code': code, 'stages': stages_for(ref, code)}))
    print(f'refcases: {len(cases)} cases '
          f'({", ".join(f"{k}={len(v["classes"])}" for k, v in cases.items())})')
    print(f'refstages: {args.stage_case}, {ref.cfg["n_layers"] + 3} stages')


if __name__ == '__main__':
    main()
