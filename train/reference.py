"""
Numpy forward pass driven entirely by weights.bin + weights.meta.json.

This exists to pin down the WGSL kernel. The shader and this file read the same
bytes and must produce the same classes; anything the shader gets wrong shows up
as a disagreement here rather than as subtly wrong colours in a browser. It is
also the readable specification of what the shader does.
"""

from __future__ import annotations

import numpy as np

from export import dequant_tensor, get_f16, load_weights


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rms_norm(x: np.ndarray, gain: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * gain


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class Reference:
    def __init__(self, out_dir: str):
        self.meta, self.planes, self.f16 = load_weights(out_dir)
        self.cfg = self.meta['config']
        self._cache: dict[str, np.ndarray] = {}

    def W(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = dequant_tensor(self.meta, self.planes, self.f16, name)
        return self._cache[name]

    def B(self, name: str) -> np.ndarray:
        t = self.meta['tensors'][name]
        o, c = t['bias_offset'], t['bias_count']
        return self.f16[o:o + c].astype(np.float32)

    def F(self, name: str) -> np.ndarray:
        return get_f16(self.meta, self.f16, name)

    def forward(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        """feats: per-token int arrays keyed like the tokenizer output. Returns (T, 9)."""
        cfg = self.cfg
        D, ED = cfg['dim'], cfg['embed_dim']
        T = len(feats['kind'])
        fo = self.meta['field_offsets']
        order = list(self.meta['field_sizes'].keys())

        table = self.W('embedding.table')
        e = np.zeros((T, ED), dtype=np.float32)
        word = feats['kind'] == 0
        for name in order:
            rows = table[fo[name] + feats[name]]
            if name in ('hash1', 'hash2'):
                rows = rows * word[:, None]
            e += rows
        flag_rows = table[self.meta['flag_offset']:
                          self.meta['flag_offset'] + self.meta['n_flag_bits']]
        bits = ((feats['flags'][:, None] >> np.arange(self.meta['n_flag_bits'])) & 1)
        e += bits.astype(np.float32) @ flag_rows

        x = e @ self.W('embedding.up').T + self.B('embedding.up')

        # Document signature -> low-rank FiLM -> per-layer scale and shift. The
        # signature is pooled here, before any layer runs, so it can condition
        # all of them.
        sig = np.concatenate([x.mean(0), x.max(0)])
        sig = _rms_norm(sig, self.F('film.norm.weight'))
        down = np.tanh(sig @ self.W('film.down').T + self.B('film.down'))
        up = down @ self.W('film.up').T + self.B('film.up')
        up = up.reshape(cfg['n_layers'], 2, cfg['dim'])
        strength = self.F('film.strength').reshape(cfg['n_layers'], cfg['dim'])

        x_orig = x.copy()
        for L in range(cfg['n_layers']):
            h = _rms_norm(x, self.F(f'layers.{L}.norm.weight'))
            gamma = 1.0 + strength[L] * np.tanh(up[L, 0])
            beta = strength[L] * np.tanh(up[L, 1])
            h = h * gamma + beta
            k = cfg['kernel_size']
            dils = cfg.get('dilations', (1, 2, 4))
            dil = dils[L] if L < len(dils) else 1
            pad = (k // 2) * dil
            hp = np.pad(h, ((pad, pad), (0, 0)))
            dw = self.W(f'layers.{L}.dw')              # (D, k)
            conv = np.zeros_like(h)
            for j in range(k):
                offset = j * dil
                conv += hp[offset:offset + T] * dw[:, j]
            conv += self.B(f'layers.{L}.dw')

            proj = conv @ self.W(f'layers.{L}.proj_in').T + self.B(f'layers.{L}.proj_in')
            cand, gate = proj[:, :D], proj[:, D:]
            b = np.tanh(cand) * _sigmoid(gate)

            has_reset = f'layers.{L}.reset_f' in self.meta['tensors']
            if has_reset:
                rf = np.log1p(np.exp(self.F(f'layers.{L}.reset_f')))[None, :]
                rb = np.log1p(np.exp(self.F(f'layers.{L}.reset_b')))[None, :]
                af = _sigmoid(self.F(f'layers.{L}.decay_f')[None, :] - rf * np.abs(b))
                ab = _sigmoid(self.F(f'layers.{L}.decay_b')[None, :] - rb * np.abs(b))
            else:
                af = _sigmoid(self.F(f'layers.{L}.decay_f'))
                ab = _sigmoid(self.F(f'layers.{L}.decay_b'))

            fwd = np.zeros_like(b)
            acc = np.zeros(D, dtype=np.float32)
            for t in range(T):
                aft = af[t] if has_reset else af
                acc = aft * acc + b[t]
                fwd[t] = acc
            bwd = np.zeros_like(b)
            acc = np.zeros(D, dtype=np.float32)
            for t in range(T - 1, -1, -1):
                abt = ab[t] if has_reset else ab
                acc = abt * acc + b[t]
                bwd[t] = acc

            y = (np.concatenate([fwd, bwd], -1) @ self.W(f'layers.{L}.proj_out').T
                 + self.B(f'layers.{L}.proj_out'))
            x = x + y * _sigmoid(self.F(f'layers.{L}.out_gate'))

        # Four pooled views: mean and max are constant across the sequence, the
        # prefix and suffix means vary with position. See GlobalContext.
        T = x.shape[0]
        mean = np.broadcast_to(x.mean(0), x.shape)
        mx = np.broadcast_to(x.max(0), x.shape)
        if 'global_ctx.decl_gate' in self.meta['tensors']:
            dg = _sigmoid(x * self.F('global_ctx.decl_gate'))
            prefix = np.cumsum(x * dg, axis=0) / np.cumsum(dg, axis=0).clip(min=1.0)
        else:
            counts = np.arange(1, T + 1, dtype=np.float32)[:, None]
            prefix = np.cumsum(x, axis=0) / counts
        counts = np.arange(1, T + 1, dtype=np.float32)[:, None]
        suffix = np.cumsum(x[::-1], axis=0)[::-1] / counts[::-1]
        pooled = np.concatenate([mean, mx, prefix, suffix], axis=-1)
        ctx = np.tanh(pooled @ self.W('global_ctx.summary').T + self.B('global_ctx.summary'))
        g = _sigmoid(x @ self.W('global_ctx.gate').T + self.B('global_ctx.gate'))

        hw = float(self.F('highway_scale')[0]) if 'highway_scale' in self.meta['tensors'] else 0.0
        hn = _rms_norm(x + hw * x_orig, self.F('head_norm.weight'))
        cat = np.concatenate([hn, ctx * g], -1)
        hid = _gelu(cat @ self.W('head_hidden').T + self.B('head_hidden'))
        return hid @ self.W('head_out').T + self.B('head_out')


def features_from_code(code: str) -> dict[str, np.ndarray]:
    import tokenizer
    toks = tokenizer.tokenize(code)
    return tokenizer.tokens_to_arrays(toks), toks


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='./checkpoints')
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
