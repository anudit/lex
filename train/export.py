"""
Exports the trained model to a flat binary the WebGPU runtime can upload directly.

Everything that is needed to reproduce the PyTorch forward pass ships. The
previous exporter silently omitted the depthwise kernels, the RMSNorm gains and
the decay logits -- 1,216 parameters -- so the file it produced could not
reconstruct the model at all, and it wrote a different scale than the one the
forward pass used. Both are structural problems, so this file is built around
two invariants:

  * Every parameter is either a quantized tensor (bit-planes + scales) or an fp16
    tensor. `verify_roundtrip` asserts the two sets cover the model exactly.
  * Quantized values come from `quant.export_tensors`, which calls the same
    `row_scale` the forward pass calls. Export cannot drift from training.

Layout of weights.bin:
    [ uint32 bit-plane blob ][ float16 blob ]
Offsets in the metadata are in units of the respective element type.
"""

from __future__ import annotations

import json
from dataclasses import asdict
import os
from pathlib import Path

import numpy as np
import torch

from model import NeuralLexer, LexerConfig, FIELD_SIZES, N_FLAG_BITS
from quant import QuantEmbedding, QuantLinear, pack_bitplanes


def export_model(model: NeuralLexer, out_dir: str, verify: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()

    planes: list[np.ndarray] = []
    f16: list[np.ndarray] = []
    tensors: dict[str, dict] = {}
    plane_off = 0
    f16_off = 0
    covered: set[str] = set()

    def add_f16(name: str, arr: np.ndarray, kind: str) -> dict:
        nonlocal f16_off
        flat = arr.astype(np.float16).ravel()
        entry = {'kind': kind, 'shape': list(arr.shape),
                 'f16_offset': f16_off, 'f16_count': int(flat.size)}
        f16.append(flat)
        f16_off += flat.size
        return entry

    for name, mod in model.named_modules():
        if not isinstance(mod, (QuantLinear, QuantEmbedding)):
            continue
        t = mod.export_tensors()
        codes = t['codes']
        cols = t['shape'][1]
        # The WGSL kernel addresses every "linear" tensor except the depthwise
        # kernel (`.dw`, read scalar-by-scalar via wgt()) as `row * words_per_row`,
        # which is only valid if each row starts on a 32-bit word boundary --
        # true as long as every row is a multiple of 32 columns wide. film_rank
        # dropping from 64 to 48 broke that for film.up without anyone touching
        # the shader, since the dense bit-packing below has no row awareness of
        # its own. Pad the row to the next word boundary with throwaway bits
        # rather than relying on every future width staying 32-aligned by luck --
        # the corresponding input is always masked to zero for the padding, so
        # what the padding bits decode to cannot affect the result.
        row_pad = 0
        if t['kind'] == 'linear' and not name.endswith('.dw') and cols % 32 != 0:
            row_pad = 32 - (cols % 32)
            codes = np.pad(codes, ((0, 0), (0, row_pad)))
        planes2d = pack_bitplanes(codes, t['bits'])
        packed = planes2d.ravel()
        entry = {
            'kind': t['kind'],
            'bits': t['bits'],
            'shape': t['shape'],
            'plane_offset': plane_off,
            'plane_count': int(packed.size),
            'words_per_plane': int(planes2d.shape[1]),
            'words_per_row': (cols + row_pad) // 32,  # 0 when the row is narrower than a word
        }
        planes.append(packed)
        plane_off += packed.size
        entry.update({k: v for k, v in add_f16(f'{name}.scale', t['scale'], 'scale').items()
                      if k in ('f16_offset', 'f16_count')})
        entry['scale_offset'] = entry.pop('f16_offset')
        entry['scale_count'] = entry.pop('f16_count')
        if t.get('groups') is not None:
            # Row -> scale-group id, so the shader can find a row's scale.
            entry['group_sizes'] = _group_sizes(t['groups'])
        if t['bias'] is not None:
            b = add_f16(f'{name}.bias', t['bias'], 'bias')
            entry['bias_offset'] = b['f16_offset']
            entry['bias_count'] = b['f16_count']
        tensors[name] = entry
        covered.add(f'{name}.weight')
        if getattr(mod, 'bias', None) is not None:
            covered.add(f'{name}.bias')

    # Everything not owned by a quantized module: norm gains, decay logits,
    # output gates. These are small and ship at full fp16 precision.
    for name, p in model.named_parameters():
        if name in covered:
            continue
        tensors[name] = add_f16(name, p.detach().cpu().numpy(), 'f16')
        covered.add(name)

    missing = {n for n, _ in model.named_parameters()} - covered
    if missing:
        raise RuntimeError(f'parameters missing from export: {sorted(missing)}')

    plane_blob = (np.concatenate(planes) if planes
                  else np.zeros(0, dtype=np.uint32)).astype(np.uint32)
    f16_blob = (np.concatenate(f16) if f16 else np.zeros(0, dtype=np.float16))

    bin_path = out / 'weights.bin'
    with bin_path.open('wb') as fh:
        fh.write(plane_blob.tobytes())
        fh.write(f16_blob.tobytes())

    cfg = model.cfg
    meta = {
        'format': 'neural-lexer-v2',
        'config': {
            **asdict(cfg),
            'dim': cfg.dim, 'embed_dim': cfg.embed_dim, 'n_layers': cfg.n_layers,
            'kernel_size': cfg.kernel_size, 'head_hidden': cfg.head_hidden,
            'num_classes': cfg.num_classes,
            'film_rank': cfg.film_rank,
            'erase_rank': getattr(cfg, 'erase_rank', 0),
            'dilations': list(getattr(cfg, 'dilations', (1, 2, 4))),
        },
        'field_sizes': dict(FIELD_SIZES),
        'field_offsets': model.embedding.field_offsets,
        'flag_offset': model.embedding.flag_offset,
        'n_flag_bits': N_FLAG_BITS,
        'plane_words': int(plane_blob.size),
        'f16_count': int(f16_blob.size),
        'plane_bytes': int(plane_blob.nbytes),
        'f16_bytes': int(f16_blob.nbytes),
        'total_bytes': int(plane_blob.nbytes + f16_blob.nbytes),
        'parameters': int(sum(p.numel() for p in model.parameters())),
        'tensors': tensors,
    }
    (out / 'weights.meta.json').write_text(json.dumps(meta, indent=2))

    report = {'bytes': meta['total_bytes'], 'kb': meta['total_bytes'] / 1024.0}
    if verify:
        report['max_abs_error'] = verify_roundtrip(model, out)
    return report | {'meta': meta}


def _group_sizes(groups: np.ndarray) -> list[int]:
    sizes: list[int] = []
    cur = groups[0]
    n = 0
    for g in groups:
        if g != cur:
            sizes.append(n)
            cur, n = g, 0
        n += 1
    sizes.append(n)
    return sizes


def load_weights(out_dir: str) -> tuple[dict, np.ndarray, np.ndarray]:
    out = Path(out_dir)
    meta = json.loads((out / 'weights.meta.json').read_text())
    raw = (out / 'weights.bin').read_bytes()
    nw = meta['plane_words']
    planes = np.frombuffer(raw[:nw * 4], dtype=np.uint32)
    f16 = np.frombuffer(raw[nw * 4:], dtype=np.float16)
    return meta, planes, f16


def dequant_tensor(meta: dict, planes: np.ndarray, f16: np.ndarray,
                   name: str) -> np.ndarray:
    """Reconstruct one quantized tensor exactly as the shader must."""
    from quant import unpack_bitplanes
    t = meta['tensors'][name]
    rows, cols = t['shape']
    bits = t['bits']
    packed = planes[t['plane_offset']:t['plane_offset'] + t['plane_count']]
    packed = packed.reshape(bits, t['words_per_plane'])
    # words_per_row*32 is the padded row width when export.py row-aligned this
    # tensor (see export_model); for anything unpadded (words_per_row is 0 for
    # a row narrower than one word, e.g. the depthwise kernel) that is smaller
    # than the true width, so fall back to the true width instead.
    padded_cols = max(cols, t.get('words_per_row', 0) * 32)
    codes = unpack_bitplanes(packed, bits, rows, padded_cols)[:, :cols]
    scales = f16[t['scale_offset']:t['scale_offset'] + t['scale_count']].astype(np.float32)
    if 'group_sizes' in t:
        scales = np.repeat(scales, t['group_sizes'])
    if bits == 1:
        q = np.where(codes == 1, 1.0, -1.0).astype(np.float32)
    else:
        q = codes.astype(np.float32) - float(2 ** (bits - 1))
    return q * scales.reshape(-1, 1)


def get_f16(meta: dict, f16: np.ndarray, name: str) -> np.ndarray:
    t = meta['tensors'][name]
    a = f16[t['f16_offset']:t['f16_offset'] + t['f16_count']].astype(np.float32)
    return a.reshape(t['shape'])


def verify_roundtrip(model: NeuralLexer, out_dir: Path, atol: float = 2e-3) -> float:
    """Assert the exported file reproduces the model it came from.

    This is the test the old suite lacked: it asserted the file existed and was
    under 31 KB, which is compatible with the file being unusable.
    """
    meta, planes, f16 = load_weights(str(out_dir))
    worst = 0.0
    for name, mod in model.named_modules():
        if not isinstance(mod, (QuantLinear, QuantEmbedding)):
            continue
        ref = mod.effective().detach().cpu().numpy()
        got = dequant_tensor(meta, planes, f16, name)
        err = float(np.abs(ref - got).max())
        worst = max(worst, err)
        if err > atol:
            raise AssertionError(
                f'{name}: export differs from the trained weights by {err:.2e} '
                f'(bits={mod.bits}); training and export are out of sync')
    for name, p in model.named_parameters():
        t = meta['tensors'].get(name)
        if t is None or t.get('kind') != 'f16':
            continue
        err = float(np.abs(p.detach().cpu().numpy() - get_f16(meta, f16, name)).max())
        worst = max(worst, err)
    return worst


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./checkpoints/best_model.pt')
    ap.add_argument('--out', default='./checkpoints')
    args = ap.parse_args()
    m = NeuralLexer()
    if os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'config' in ck:
            m = NeuralLexer(LexerConfig(**ck['config']))
        m.load_state_dict(ck['model_state_dict'])
    r = export_model(m, args.out)
    print(f"weights.bin {r['kb']:.2f} KB   round-trip max error {r['max_abs_error']:.2e}")
