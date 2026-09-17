"""
Exports the trained model to a flat binary the WebGPU runtime can upload directly.

Everything that is needed to reproduce the PyTorch forward pass ships, built
around two invariants:

  * Every parameter is either a quantized tensor (bit-planes + scales) or a
    scalar tensor (8-bit codes for lex-lite, fp16 for lex-large).
    `verify_roundtrip` asserts the two sets cover the model exactly.
  * Quantized values come from `quant.export_tensors`, which calls the same
    `row_scale` the forward pass calls. Export cannot drift from training.

Layout of weights.bin:
    [ uint32 bit-plane blob ][ float16 blob ][ uint8 blob (scalar_bits=8 only) ]
Offsets in the metadata are in units of the respective element type.

With scalar_bits=8 the `f16_offset`s index a *decoded* float table: the loader
walks `scalar_segments` in order, reading fp16 headers and raw values from the
float16 blob and 8-bit codes from the uint8 blob, and produces exactly the
table an fp16 export would have produced. The shader never sees the encoding.
"""

from __future__ import annotations

import json
from dataclasses import asdict
import os
from pathlib import Path

import numpy as np
import torch

from model import NeuralLexer, LexerConfig, N_FLAG_BITS
from quant import (QuantEmbedding, QuantLinear, int8_codes, log8_codes, pack_bitplanes,
                   row_scale)


def export_model(model: NeuralLexer, out_dir: str, verify: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    # The file is the quantized model; a freshly loaded checkpoint has not had
    # the 8-bit scale path switched on yet.
    model.set_quant(True)

    planes: list[np.ndarray] = []
    f16: list[np.ndarray] = []          # encoded float16 stream (headers + raw)
    u8: list[np.ndarray] = []           # encoded 8-bit codes
    table: list[np.ndarray] = []        # decoded scalar table (float32)
    segments: list[dict] = []
    tensors: dict[str, dict] = {}
    plane_off = 0
    f16_off = 0
    covered: set[str] = set()
    scalar8 = getattr(model.cfg, 'scalar_bits', 16) == 8

    def add_f16(name: str, arr: np.ndarray, kind: str, log_codes=None) -> dict:
        nonlocal f16_off
        arr = np.asarray(arr, dtype=np.float32)
        flat = arr.ravel()
        if not scalar8:
            f16.append(flat.astype(np.float16))
            table.append(flat.astype(np.float16).astype(np.float32))
            segments.append({'enc': 'f16', 'count': int(flat.size)})
        elif log_codes is not None:
            codes, lo, step = log_codes
            f16.append(np.array([lo, step], dtype=np.float16))
            u8.append(codes.ravel().astype(np.uint8))
            table.append(np.exp(np.float32(lo) + codes.ravel().astype(np.float32)
                                * np.float32(step)).astype(np.float32))
            segments.append({'enc': 'log8', 'count': int(flat.size)})
        else:
            codes, step = int8_codes(flat)
            f16.append(np.array([step], dtype=np.float16))
            u8.append(codes.view(np.uint8))
            table.append(codes.astype(np.float32) * np.float32(step))
            segments.append({'enc': 'i8', 'count': int(flat.size)})
        entry = {'kind': kind, 'shape': list(arr.shape),
                 'f16_offset': f16_off, 'f16_count': int(flat.size)}
        f16_off += flat.size
        return entry

    def scale_log_codes(mod) -> tuple | None:
        if not scalar8:
            return None
        w = mod.weight.detach()
        if isinstance(mod, QuantEmbedding):
            raw = row_scale(w, mod.bits, mod.groups, mod.n_scales).squeeze(-1)
            codes, lo, step = log8_codes(raw.cpu().numpy())
            if mod.groups is not None:
                g = mod.groups.cpu().numpy()
                first = np.zeros(mod.n_scales, dtype=np.int64)
                first[g[::-1]] = np.arange(len(g))[::-1]
                codes = codes[first]
            return codes, lo, step
        raw = row_scale(w, mod.bits).squeeze(-1)
        return log8_codes(raw.cpu().numpy())

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
        entry.update({k: v for k, v in add_f16(f'{name}.scale', t['scale'], 'scale',
                                               scale_log_codes(mod)).items()
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
    # output gates.
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
    u8_blob = (np.concatenate(u8) if u8 else np.zeros(0, dtype=np.uint8))

    bin_path = out / 'weights.bin'
    with bin_path.open('wb') as fh:
        fh.write(plane_blob.tobytes())
        fh.write(f16_blob.tobytes())
        fh.write(u8_blob.tobytes())

    cfg = model.cfg
    meta = {
        'format': 'neural-lexer-v2',
        'config': {
            # Dropout is training-only; keeping it out leaves exported metadata
            # byte-identical to pre-dropout exports.
            **{k: v for k, v in asdict(cfg).items() if k != 'dropout'},
            'dim': cfg.dim, 'embed_dim': cfg.embed_dim, 'n_layers': cfg.n_layers,
            'kernel_size': cfg.kernel_size, 'head_hidden': cfg.head_hidden,
            'num_classes': cfg.num_classes,
            'film_rank': cfg.film_rank,
            'erase_rank': getattr(cfg, 'erase_rank', 0),
            'dilations': list(getattr(cfg, 'dilations', (1, 2, 4))),
        },
        'feature_version': getattr(cfg, 'feature_version', 1),
        'field_sizes': dict(cfg.fields()),
        'field_offsets': model.embedding.field_offsets,
        'flag_offset': model.embedding.flag_offset,
        'n_flag_bits': N_FLAG_BITS,
        'plane_words': int(plane_blob.size),
        # f16_count is the size of the decoded scalar table the offsets index.
        'f16_count': int(f16_off),
        'f16_stream_count': int(f16_blob.size),
        'u8_count': int(u8_blob.size),
        'scalar_bits': 8 if scalar8 else 16,
        'scalar_segments': segments,
        'plane_bytes': int(plane_blob.nbytes),
        'f16_bytes': int(f16_blob.nbytes),
        'u8_bytes': int(u8_blob.nbytes),
        'total_bytes': int(plane_blob.nbytes + f16_blob.nbytes + u8_blob.nbytes),
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
    if meta.get('scalar_bits', 16) != 8:
        f16 = np.frombuffer(raw[nw * 4:], dtype=np.float16)
        return meta, planes, f16
    nf = meta['f16_stream_count']
    f16s = np.frombuffer(raw[nw * 4:nw * 4 + nf * 2], dtype=np.float16).astype(np.float32)
    u8s = np.frombuffer(raw[nw * 4 + nf * 2:], dtype=np.uint8)
    return meta, planes, decode_scalars(meta['scalar_segments'], f16s, u8s)


def decode_scalars(segments: list[dict], f16s: np.ndarray, u8s: np.ndarray) -> np.ndarray:
    """Expand the encoded scalar streams into the flat float table."""
    out = []
    fi = ui = 0
    for seg in segments:
        n = seg['count']
        if seg['enc'] == 'f16':
            out.append(f16s[fi:fi + n]); fi += n
        elif seg['enc'] == 'i8':
            step = f16s[fi]; fi += 1
            out.append(u8s[ui:ui + n].view(np.int8).astype(np.float32) * step); ui += n
        else:
            lo, step = f16s[fi], f16s[fi + 1]; fi += 2
            out.append(np.exp(lo + u8s[ui:ui + n].astype(np.float32) * step)); ui += n
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


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

    Checks every quantized tensor and every scalar against the model's own
    (quantized) values.
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
    scalar8 = meta.get('scalar_bits', 16) == 8
    for name, p in model.named_parameters():
        t = meta['tensors'].get(name)
        owner = meta['tensors'].get(name[:-len('.bias')]) if name.endswith('.bias') else None
        if t is not None and t.get('kind') == 'f16':
            ref = p.detach()
            got = get_f16(meta, f16, name)
        elif owner is not None and 'bias_offset' in owner:
            ref = p.detach()
            got = f16[owner['bias_offset']:owner['bias_offset'] + owner['bias_count']]
        else:
            continue
        if scalar8:
            from quant import fake_int8
            ref = fake_int8(ref)
        err = float(np.abs(ref.cpu().numpy().reshape(-1) - np.asarray(got).reshape(-1)).max())
        worst = max(worst, err)
        if err > atol:
            raise AssertionError(f'{name}: exported scalar differs by {err:.2e}')
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
