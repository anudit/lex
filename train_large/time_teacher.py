"""Time teacher candidates' training steps on the GPU that will train them.

Per-step cost does not transfer between devices: on MPS the dilation-16/32
depthwise convs of a 6-layer teacher dominated (4.7x the 192x4 teacher's step
at 256 wide), which may not hold for CUDA kernels. Run this on the RTX box
before choosing --dim/--n-layers for train_teacher.py. It memory-maps a few
real train windows instead of loading the 18 GB cache, and measures a full
optimizer step (forward, backward, AdamW) under the same bf16 autocast and
full-precision mode the teacher trains with, plus peak CUDA memory.

    .venv/bin/python time_teacher.py
    .venv/bin/python time_teacher.py --batch-size 32 --shapes 256x4 256x6 320x4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import data_pipeline as dp
from model import LexerConfig, NeuralLexer
from smoke_mps import mmap_member

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'train'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

REFERENCE = '192x4'  # the teacher behind the 2026-09-13 student


def shape_config(spec: str) -> LexerConfig:
    """'256x4' -> the train_teacher.py proportions at that width and depth."""
    dim, layers = (int(v) for v in spec.lower().split('x'))
    return LexerConfig(dim=dim, embed_dim=dim // 2, head_hidden=dim * 2, film_rank=dim // 2,
                       erase_rank=dim // 4 if dim >= 256 else 32, n_layers=layers,
                       dilations=tuple(2 ** i for i in range(layers)), kernel_size=7,
                       input_bits=1, head_bits=1, output_bits=1,
                       dropout=0.0 if spec == REFERENCE else 0.1)


def load_batch(dataset: Path, batch_size: int, device: torch.device):
    path = dataset / 'train.npz'
    offsets = mmap_member(path, 'offsets')
    full = np.flatnonzero(np.diff(offsets[:100_000]) == 512)[:batch_size]
    feats = {}
    for key in dp.FEATURE_KEYS:
        array = mmap_member(path, key)
        feats[key] = torch.from_numpy(np.stack(
            [np.asarray(array[offsets[i]:offsets[i] + 512]) for i in full])).to(device)
    valid = torch.ones(len(full), 512, dtype=torch.bool, device=device)
    return feats, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./corpus/dataset')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--shapes', nargs='+', default=[REFERENCE, '256x4', '256x6'])
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_float32_matmul_precision('high')
        autocast = lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        sync = torch.cuda.synchronize
        print(f'>> {torch.cuda.get_device_name()} (torch {torch.__version__}, bf16)')
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        import contextlib
        autocast = contextlib.nullcontext
        sync = torch.mps.synchronize if device.type == 'mps' else (lambda: None)
        print(f'>> {device} (fp32) -- not representative of the RTX box')

    shapes = args.shapes if REFERENCE in args.shapes else [REFERENCE, *args.shapes]
    feats, valid = load_batch(Path(args.dataset), args.batch_size, device)
    results = {}
    for spec in shapes:
        model = NeuralLexer(shape_config(spec)).to(device)
        model.set_quant(False)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                fused=device.type == 'cuda')
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        warmup = 3
        for step in range(warmup + args.steps):
            if step == warmup:
                sync()
                start = time.perf_counter()
            with autocast():
                loss = model(feats, valid).float().logsumexp(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sync()
        per_step = (time.perf_counter() - start) / args.steps
        params = sum(p.numel() for p in model.parameters())
        memory = (f'{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB peak'
                  if device.type == 'cuda' else '')
        results[spec] = per_step
        print(f'   {spec:>7}  {params / 1e6:5.2f}M params  {per_step * 1000:7.1f} ms/step  '
              f'{per_step / results[REFERENCE]:4.2f}x  {memory}', flush=True)
        del model, opt
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print('\nA fold teacher runs half the reference steps per epoch, so two fold '
          'teachers at ratio r cost about r x the old single-teacher run in total.')


if __name__ == '__main__':
    main()
