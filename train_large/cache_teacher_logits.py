"""Run the teacher once and cache FP16 logits for exact, cheap distillation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import data_pipeline
from model import LexerConfig, NeuralLexer

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'train'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./checkpoints_teacher/best_model.pt')
    parser.add_argument('--dataset', default='./corpus/dataset')
    parser.add_argument('--out', default='./corpus/dataset/teacher_logits.npy')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--compile-mode', default='max-autotune')
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location('_lex_cache_trainer', BASE_DIR / 'train.py')
    assert spec and spec.loader
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)

    device = trainer.pick_device()
    if device.type != 'cuda':
        raise SystemExit('teacher logit caching is intended to run on CUDA')
    torch.set_float32_matmul_precision('high')
    amp_dtype = torch.bfloat16

    datasets, _ = data_pipeline.build(cache=args.dataset, total_tokens=None)
    dataset = datasets['train']
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = NeuralLexer(LexerConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.set_quant(False)
    if args.compile_mode != 'none':
        model.compile(mode=args.compile_mode)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.partial.npy')
    progress_path = output.with_suffix('.progress.json')
    shape = (len(dataset), dataset.seq_len, model.cfg.num_classes)
    completed = 0
    if partial.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if tuple(progress.get('shape', ())) == shape:
            completed = int(progress.get('completed', 0))
            logits_cache = np.lib.format.open_memmap(partial, mode='r+')
        else:
            raise SystemExit('partial teacher cache has a different dataset shape; remove it')
    else:
        logits_cache = np.lib.format.open_memmap(
            partial, mode='w+', dtype=np.float16, shape=shape)

    subset = Subset(dataset, range(completed, len(dataset)))
    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None)
    print(f'>> device: {device}; caching {len(dataset) - completed:,} windows '
          f'from offset {completed:,}', flush=True)

    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            feats, _, valid, _ = trainer.to_device(batch, device)
            with trainer.autocast_context(device, amp_dtype):
                logits = model(feats, valid)
            indices = batch['index'].numpy()
            logits_cache[indices] = logits.float().cpu().numpy().astype(np.float16)
            completed = int(indices[-1]) + 1
            if step % 25 == 0 or completed == len(dataset):
                logits_cache.flush()
                progress_path.write_text(json.dumps({
                    'shape': shape, 'completed': completed,
                }, indent=2) + '\n')
                print(f'   {completed:,}/{len(dataset):,} windows', flush=True)

    logits_cache.flush()
    del logits_cache
    partial.replace(output)
    progress_path.unlink(missing_ok=True)
    meta_path = output.with_suffix('.meta.json')
    meta_path.write_text(json.dumps({
        'format': 'lex-teacher-logits-v1',
        'shape': shape,
        'dtype': 'float16',
        'checkpoint_sha256': sha256(checkpoint_path),
        'dataset_meta_sha256': sha256(Path(args.dataset) / 'meta.json'),
    }, indent=2) + '\n')
    print(f'>> wrote {output} ({output.stat().st_size / 1024**3:.2f} GiB)', flush=True)


if __name__ == '__main__':
    main()
