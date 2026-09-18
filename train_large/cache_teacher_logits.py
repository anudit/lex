"""Run the teacher once and cache FP16 logits for exact, cheap distillation.

With one checkpoint, every training window gets that teacher's logits. With
cross-fitted teachers (train_teacher.py --teacher-folds K --teacher-fold k, one
checkpoint per fold), every window gets the average logits of the teachers
that did *not* train on it. A teacher scoring its own training windows is
near-memorized (95.8% train accuracy against ~90% val in the 192-wide run), so
its soft targets mostly echo the hard labels, label noise included; held-out
logits carry the uncertainty the student is supposed to learn from. Windows
from languages too small to split (fold -1) get the average of all teachers.
"""

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


def fold_plan(checkpoints: list[dict]) -> tuple[int, list[int]]:
    """(fold count, each checkpoint's fold). Fold count 0 means no cross-fitting."""
    folds = [(int(c.get('train_args', {}).get('teacher_folds', 0) or 0),
              int(c.get('train_args', {}).get('teacher_fold', 0) or 0)) for c in checkpoints]
    counts = {k for k, _ in folds}
    if counts == {0} or counts == {1}:
        if len(checkpoints) != 1:
            raise SystemExit('multiple checkpoints given, but none was trained with --teacher-folds')
        return 0, [0]
    if len(counts) != 1:
        raise SystemExit(f'checkpoints disagree on --teacher-folds: {sorted(counts)}')
    k = counts.pop()
    got = sorted(f for _, f in folds)
    if got != list(range(k)):
        raise SystemExit(f'--teacher-folds {k} needs one checkpoint per fold 0..{k - 1}; got folds {got}')
    return k, [f for _, f in folds]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='+', default=['./checkpoints_teacher/best_model.pt'],
                        help='one teacher, or one cross-fitted teacher per fold')
    parser.add_argument('--dataset', default='./corpus/dataset_v2')
    parser.add_argument('--out', default='./corpus/dataset_v2/teacher_logits.npy')
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
        print(f'>> warning: caching on {device}; this is only practical for smoke tests', flush=True)
    torch.set_float32_matmul_precision('high')
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else None

    datasets, _ = data_pipeline.build(cache=args.dataset, total_tokens=None)
    dataset = datasets['train']
    checkpoint_paths = [Path(p) for p in args.checkpoint]
    checkpoints = [torch.load(p, map_location=device, weights_only=False) for p in checkpoint_paths]
    n_folds, model_folds = fold_plan(checkpoints)
    models = []
    for checkpoint in checkpoints:
        model = NeuralLexer(LexerConfig(**checkpoint['config'])).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.set_quant(False)
        if args.compile_mode != 'none' and device.type == 'cuda':
            model.compile(mode=args.compile_mode)
        models.append(model)
    del checkpoints

    num_classes = models[0].cfg.num_classes
    if n_folds:
        window_folds = data_pipeline.teacher_folds(dataset, n_folds)
        # mix[f + 1, m] = weight of model m for a window in fold f; row 0 is shared.
        mix = np.zeros((n_folds + 1, len(models)), dtype=np.float32)
        mix[0, :] = 1.0 / len(models)
        for f in range(n_folds):
            held_out = [m for m, mf in enumerate(model_folds) if mf != f]
            mix[f + 1, held_out] = 1.0 / len(held_out)
        mix_t = torch.from_numpy(mix).to(device)
        shared = int((window_folds < 0).sum())
        print(f'>> cross-fitted: {len(models)} teachers over {n_folds} folds; '
              f'{len(dataset) - shared:,} windows held out, {shared:,} shared', flush=True)
    else:
        window_folds = None

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.partial.npy')
    progress_path = output.with_suffix('.progress.json')
    shape = (len(dataset), dataset.seq_len, num_classes)
    completed = 0
    if partial.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if tuple(progress.get('shape', ())) != shape:
            raise SystemExit('partial teacher cache has a different dataset shape; remove it')
        if progress.get('checkpoints') not in (None, [str(p) for p in checkpoint_paths]):
            raise SystemExit('partial teacher cache came from different checkpoints; remove it')
        completed = int(progress.get('completed', 0))
        logits_cache = np.lib.format.open_memmap(partial, mode='r+')
    else:
        logits_cache = np.lib.format.open_memmap(
            partial, mode='w+', dtype=np.float16, shape=shape)

    subset = Subset(dataset, range(completed, len(dataset)))
    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None)
    print(f'>> device: {device}; caching {len(dataset) - completed:,} windows '
          f'from offset {completed:,}', flush=True)

    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            feats, _, valid, _ = trainer.to_device(batch, device)
            indices = batch['index'].numpy()
            with trainer.autocast_context(device, amp_dtype):
                outs = [model(feats, valid).float() for model in models]
            if window_folds is None:
                logits = outs[0]
            else:
                weights = mix_t[torch.from_numpy(window_folds[indices].astype(np.int64) + 1).to(device)]
                logits = torch.einsum('mbtc,bm->btc', torch.stack(outs), weights)
            logits_cache[indices] = logits.cpu().numpy().astype(np.float16)
            completed = int(indices[-1]) + 1
            if step % 25 == 0 or completed == len(dataset):
                logits_cache.flush()
                progress_path.write_text(json.dumps({
                    'shape': shape, 'completed': completed,
                    'checkpoints': [str(p) for p in checkpoint_paths],
                }, indent=2) + '\n')
                print(f'   {completed:,}/{len(dataset):,} windows', flush=True)

    logits_cache.flush()
    del logits_cache
    partial.replace(output)
    progress_path.unlink(missing_ok=True)
    meta_path = output.with_suffix('.meta.json')
    meta_path.write_text(json.dumps({
        'format': 'lex-teacher-logits-v2',
        'shape': shape,
        'dtype': 'float16',
        'checkpoint_sha256': [sha256(p) for p in checkpoint_paths],
        'teacher_folds': n_folds,
        'checkpoint_folds': model_folds if n_folds else None,
        'dataset_meta_sha256': sha256(Path(args.dataset) / 'meta.json'),
    }, indent=2) + '\n')
    print(f'>> wrote {output} ({output.stat().st_size / 1024**3:.2f} GiB)', flush=True)


if __name__ == '__main__':
    main()
