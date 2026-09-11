"""Evaluate a large checkpoint, including teacher-subset and weakest-tail views."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import data_pipeline
from languages import LANGUAGE_META, TARGET_LANGUAGES
from model import LexerConfig, NeuralLexer

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'train'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./checkpoints_student/best_model.pt')
    parser.add_argument('--dataset', default='./corpus/dataset')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location('_lex_large_eval_train', BASE_DIR / 'train.py')
    assert spec and spec.loader
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)

    device = trainer.pick_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NeuralLexer(LexerConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    datasets, meta = data_pipeline.build(cache=args.dataset, total_tokens=None)
    loader = DataLoader(
        datasets[args.split], batch_size=args.batch_size, num_workers=args.workers)
    criterion = trainer.BoundaryWeightedCrossEntropy(
        trainer.class_weights(meta['label_counts'], device))
    result = trainer.evaluate(model, loader, device, criterion, quantized=True)

    print(f'{args.split}: weighted {100 * result["weighted"]:.2f}%  '
          f'macro {100 * result["macro"]:.2f}%  micro {100 * result["micro"]:.2f}%  '
          f'boundary {100 * result["boundary_f1"]:.2f}%')
    for engine in ('shiki', 'highlight.js'):
        languages = [
            language for language in TARGET_LANGUAGES
            if LANGUAGE_META[language]['teacher']['engine'] == engine
        ]
        scores = [result['per_lang'].get(language, 0.0) for language in languages]
        print(f'{engine:12s}: {100 * sum(scores) / len(scores):.2f}% macro '
              f'over {len(languages)} grammars')
    print('\nWeakest grammars:')
    for language, score in sorted(result['per_lang'].items(), key=lambda item: item[1])[:25]:
        engine = LANGUAGE_META[language]['teacher']['engine']
        print(f'  {language:18s} {100 * score:6.2f}%  {engine}')


if __name__ == '__main__':
    main()
