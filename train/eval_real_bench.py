"""
Evaluate a checkpoint against gpu-lexer's own held-out verification corpus,
using gpu-lexer's own scoring methodology (token-level exact match, confidence-
gated, weighted by GitHub pusher share of the top 25 languages).

This exists because lex's own val/test split -- built from the same corpus and
sampling era as train -- can silently drift from what the public gpu-lexer
benchmark actually reports. Scoring against the real corpus with the real
methodology closes that gap: the number this script prints is the number that
should show up on the website, not a proxy for it. `RealBenchEval` is also
imported by train.py so checkpoint selection during training is driven by this
same metric, not just lex's own val split.

Requires the gpu-lexer repo checked out alongside this one and its
verification shard already built (`pnpm corpus:fetch -- --split verification
&& pnpm corpus:build -- --split verification` in packages/training there).
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import tokenizer
from data_pipeline import align
from labels import CLASS_NAMES, MASK

DEFAULT_GPU_LEXER_ROOT = Path(__file__).resolve().parents[2] / 'gpu-lexer'


def _load_verification(shard_path: Path) -> list[dict]:
    items = []
    with gzip.open(shard_path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _char_classes_from_source_labels(source: str, source_labels: list[dict]) -> np.ndarray:
    """gpu-lexer's sourceLabels are [{from, to, class, confidence?}, ...],
    disjoint and sorted. Expand to a per-character class array, matching
    decode_rle's convention (9 = masked/uncovered)."""
    out = np.full(len(source), 9, dtype=np.uint8)
    for label in source_labels:
        if label.get('confidence') == 0:
            continue
        cls = CLASS_NAMES.index(label['class'])
        out[label['from']:label['to']] = cls
    return out


class RealBenchEval:
    """Loads gpu-lexer's verification shard once; `evaluate()` scores any
    checkpoint's model against it cheaply (no re-parsing the shard)."""

    def __init__(self, gpu_lexer_root: str | Path = DEFAULT_GPU_LEXER_ROOT, top: int = 25):
        root = Path(gpu_lexer_root)
        popularity = json.loads(
            (root / 'packages/training/data/language-popularity.json').read_text())
        self.top_languages = [l for l in popularity['languages']
                               if not l.get('supplemental')][:top]
        self.top_families = {l['family'] for l in self.top_languages}
        self.total_pushers = sum(l['pushers'] for l in self.top_languages)

        shard_path = root / 'packages/training/data/generated/shards/verification.jsonl.gz'
        raw_items = _load_verification(shard_path)
        # Pre-tokenize and pre-align once: this is the expensive part (~1100
        # files) and is identical on every call, so do it only at construction.
        self.examples = []
        for item in raw_items:
            family = item.get('family')
            if family not in self.top_families:
                continue
            source = item['source']
            char_cls = _char_classes_from_source_labels(source, item['sourceLabels'])
            toks = tokenizer.tokenize(source)
            if not toks:
                continue
            gold = align(toks, char_cls)
            if (gold != MASK).sum() == 0:
                continue
            self.examples.append((family, toks, gold))

    @torch.no_grad()
    def evaluate(self, model, device) -> dict:
        was_training = model.training
        model.eval()
        per_family = defaultdict(lambda: {'correct': 0, 'total': 0})
        for family, toks, gold in self.examples:
            arrays = tokenizer.tokens_to_arrays(toks)
            feats = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in arrays.items()}
            valid = torch.ones(1, len(toks), dtype=torch.bool, device=device)
            logits = model(feats, valid)
            preds = logits.argmax(-1).squeeze(0).cpu().numpy()

            mask = gold != MASK
            bucket = per_family[family]
            bucket['correct'] += int((preds[mask] == gold[mask]).sum())
            bucket['total'] += int(mask.sum())
        if was_training:
            model.train()

        weighted = 0.0
        per_lang = {}
        for lang in self.top_languages:
            family = lang['family']
            weight = lang['pushers'] / self.total_pushers
            bucket = per_family.get(family, {'correct': 0, 'total': 0})
            acc = bucket['correct'] / max(1, bucket['total'])
            weighted += weight * acc
            per_lang[family] = {'accuracy': acc, 'n': bucket['total'], 'weight': weight}
        micro_correct = sum(b['correct'] for b in per_family.values())
        micro_total = sum(b['total'] for b in per_family.values())
        return {
            'weighted': weighted,
            'micro': micro_correct / max(1, micro_total),
            'per_lang': per_lang,
            'files': len(self.examples),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./checkpoints_hero3_cont/best_model.pt')
    ap.add_argument('--gpu-lexer-root', default=str(DEFAULT_GPU_LEXER_ROOT))
    ap.add_argument('--top', type=int, default=25)
    args = ap.parse_args()

    from model import LexerConfig, NeuralLexer
    from train import pick_device

    bench = RealBenchEval(args.gpu_lexer_root, args.top)
    print(f'loaded {len(bench.examples)} verification items '
          f'across {len(bench.top_families)} languages')

    device = pick_device()
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NeuralLexer(LexerConfig(**ck['config']) if 'config' in ck else LexerConfig()).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.set_quant(True)

    result = bench.evaluate(model, device)
    for lang in sorted(bench.top_languages, key=lambda l: -l['pushers']):
        family = lang['family']
        info = result['per_lang'][family]
        print(f"  {family:14s} {100 * info['accuracy']:6.2f}%  (n={info['n']:6d}, "
              f"weight {100 * info['weight']:5.2f}%)")

    print(f"\nreal-bench weighted accuracy: {100 * result['weighted']:.2f}%")
    print(f"real-bench micro accuracy:    {100 * result['micro']:.2f}%")


if __name__ == '__main__':
    main()
