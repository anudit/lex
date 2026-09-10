"""
Measures the gap between what we train and what we are scored on.

We train per-token cross-entropy with inverse-sqrt class weights. The benchmark
scores per non-whitespace *character*, unweighted by class, weighted by language
popularity. Those are three separate mismatches, and each one is a place where
the model is being optimized for something nobody measures.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

import data_pipeline as dp
from labels import CLASS_NAMES, MASK, NUM_CLASSES
from languages import TARGET_LANGUAGES, weights as lang_weights
from model import LexerConfig, NeuralLexer
from train import pick_device, to_device


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./checkpoints/best_model.pt')
    ap.add_argument('--dataset', default='./corpus/dataset')
    args = ap.parse_args()

    device = pick_device()
    ds, meta = dp.build(cache=args.dataset)
    loader = DataLoader(ds['val'], batch_size=32, num_workers=2)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NeuralLexer(LexerConfig(**ck['config']) if 'config' in ck else LexerConfig()).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    model.set_quant(True)

    # Token length is recoverable from the length bucket: bucket b covers
    # [2^b, 2^(b+1)), so 2^b is a lower bound and is enough to rank them.
    tok_hit = tok_tot = 0
    chr_hit = chr_tot = 0
    by_len = defaultdict(lambda: [0, 0])
    by_class = defaultdict(lambda: [0, 0, 0, 0])  # tok_hit, tok_tot, chr_hit, chr_tot

    for batch in loader:
        feats, labels, valid, _ = to_device(batch, device)
        preds = model(feats, valid).argmax(-1)
        m = labels != MASK
        correct = (preds == labels) & m
        lb = feats['len_bucket']
        # Approximate character weight from the bucket midpoint.
        w = torch.where(lb == 0, 1.0, (1.5 * torch.pow(2.0, lb.float())))
        w = w * m.float()

        tok_hit += correct.sum().item()
        tok_tot += m.sum().item()
        chr_hit += (correct.float() * w).sum().item()
        chr_tot += w.sum().item()

        lb_c, cor_c, m_c, w_c = lb.cpu(), correct.cpu(), m.cpu(), w.cpu()
        lab_c = labels.cpu()
        for b in range(8):
            sel = (lb_c == b) & m_c
            by_len[b][0] += cor_c[sel].sum().item()
            by_len[b][1] += sel.sum().item()
        for c in range(NUM_CLASSES):
            sel = (lab_c == c) & m_c
            by_class[c][0] += cor_c[sel].sum().item()
            by_class[c][1] += sel.sum().item()
            by_class[c][2] += (cor_c[sel].float() * w_c[sel]).sum().item()
            by_class[c][3] += w_c[sel].sum().item()

    print(f'\nval, {tok_tot:,} supervised tokens')
    print(f'  per-token accuracy      {100 * tok_hit / tok_tot:6.2f}%   <- what we train on')
    print(f'  per-character accuracy  {100 * chr_hit / chr_tot:6.2f}%   <- what the benchmark scores')
    print(f'  gap                     {100 * (tok_hit / tok_tot - chr_hit / chr_tot):+6.2f} pts')

    print('\naccuracy by token length (long tokens carry more characters)')
    print(f'  {"bucket":8s} {"~chars":>7s} {"tokens":>10s} {"share":>7s} {"acc":>7s}')
    for b in sorted(by_len):
        h, t = by_len[b]
        if not t:
            continue
        approx = 1 if b == 0 else int(1.5 * 2 ** b)
        print(f'  {b:8d} {approx:7d} {t:10,d} {100 * t / tok_tot:6.1f}% {100 * h / t:6.2f}%')

    print('\nper class: token accuracy vs character-weighted share')
    print(f'  {"class":9s} {"tok share":>9s} {"chr share":>9s} {"tok acc":>8s} {"chr acc":>8s}')
    for c in range(NUM_CLASSES):
        h, t, ch, ct = by_class[c]
        if not t:
            continue
        print(f'  {CLASS_NAMES[c]:9s} {100 * t / tok_tot:8.2f}% {100 * ct / chr_tot:8.2f}% '
              f'{100 * h / t:7.2f}% {100 * ch / ct:7.2f}%')


if __name__ == '__main__':
    main()
