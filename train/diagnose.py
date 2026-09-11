"""
Attributes the remaining error to its causes, so capacity decisions are made on
evidence rather than on the assumption that a bigger model is a better one.

Two questions:
  1. Is the model converged, or just undertrained?
  2. Where does the residual error actually land? -- confusion matrix, and which
     languages cost the most weighted accuracy.

Note on what is *not* measured here: running the trained checkpoint with the
quantizers switched off does not measure "the cost of quantization". After QAT
the latent weights are only meaningful through the quantizer -- their scale is
absorbed into it -- so evaluating them raw gives a different model, not a
full-precision version of this one. Measuring the quantization cost properly
needs a separately trained full-precision run.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data_pipeline as dp
from labels import CLASS_NAMES, MASK, NUM_CLASSES
from languages import TARGET_LANGUAGES, weights as lang_weights
from model import LexerConfig, NeuralLexer
from train import pick_device, to_device


@torch.no_grad()
def run(model, loader, device, quant):
    model.eval()
    model.set_quant(quant)
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    n_lang = len(TARGET_LANGUAGES)
    hit = torch.zeros(n_lang)
    tot = torch.zeros(n_lang)
    for batch in loader:
        feats, labels, valid, lang = to_device(batch, device)
        preds = model(feats, valid).argmax(-1)
        m = labels != MASK
        correct = (preds == labels) & m
        lc = lang.cpu()
        hit.index_add_(0, lc, correct.sum(1).float().cpu())
        tot.index_add_(0, lc, m.sum(1).float().cpu())
        vl = labels[m].cpu()
        vp = preds[m].cpu()
        idx = vl * NUM_CLASSES + vp
        conf.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    per_lang = {TARGET_LANGUAGES[i]: (hit[i] / tot[i]).item()
                for i in range(n_lang) if tot[i] > 0}
    w = lang_weights()
    return {
        'weighted': sum(w[l] * per_lang.get(l, 0.0) for l in TARGET_LANGUAGES),
        'micro': (hit.sum() / tot.sum()).item(),
        'conf': conf,
        'per_lang': per_lang,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./checkpoints/best_model.pt')
    ap.add_argument('--dataset', default='./corpus/dataset')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    device = pick_device()
    ds, meta = dp.build(cache=args.dataset)
    val = DataLoader(ds['val'], batch_size=args.batch_size,
                     num_workers=args.workers)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NeuralLexer(LexerConfig(**ck['config']) if 'config' in ck else LexerConfig()).to(device)
    model.load_state_dict(ck['model_state_dict'])

    q = run(model, val, device, quant=True)
    print(f'\nQuantized model as shipped: weighted {100 * q["weighted"]:.2f}%  '
          f'micro {100 * q["micro"]:.2f}%')

    print('\n1. Convergence')
    hist = json.load(open(os.path.join(os.path.dirname(args.checkpoint), 'history.json')))
    qh = [h for h in hist if h['quant']]
    print('   last 5 QAT epochs (val loss / weighted):')
    for h in qh[-5:]:
        print(f'     epoch {h["epoch"]:2d}  loss {h["loss"]:.4f}  weighted {100 * h["weighted"]:.2f}%')
    trend = qh[-1]['loss'] - qh[-2]['loss']
    print(f'   val loss still {"falling" if trend < 0 else "flat/rising"} '
          f'({trend:+.4f} on the last epoch)')

    print('\n2. Where the error lands (rows = truth, quantized model)')
    conf = q['conf'].numpy()
    tot = conf.sum(1)
    print(f'   {"truth":9s} {"n":>8s} {"recall":>7s}   top confusions')
    order = np.argsort(-conf.sum(1))
    for i in order:
        if tot[i] == 0:
            continue
        row = conf[i].copy()
        rec = row[i] / tot[i]
        row[i] = 0
        top = np.argsort(-row)[:3]
        conf_s = ', '.join(f'{CLASS_NAMES[j]} {100 * row[j] / tot[i]:.1f}%'
                           for j in top if row[j] > 0)
        print(f'   {CLASS_NAMES[i]:9s} {tot[i]:8,d} {100 * rec:6.1f}%   {conf_s}')

    print('\n3. Weakest languages by weighted contribution to the loss')
    w = lang_weights()
    losses = sorted(((w[l] * (1 - a), l, a) for l, a in q['per_lang'].items()),
                    reverse=True)
    for contrib, l, a in losses[:8]:
        print(f'   {l:12s} acc {100 * a:5.1f}%  weight {100 * w[l]:5.2f}%  '
              f'costs {100 * contrib:.2f} pts')


if __name__ == '__main__':
    main()
