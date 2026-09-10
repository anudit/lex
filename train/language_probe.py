"""
Does the model's pooled context already know what language it is reading?

Magika (ICSE 2025) shows content-type detection from a small byte sample is
essentially solved -- 99% F1 on text types, including the languages we are
weakest at (CSS 99%, PowerShell 99%, Perl 99%, Shell 98%, Markdown 92%). And a
language hint is worth +12.69 points to our own feature ceiling.

That suggests an auxiliary language-classification head rather than a new input:
if the pooled global context does not linearly encode language, an auxiliary loss
would force it to, at no inference cost and no bundle growth. If it already does,
the problem is the head failing to use it, and an auxiliary loss buys nothing.

This trains a linear probe on the frozen pooled context to find out which.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import data_pipeline as dp
from languages import TARGET_LANGUAGES
from model import LexerConfig, NeuralLexer
from train import pick_device, to_device


@torch.no_grad()
def collect(model: NeuralLexer, loader: DataLoader, device: torch.device, limit: int,
            stage: str = 'post'):
    """Pooled representation per window, plus its language.

    `stage='pre'` pools the embedding output, before any recurrent layer has run.
    That matters for design: a signal available there can condition every layer,
    whereas one that only appears after the layers can condition nothing but the
    head.
    """
    feats_out, langs = [], []
    for batch in loader:
        feats, _, valid, lang = to_device(batch, device)
        x = model.embedding(feats)
        if stage == 'post':
            for layer in model.layers:
                x = layer(x, valid)
        m = valid.unsqueeze(-1).to(x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(dim=1) / denom.squeeze(1)
        mx = x.masked_fill(~valid.unsqueeze(-1), float('-inf')).amax(dim=1)
        mx = torch.nan_to_num(mx, neginf=0.0)
        feats_out.append(torch.cat([mean, mx], dim=-1).cpu())
        langs.append(lang.cpu())
        if sum(f.shape[0] for f in feats_out) >= limit:
            break
    return torch.cat(feats_out).numpy(), torch.cat(langs).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./checkpoints/best_model.pt')
    ap.add_argument('--dataset', default='./corpus/dataset')
    ap.add_argument('--limit', type=int, default=20000)
    ap.add_argument('--stage', choices=('pre', 'post'), default='post',
                    help="'pre' pools before the recurrent layers, 'post' after")
    args = ap.parse_args()

    device = pick_device()
    ds, _ = dp.build(cache=args.dataset)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NeuralLexer(LexerConfig(**ck['config']) if 'config' in ck else LexerConfig()).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    model.set_quant(True)

    stage = args.stage
    tr_x, tr_y = collect(model, DataLoader(ds['train'], batch_size=64, shuffle=True,
                                           num_workers=2), device, args.limit, stage)
    va_x, va_y = collect(model, DataLoader(ds['val'], batch_size=64, num_workers=2),
                         device, args.limit, stage)

    # Linear probe: a single softmax layer on the frozen pooled representation.
    xt = torch.from_numpy(tr_x).float().to(device)
    yt = torch.from_numpy(tr_y).long().to(device)
    xv = torch.from_numpy(va_x).float().to(device)
    yv = torch.from_numpy(va_y).long().to(device)
    mu, sd = xt.mean(0, keepdim=True), xt.std(0, keepdim=True).clamp_min(1e-5)
    xt, xv = (xt - mu) / sd, (xv - mu) / sd

    probe = torch.nn.Linear(xt.shape[1], len(TARGET_LANGUAGES)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    for step in range(600):
        opt.zero_grad()
        loss = lossf(probe(xt), yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = probe(xv).argmax(-1)
        acc = (pred == yv).float().mean().item()
        majority = np.bincount(tr_y).max() / len(tr_y)

    print(f'\nlinear probe [{stage}-layers] on the frozen pooled context '
          f'({len(tr_y):,} train / {len(yv):,} val windows)')
    print(f'  language accuracy   {100 * acc:6.2f}%')
    print(f'  majority-class base {100 * majority:6.2f}%')
    print(f'  chance              {100 / len(TARGET_LANGUAGES):6.2f}%')

    print('\nper-language probe recall (the languages we classify worst first):')
    per_lang = ck['per_lang']
    order = sorted(range(len(TARGET_LANGUAGES)),
                   key=lambda i: per_lang.get(TARGET_LANGUAGES[i], 1.0))
    with torch.no_grad():
        pv = probe(xv).argmax(-1).cpu().numpy()
    for i in order[:12]:
        sel = yv.cpu().numpy() == i
        if sel.sum() < 5:
            continue
        r = (pv[sel] == i).mean()
        lang = TARGET_LANGUAGES[i]
        print(f'  {lang:12s} probe {100 * r:6.2f}%   '
              f'token acc {100 * per_lang.get(lang, 0):6.2f}%   n={int(sel.sum())}')


if __name__ == '__main__':
    main()
