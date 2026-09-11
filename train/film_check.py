"""
Is the FiLM conditioning actually being used?

`strength` initializes to zero, which makes conditioning exactly the identity, so
the model can ignore it entirely and nothing in the loss would say so. This
reports how far each layer has moved away from that, and how much the resulting
scale and shift actually vary across languages -- a channel whose gamma is the
same for Markdown and C is not conditioning on anything.
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
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./checkpoints/best_model.pt')
    ap.add_argument('--dataset', default='./corpus/dataset')
    ap.add_argument('--windows', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    device = pick_device()
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NeuralLexer(LexerConfig(**ck['config']) if 'config' in ck else LexerConfig()).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    model.set_quant(True)

    st = model.film.strength.detach().cpu().numpy()
    print(f'\ncheckpoint epoch {ck["epoch"]}   FiLM strength (0 = conditioning off)')
    for i, row in enumerate(st):
        print(f'  layer {i}   mean |s| {np.abs(row).mean():.4f}   '
              f'max |s| {np.abs(row).max():.4f}   '
              f'channels above 0.01: {int((np.abs(row) > 0.01).sum())}/{len(row)}')

    ds, _ = dp.build(cache=args.dataset)
    loader = DataLoader(ds['val'], batch_size=64, num_workers=args.workers)
    per_lang: dict[int, list[np.ndarray]] = {}
    codes: list[np.ndarray] = []
    seen = 0
    for batch in loader:
        feats, _, valid, lang = to_device(batch, device)
        sig = model.doc_signature(feats, valid)
        # The low-rank code, to check it is not saturated. A unit pinned at +/-1
        # for every document carries no information no matter how large the
        # per-channel strength grows.
        codes.append(torch.tanh(model.film.down(model.film.norm(sig))).cpu().numpy())
        films = model.film(sig)
        gam = torch.cat([g.squeeze(1) for g, _ in films], dim=-1).cpu().numpy()
        for row, l in zip(gam, lang.cpu().numpy()):
            per_lang.setdefault(int(l), []).append(row)
        seen += gam.shape[0]
        if seen >= args.windows:
            break

    C = np.concatenate(codes)
    print(f'\nlow-rank code (tanh of the projected signature), n={len(C)} windows')
    print(f'  saturated (|tanh| > 0.99) {100 * (np.abs(C) > 0.99).mean():5.1f}% of units')
    print(f'  spread across documents   {C.std(0).mean():.4f}')

    means = {l: np.mean(v, axis=0) for l, v in per_lang.items() if len(v) >= 5}
    if len(means) < 2:
        print('\nnot enough per-language windows to compare')
        return
    stack = np.stack(list(means.values()))
    between = stack.std(axis=0).mean()
    within = np.mean([np.std(v, axis=0).mean() for l, v in per_lang.items() if len(v) >= 5])
    print(f'\ngamma spread across languages (between) {between:.4f}')
    print(f'gamma spread within a language  (within)  {within:.4f}')
    print(f'ratio between/within {between / max(1e-9, within):.2f}  '
          f'(>1 means the conditioning is language-specific)')

    # The languages whose conditioning is most unlike the average.
    avg = stack.mean(axis=0)
    dist = {TARGET_LANGUAGES[l]: float(np.abs(m - avg).mean()) for l, m in means.items()}
    print('\nmost distinctly conditioned languages:')
    for lang, d in sorted(dist.items(), key=lambda kv: -kv[1])[:8]:
        print(f'  {lang:12s} {d:.4f}')

    _ablate(model, ds, device, ck, args.workers)


@torch.no_grad()
def _ablate(model, ds, device, ck, workers: int) -> None:
    """Upper bound on FiLM's contribution: switch it off at inference.

    This is not the same as training without FiLM -- the model has adapted to it,
    so switching it off mid-flight hurts more than never having had it. That makes
    the drop an *upper* bound: if it is small, FiLM cannot be worth its bytes, and
    a full ablation run is unnecessary.
    """
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from train import evaluate

    if model.film is None:
        print('\nno FiLM in this checkpoint')
        return

    loader = DataLoader(ds['val'], batch_size=32, num_workers=workers)
    crit = nn.CrossEntropyLoss(ignore_index=dp.MASK)

    on = evaluate(model, loader, device, crit)
    saved = model.film.strength.detach().clone()
    model.film.strength.zero_()          # gamma = 1, beta = 0 -> exact identity
    off = evaluate(model, loader, device, crit)
    model.film.strength.copy_(saved)

    print(f'\nFiLM switched off at inference (upper bound on its contribution)')
    print(f'  with FiLM     weighted {100 * on["weighted"]:.2f}%  micro {100 * on["micro"]:.2f}%')
    print(f'  without       weighted {100 * off["weighted"]:.2f}%  micro {100 * off["micro"]:.2f}%')
    print(f'  drop          {100 * (on["weighted"] - off["weighted"]):+.2f} points')

    deltas = sorted(((on['per_lang'][l] - off['per_lang'].get(l, 0.0), l)
                     for l in on['per_lang']), reverse=True)
    print('\n  languages that rely on it most:')
    for d, l in deltas[:6]:
        print(f'    {l:12s} {100 * d:+6.2f}  ({100 * on["per_lang"][l]:.1f}% with, '
              f'{100 * off["per_lang"].get(l, 0):.1f}% without)')


if __name__ == '__main__':
    main()
