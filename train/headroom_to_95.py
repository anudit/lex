"""
What it would take to reach 95% weighted agreement.

Splits the remaining error three ways so effort goes where the points are:
  * how much sits in the head languages vs the long tail
  * how much is a single confusion pair the model keeps making
  * how much is left after the languages with known-bad corpora are fixed
"""

from __future__ import annotations

import argparse

import torch

from languages import TARGET_LANGUAGES, TOP25_PUSHER_SHARE, weights as lang_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./checkpoints/best_model.pt')
    ap.add_argument('--target', type=float, default=0.95)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    acc = ck['per_lang']
    w = lang_weights()
    cur = sum(w[l] * acc.get(l, 0.0) for l in TARGET_LANGUAGES)
    gap = args.target - cur

    print(f'current {100 * cur:.2f}%   target {100 * args.target:.0f}%   '
          f'gap {100 * gap:.2f} points\n')

    # Points available per language: its weight times how far it is from target.
    loss = sorted(((w[l] * max(0.0, args.target - a), l, a) for l, a in acc.items()),
                  reverse=True)
    print('Points recoverable per language, if lifted to the target:')
    print(f'  {"language":12s} {"acc":>7s} {"weight":>7s} {"points":>7s} {"cumulative":>11s}')
    cum = 0.0
    for pts, lang, a in loss[:14]:
        cum += pts
        print(f'  {lang:12s} {100 * a:6.1f}% {100 * w[lang]:6.2f}% '
              f'{100 * pts:6.2f}  {100 * cum:10.2f}')

    head = [l for l in acc if l in TOP25_PUSHER_SHARE]
    tail = [l for l in acc if l not in TOP25_PUSHER_SHARE]
    hp = sum(w[l] * max(0.0, args.target - acc[l]) for l in head)
    tp = sum(w[l] * max(0.0, args.target - acc[l]) for l in tail)
    print(f'\n  top-25 languages    {100 * hp:5.2f} points recoverable '
          f'({100 * sum(w[l] for l in head):.1f}% of weight)')
    print(f'  tail languages      {100 * tp:5.2f} points recoverable '
          f'({100 * sum(w[l] for l in tail):.1f}% of weight)')

    # Two languages carry a third of the weight; nothing else can substitute.
    big = sorted(((w[l], l, acc[l]) for l in acc), reverse=True)[:3]
    print('\n  the three heaviest languages:')
    for wt, l, a in big:
        print(f'    {l:12s} weight {100 * wt:5.2f}%  acc {100 * a:5.1f}%  '
              f'-> {100 * wt * max(0.0, args.target - a):.2f} points to target')

    print('\nWhat each scenario reaches:')
    scenarios = {
        'fix the 6 worst tail languages to 90%':
            [l for _, l, a in loss if l not in TOP25_PUSHER_SHARE and acc[l] < 0.90][:6],
        'lift every language below 85% to 90%':
            [l for l in acc if acc[l] < 0.85],
    }
    for name, langs in scenarios.items():
        new = dict(acc)
        for l in langs:
            new[l] = max(new[l], 0.90)
        got = sum(w[l] * new.get(l, 0.0) for l in TARGET_LANGUAGES)
        print(f'  {name:42s} -> {100 * got:.2f}%')

    # Ceiling if every language matched the best one currently achieved.
    best = max(acc.values())
    ideal = sum(w[l] * best for l in TARGET_LANGUAGES if l in acc)
    print(f'  every language at the current best ({100 * best:.1f}%)      -> {100 * ideal:.2f}%')


if __name__ == '__main__':
    main()
