"""
Final evaluation on the held-out *test* split, plus a span demo.

The test split is untouched by training and by the checkpoint selection that used
the val split, so this is the number to quote for the model in isolation. The
head-to-head against other engines lives in the demo, because gpu-lexer needs a
browser.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

import data_pipeline as dp
import tokenizer
from labels import CLASS_NAMES
from languages import TOP25_PUSHER_SHARE, weights as lang_weights
from model import LexerConfig, NeuralLexer
from train import evaluate, pick_device

SAMPLES = {
    'javascript': 'export async function fetchUsers(ids = []) {\n'
                  '  const MAX = 3.14;\n'
                  '  return ids.map((id) => `#${id}`); // done\n}',
    'python': 'from typing import Iterable\n\n'
              'def total(xs: Iterable[int]) -> int:\n'
              '    """Sum them."""\n'
              '    return sum(xs) + MAX_N',
    'rust': 'pub fn counts(text: &str) -> HashMap<&str, usize> {\n'
            '    let mut m = HashMap::new();  // tally\n'
            '    for w in text.split_whitespace() { *m.entry(w).or_insert(0) += 1; }\n'
            '    m\n}',
}


def spans(model: NeuralLexer, code: str, device: torch.device) -> list[dict]:
    """Merge adjacent same-class tokens into renderable spans."""
    toks = tokenizer.tokenize(code)
    arrays = tokenizer.tokens_to_arrays(toks)
    feats = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in arrays.items()}
    model.eval()
    model.set_quant(True)
    with torch.no_grad():
        preds = model(feats, torch.ones(1, len(toks), dtype=torch.bool, device=device))
        preds = preds.argmax(-1).squeeze(0).cpu().numpy()

    out: list[dict] = []
    cur = None
    for tok, c in zip(toks, preds):
        if tok.kind in (1, 2):
            continue
        name = CLASS_NAMES[int(c)]
        if cur and cur['type'] == name:
            cur['end'] = tok.end
        else:
            if cur:
                out.append(cur)
            cur = {'type': name, 'start': tok.start, 'end': tok.end}
    if cur:
        out.append(cur)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./checkpoints/best_model.pt')
    ap.add_argument('--dataset', default='./corpus/dataset')
    ap.add_argument('--split', default='test')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    device = pick_device()
    ds, meta = dp.build(cache=args.dataset)
    loader = DataLoader(ds[args.split], batch_size=args.batch_size,
                        num_workers=args.workers)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NeuralLexer(LexerConfig(**ck['config']) if 'config' in ck else LexerConfig()).to(device)
    model.load_state_dict(ck['model_state_dict'])

    crit = torch.nn.CrossEntropyLoss(ignore_index=dp.MASK)
    res = evaluate(model, loader, device, crit)

    print(f'\n{args.split} split ({len(ds[args.split]):,} windows), '
          f'checkpoint from epoch {ck["epoch"]}')
    print(f'  weighted agreement with Shiki  {100 * res["weighted"]:.2f}%')
    print(f'  unweighted (micro)             {100 * res["micro"]:.2f}%')

    print('\nper class:')
    for name, acc in res['per_class'].items():
        print(f'  {name:9s} {100 * acc:6.2f}%')

    w = lang_weights()
    print('\nper language (* = GitHub top 25):')
    for lang, acc in sorted(res['per_lang'].items(), key=lambda kv: -w[kv[0]]):
        star = '*' if lang in TOP25_PUSHER_SHARE else ' '
        print(f' {star}{lang:12s} {100 * acc:6.2f}%   weight {100 * w[lang]:5.2f}%')

    print('\nspans:')
    for lang, code in SAMPLES.items():
        print(f'\n--- {lang} ---')
        for s in spans(model, code, device):
            text = code[s['start']:s['end']].replace('\n', '\\n')
            print(f"  {s['type']:9s} {text!r}")


if __name__ == '__main__':
    main()
