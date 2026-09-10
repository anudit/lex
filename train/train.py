"""
Training loop for the neural lexer.

Two things here are not standard boilerplate and matter for the result:

  * The reported metric is popularity-weighted agreement with Shiki, matching the
    benchmark the model is judged on: per-language accuracy combined under
    GitHub pusher-count weights. A corpus-shaped average would let a large,
    easy language mask failures in a heavily-weighted one.
  * Class weights are inverse-sqrt frequency. `number` and `constant` are ~1% of
    tokens each; unweighted, the model can drop both entirely and still look
    fine on raw accuracy while producing visibly wrong highlighting.

Quantization is annealed rather than switched on: a full-precision warmup finds
a good basin, then QAT sharpens the weights onto the grid. Flipping straight to
1-bit from random init wastes most of the run recovering.

An auxiliary language-classification head is attached to the model's document
signature during training only. A linear probe showed that signature is already
84% language-separable by accident; supervising it makes that explicit, which is
what the FiLM conditioning downstream depends on. The head lives here rather than
in the model so it costs nothing at inference and never reaches the exported file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import data_pipeline as dp
import export
from labels import CLASS_NAMES, MASK, NUM_CLASSES
from languages import TARGET_LANGUAGES, weights as lang_weights
from model import LexerConfig, NeuralLexer


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def class_weights(counts: dict[str, int], device: torch.device) -> torch.Tensor:
    freq = torch.tensor([max(1, counts.get(n, 0)) for n in CLASS_NAMES],
                        dtype=torch.float32)
    w = (freq.sum() / freq).sqrt()
    return (w / w.mean()).to(device)


def to_device(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    feats = {k: v.to(device, non_blocking=True)
             for k, v in batch.items() if k in dp.FEATURE_KEYS}
    return (feats, batch['label'].to(device), batch['valid'].to(device),
            batch['lang'].to(device))


@torch.no_grad()
def evaluate(model: NeuralLexer, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> dict:
    model.eval()
    model.set_quant(True)  # always score the model as it will actually ship
    n_lang = len(TARGET_LANGUAGES)
    hit = torch.zeros(n_lang)
    tot = torch.zeros(n_lang)
    cls_hit = torch.zeros(NUM_CLASSES)
    cls_tot = torch.zeros(NUM_CLASSES)
    loss_sum = 0.0
    batches = 0

    for batch in loader:
        feats, labels, valid, lang = to_device(batch, device)
        logits = model(feats, valid)
        loss_sum += criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1)).item()
        batches += 1
        preds = logits.argmax(-1)
        mask = labels != MASK
        correct = (preds == labels) & mask
        # Per-language tallies: one row of the batch is one language.
        lang_cpu = lang.cpu()
        hit.index_add_(0, lang_cpu, correct.sum(1).float().cpu())
        tot.index_add_(0, lang_cpu, mask.sum(1).float().cpu())
        vl = labels[mask].cpu()
        vp = preds[mask].cpu()
        cls_tot.index_add_(0, vl, torch.ones_like(vl, dtype=torch.float))
        cls_hit.index_add_(0, vl, (vp == vl).float())

    per_lang = {TARGET_LANGUAGES[i]: (hit[i] / tot[i]).item()
                for i in range(n_lang) if tot[i] > 0}
    w = lang_weights()
    # Unsupported languages score zero, exactly as the reference benchmark does.
    weighted = sum(w[l] * per_lang.get(l, 0.0) for l in TARGET_LANGUAGES)
    micro = (hit.sum() / tot.sum().clamp_min(1)).item()
    return {
        'loss': loss_sum / max(1, batches),
        'weighted': weighted,
        'micro': micro,
        'per_lang': per_lang,
        'per_class': {CLASS_NAMES[i]: (cls_hit[i] / cls_tot[i]).item()
                      for i in range(NUM_CLASSES) if cls_tot[i] > 0},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='./corpus/dataset')
    ap.add_argument('--total-tokens', type=int, default=None,
                    help='corpus size to slice; defaults to whatever is already '
                         'cached at --dataset, or 24M for a fresh cache')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--warmup-epochs', type=int, default=3,
                    help='full-precision epochs before quantization-aware training')
    ap.add_argument('--out-dir', default='./checkpoints')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-steps', type=int, default=0, help='0 = full epochs')
    ap.add_argument('--film-rank', type=int, default=48,
                    help='rank of the FiLM projection; 0 disables conditioning')
    ap.add_argument('--lang-loss', type=float, default=0.2,
                    help='weight of the auxiliary language loss; 0 disables it')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device()
    print(f'>> device: {device}')

    ds, meta = dp.build(cache=args.dataset, total_tokens=args.total_tokens)
    print(f">> corpus: {meta['total_tokens']:,} tokens, {meta['files']:,} files, "
          f"{meta['languages_covered']}/{meta['languages_target']} languages "
          f"({100 * meta['weight_covered']:.1f}% of eval weight)")
    print(f">> windows: {len(ds['train']):,} train / {len(ds['val']):,} val / "
          f"{len(ds['test']):,} test  (split by file)")

    # Token budgets are popularity-weighted, so high-resource languages (JS, Python,
    # TypeScript, ...) fill most of every epoch by raw window count. Left alone, that
    # starves low-resource languages of gradient signal even when their own token
    # budget is unchanged -- shared capacity gets bid away by whoever has more windows.
    # Weight each window by the inverse size of its language's window count so every
    # language gets roughly equal expected exposure per epoch.
    train_lang_counts = np.bincount(ds['train'].langs, minlength=len(TARGET_LANGUAGES))
    per_window_weight = 1.0 / np.maximum(train_lang_counts[ds['train'].langs], 1)
    train_sampler = WeightedRandomSampler(
        torch.from_numpy(per_window_weight).double(),
        num_samples=len(ds['train']), replacement=True)

    train_loader = DataLoader(ds['train'], batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.workers, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(ds['val'], batch_size=args.batch_size,
                            num_workers=args.workers)

    model = NeuralLexer(LexerConfig(film_rank=args.film_rank)).to(device)
    size = model.size_report()
    print(f">> model: {size['total_parameters']:,} params, "
          f"{size['packed_kb']:.2f} KB packed (gpu-lexer: 41,321 params, 31.0 KB)")

    cw = class_weights(meta['label_counts'], device)
    criterion = nn.CrossEntropyLoss(ignore_index=MASK, weight=cw, label_smoothing=0.03)

    # Training-only auxiliary head. Its parameters are optimized alongside the
    # model but are never part of it.
    lang_head = nn.Linear(model.cfg.dim * 2, len(TARGET_LANGUAGES)).to(device)
    lang_criterion = nn.CrossEntropyLoss()

    steps_per_epoch = args.max_steps or len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(lang_head.parameters()),
        lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.15)

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        quant = epoch > args.warmup_epochs
        model.train()
        model.set_quant(quant)
        t0 = time.time()
        run_loss = 0.0
        run_aux = 0.0
        aux_hit = aux_tot = 0
        hit = tot = 0
        for step, batch in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            feats, labels, valid, lang = to_device(batch, device)
            want_sig = args.lang_loss > 0
            out = model(feats, valid, return_signature=want_sig)
            logits, sig = out if want_sig else (out, None)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))
            if want_sig:
                aux = lang_criterion(lang_head(sig), lang)
                loss = loss + args.lang_loss * aux
                run_aux += aux.item()
                aux_hit += (lang_head(sig).argmax(-1) == lang).sum().item()
                aux_tot += lang.numel()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
            with torch.no_grad():
                m = labels != MASK
                hit += ((logits.argmax(-1) == labels) & m).sum().item()
                tot += m.sum().item()
            if step % 200 == 0:
                print(f'   epoch {epoch} step {step}/{steps_per_epoch} '
                      f'loss {run_loss / (step + 1):.4f} acc {100 * hit / max(1, tot):.1f}%',
                      flush=True)

        val = evaluate(model, val_loader, device, criterion)
        mode = '1-bit QAT' if quant else 'FP warmup'
        # The auxiliary language accuracy is reported because the FiLM
        # conditioning downstream is only as good as the signature it reads.
        aux_str = (f' lang {100 * aux_hit / max(1, aux_tot):.1f}%'
                   if aux_tot else '')
        print(f'Epoch {epoch:2d}/{args.epochs} [{mode}] {time.time() - t0:.0f}s '
              f'train_loss {run_loss / steps_per_epoch:.4f} '
              f'train_acc {100 * hit / max(1, tot):.1f}%{aux_str} | '
              f'val_loss {val["loss"]:.4f} '
              f'WEIGHTED {100 * val["weighted"]:.2f}% micro {100 * val["micro"]:.2f}%',
              flush=True)
        history.append({'epoch': epoch, 'quant': quant, **{
            k: v for k, v in val.items() if k != 'per_lang'}})

        # Only checkpoint from the quantized regime: a full-precision model that
        # scores well says nothing about what ships.
        if quant and val['weighted'] > best:
            best = val['weighted']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'weighted': val['weighted'], 'micro': val['micro'],
                        'per_lang': val['per_lang'], 'per_class': val['per_class'],
                        'config': vars(model.cfg)},
                       os.path.join(args.out_dir, 'best_model.pt'))
            print(f'   -> saved best (weighted {100 * best:.2f}%)')

    with open(os.path.join(args.out_dir, 'history.json'), 'w') as fh:
        json.dump(history, fh, indent=2)

    best_ckpt_path = os.path.join(args.out_dir, 'best_model.pt')
    if os.path.exists(best_ckpt_path):
        ck = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        if 'config' in ck:
            model = NeuralLexer(LexerConfig(**ck['config'])).to(device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    rep = export.export_model(model, args.out_dir)
    print(f"\n>> exported {rep['kb']:.2f} KB, round-trip max error {rep['max_abs_error']:.2e}")
    print(f">> best weighted agreement with Shiki: {100 * best:.2f}% "
          f"(gpu-lexer reference: 90.20%)")


if __name__ == '__main__':
    main()
