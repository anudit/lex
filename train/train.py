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
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import data_pipeline as dp
import export
from eval_real_bench import RealBenchEval, DEFAULT_GPU_LEXER_ROOT as _REAL_BENCH_DEFAULT_ROOT
from labels import CLASS_NAMES, MASK, NUM_CLASSES
from languages import (
    TARGET_LANGUAGES, TRAIN_EXCLUDED_LANGUAGES, benchmark_family_language,
    weights as lang_weights,
)
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


class BoundaryWeightedCrossEntropy(nn.Module):
    """Class-balanced CE with extra pressure at lexical state transitions."""

    def __init__(self, weight: torch.Tensor, boundary_boost: float = 1.0,
                 label_smoothing: float = 0.03):
        super().__init__()
        self.register_buffer('class_weight', weight)
        self.boundary_boost = boundary_boost
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat_loss = F.cross_entropy(
            logits.reshape(-1, NUM_CLASSES), labels.reshape(-1),
            ignore_index=MASK, weight=self.class_weight,
            label_smoothing=self.label_smoothing, reduction='none').view_as(labels)
        valid = labels != MASK
        boundary, _, _ = _boundary_masks(labels, labels)
        token_weight = 1.0 + self.boundary_boost * boundary.to(flat_loss.dtype)
        return (flat_loss * token_weight * valid).sum() / valid.sum().clamp_min(1)


def _boundary_masks(labels: torch.Tensor, preds: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Boundary-adjacent tokens and transitions over consecutive scored tokens."""
    valid = labels != MASK
    B, T = labels.shape
    positions = torch.arange(T, device=labels.device).expand(B, T)
    seen = torch.where(valid, positions, positions.new_full((), -1))
    previous = torch.cummax(seen, dim=1).values
    previous = torch.cat([previous.new_full((B, 1), -1), previous[:, :-1]], dim=1)
    has_previous = valid & (previous >= 0)
    previous_safe = previous.clamp_min(0)
    previous_gold = labels.gather(1, previous_safe)
    previous_pred = preds.gather(1, previous_safe)
    gold = has_previous & (labels != previous_gold)
    predicted = has_previous & (preds != previous_pred)

    adjacent = gold.to(torch.int32)
    adjacent.scatter_add_(1, previous_safe, gold.to(torch.int32))
    return adjacent > 0, gold, predicted


def _boundary_counts(labels: torch.Tensor, preds: torch.Tensor) -> tuple[int, int, int]:
    _, gold, predicted = _boundary_masks(labels, preds)
    return (int((gold & predicted).sum()), int(predicted.sum()), int(gold.sum()))


def construct_window_multipliers(dataset: dp.LexerDataset, boost: float) -> np.ndarray:
    """Boost windows containing a complete comment or string run."""
    out = np.ones(len(dataset), dtype=np.float64)
    if boost <= 0:
        return out
    from labels import COMMENT, STRING
    labels = dataset.flat['label']
    for i in range(len(dataset)):
        a, b = int(dataset.offsets[i]), int(dataset.offsets[i + 1])
        scored = labels[a:b]
        scored = scored[scored != MASK]
        complete = 0
        for cls in (COMMENT, STRING):
            inside = scored == cls
            if inside.size >= 3:
                starts = inside & np.r_[True, ~inside[:-1]]
                ends = inside & np.r_[~inside[1:], True]
                complete += int(min(starts.sum(), ends.sum()))
        out[i] += boost * min(1.0, complete / 2.0)
    return out


def sampling_probabilities(real_bench: RealBenchEval | None, exponent: float,
                           tail_floor: float, include_excluded: bool) -> np.ndarray:
    """Desired per-language draw probability before dividing by window count."""
    active = np.array([
        include_excluded or lang not in TRAIN_EXCLUDED_LANGUAGES
        for lang in TARGET_LANGUAGES
    ], dtype=bool)
    raw = np.zeros(len(TARGET_LANGUAGES), dtype=np.float64)
    index = {lang: i for i, lang in enumerate(TARGET_LANGUAGES)}
    if real_bench is not None:
        for item in real_bench.top_languages:
            lang = benchmark_family_language(item['family'])
            if lang in index:
                raw[index[lang]] += item['pushers'] / real_bench.total_pushers
    else:
        fallback = lang_weights()
        raw[:] = [fallback[lang] for lang in TARGET_LANGUAGES]
    raw[active] = np.maximum(raw[active], tail_floor)
    raw[active] **= exponent
    raw[~active] = 0.0
    return raw / raw.sum()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def to_device(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    feats = {k: v.to(device, non_blocking=True)
             for k, v in batch.items() if k in dp.FEATURE_KEYS}
    return (feats, batch['label'].to(device), batch['valid'].to(device),
            batch['lang'].to(device))


# Bracket characters as they appear in `first_char`/`last_char`: those fields are
# the raw ASCII code for any character below 128 (see tokenizer.char_bucket), so
# a single-character symbol token's identity is already sitting in the feature
# tensors gpu-lexer's tree model gets this signal -- comment-state, string-state,
# bracket depth -- as an explicit multi-task target predicted from the hidden
# state and fed back into the classifier. Reproducing that here costs no new
# labels or tokenizer changes: comment/string continuity comes from the existing
# gold labels shifted by one token, and bracket depth is a deterministic
# cumulative count over symbol tokens already in the batch.
_OPEN = {40: 0, 123: 1, 91: 2}    # ( { [
_CLOSE = {41: 0, 125: 1, 93: 2}  # ) } ]


def structural_targets(feats: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """[B, T, 8] float target: prev-token in-comment, prev-token in-string, then
    depth>=1 and depth>=2 buckets for parens/curlies/squares."""
    kind = feats['kind']
    first_char = feats['first_char']
    is_symbol = kind == 3
    B, T = kind.shape
    device = kind.device

    delta = torch.zeros(3, B, T, device=device)
    for char, bracket in _OPEN.items():
        delta[bracket] += ((first_char == char) & is_symbol).float()
    for char, bracket in _CLOSE.items():
        delta[bracket] -= ((first_char == char) & is_symbol).float()
    depth = delta.cumsum(dim=-1).clamp(min=0)  # [3, B, T]

    from labels import COMMENT, STRING
    prev_comment = torch.zeros(B, T, device=device)
    prev_string = torch.zeros(B, T, device=device)
    prev_comment[:, 1:] = (labels[:, :-1] == COMMENT).float()
    prev_string[:, 1:] = (labels[:, :-1] == STRING).float()

    return torch.stack([
        prev_comment, prev_string,
        (depth[0] >= 1).float(), (depth[0] >= 2).float(),
        (depth[1] >= 1).float(), (depth[1] >= 2).float(),
        (depth[2] >= 1).float(), (depth[2] >= 2).float(),
    ], dim=-1)


NUM_STRUCT_BITS = 8


@torch.no_grad()
def evaluate(model: NeuralLexer, loader: DataLoader, device: torch.device,
             criterion: nn.Module, quantized: bool = True) -> dict:
    model.eval()
    model.set_quant(quantized)
    n_lang = len(TARGET_LANGUAGES)
    hit = torch.zeros(n_lang)
    tot = torch.zeros(n_lang)
    cls_hit = torch.zeros(NUM_CLASSES)
    cls_tot = torch.zeros(NUM_CLASSES)
    boundary_tp = boundary_pred = boundary_gold = 0
    loss_sum = 0.0
    batches = 0

    for batch in loader:
        feats, labels, valid, lang = to_device(batch, device)
        logits = model(feats, valid)
        if isinstance(criterion, BoundaryWeightedCrossEntropy):
            loss_sum += criterion(logits, labels).item()
        else:
            loss_sum += criterion(
                logits.reshape(-1, NUM_CLASSES), labels.reshape(-1)).item()
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
        tp, npred, ngold = _boundary_counts(labels, preds)
        boundary_tp += tp
        boundary_pred += npred
        boundary_gold += ngold
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
        'boundary_f1': (2 * boundary_tp / max(1, boundary_pred + boundary_gold)),
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
    ap.add_argument('--film-rank', type=int, default=32,
                    help='rank of the FiLM projection; 0 disables conditioning')
    ap.add_argument('--erase-rank', type=int, default=8,
                    help='rank of each layer selective erase projection; 0 uses legacy reset')
    ap.add_argument('--sampler', choices=('equal', 'tempered'), default='tempered')
    ap.add_argument('--sampling-exponent', type=float, default=0.5,
                    help='0 is equal-language sampling, 1 is natural benchmark weighting')
    ap.add_argument('--tail-floor', type=float, default=0.001,
                    help='minimum pre-tempering probability for non-benchmark languages')
    ap.add_argument('--include-excluded-languages', action='store_true',
                    help='train all 57 languages instead of the promoted 52-language set')
    ap.add_argument('--construct-boost', type=float, default=1.0,
                    help='sampling boost for windows with complete string/comment runs')
    ap.add_argument('--boundary-boost', type=float, default=1.0,
                    help='extra CE weight on tokens adjacent to a gold class transition')
    ap.add_argument('--calibration-fraction', type=float, default=0.2,
                    help='final fraction of epochs sampled at natural benchmark weights')
    ap.add_argument('--teacher-checkpoint', default='',
                    help='optional wider teacher checkpoint for logit distillation')
    ap.add_argument('--distill-weight', type=float, default=0.2)
    ap.add_argument('--distill-temperature', type=float, default=2.0)
    ap.add_argument('--full-precision', action='store_true',
                    help='train and select a non-shipping float teacher without QAT')
    ap.add_argument('--seed', type=int, default=20260911)
    ap.add_argument('--lang-loss', type=float, default=0.2,
                    help='weight of the auxiliary language loss; 0 disables it')
    ap.add_argument('--struct-loss', type=float, default=0.2,
                    help='weight of the auxiliary structural-state loss '
                         '(comment/string continuity, bracket depth); 0 disables it')
    ap.add_argument('--mine-weak-languages', action='store_true',
                    help='after each epoch, boost sampling weight for languages the '
                         'model is currently weakest on (up to 3x), on top of the '
                         'configured base sampling policy')
    ap.add_argument('--real-bench-root', default=str(_REAL_BENCH_DEFAULT_ROOT),
                    help='path to a gpu-lexer checkout with its verification shard '
                         'built; when it exists, checkpoint selection uses accuracy '
                         'on that real corpus instead of the internal val split, '
                         'since that split can drift from what the public benchmark '
                         'reports. Pass an empty string to fall back to val-only '
                         'selection.')
    ap.add_argument('--dim', type=int, default=64,
                    help='per-layer recurrent hidden width')
    ap.add_argument('--embed-dim', type=int, default=32,
                    help='embedding table column width')
    ap.add_argument('--head-hidden', type=int, default=96,
                    help='classifier MLP hidden width')
    ap.add_argument('--resume', default='',
                    help='continue training an existing checkpoint (its own config wins '
                         'over --film-rank/--dim/--embed-dim/--head-hidden). Skips FP '
                         'warmup -- the loaded model is already quantization-aware -- and '
                         'uses a plain decay-only schedule from --lr instead of a fresh '
                         'OneCycleLR ramp, since restarting the ramp on an already-'
                         'converged quantized model would push it away from its optimum '
                         'before re-converging.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f'>> device: {device}')

    ds, meta = dp.build(cache=args.dataset, total_tokens=args.total_tokens)
    print(f">> corpus: {meta['total_tokens']:,} tokens, {meta['files']:,} files, "
          f"{meta['languages_covered']}/{meta['languages_target']} languages "
          f"({100 * meta['weight_covered']:.1f}% of eval weight)")
    print(f">> windows: {len(ds['train']):,} train / {len(ds['val']):,} val / "
          f"{len(ds['test']):,} test  (split by file)")

    real_bench = None
    if args.real_bench_root and os.path.exists(args.real_bench_root):
        real_bench = RealBenchEval(args.real_bench_root)
        print(f'>> real-bench: {len(real_bench.examples)} files from '
              f'{args.real_bench_root} -- checkpoint selection uses this, not val')
    elif args.real_bench_root:
        print(f'>> real-bench root {args.real_bench_root} not found -- '
              f'falling back to val-only checkpoint selection')

    train_lang_counts = np.bincount(ds['train'].langs, minlength=len(TARGET_LANGUAGES))
    exponent = 0.0 if args.sampler == 'equal' else args.sampling_exponent
    desired = sampling_probabilities(real_bench, exponent, args.tail_floor,
                                     args.include_excluded_languages)
    construct_mult = construct_window_multipliers(ds['train'], args.construct_boost)

    def window_weights(language_probs: np.ndarray) -> np.ndarray:
        per_lang = language_probs / np.maximum(train_lang_counts, 1)
        return per_lang[ds['train'].langs] * construct_mult

    base_per_window_weight = window_weights(desired)
    train_sampler = WeightedRandomSampler(
        torch.from_numpy(base_per_window_weight).double(),
        num_samples=len(ds['train']), replacement=True,
        generator=torch.Generator().manual_seed(args.seed))
    active_names = [lang for lang, p in zip(TARGET_LANGUAGES, desired) if p > 0]
    excluded = sorted(set(TARGET_LANGUAGES) - set(active_names))
    print(f">> sampler: {args.sampler}, {len(active_names)} training languages, "
          f"exponent {exponent:.2f}, construct boost {args.construct_boost:.2f}")
    if excluded:
        print(f">> training exclusions (still evaluated): {', '.join(excluded)}")

    train_loader = DataLoader(ds['train'], batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.workers, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(ds['val'], batch_size=args.batch_size,
                            num_workers=args.workers)

    resume_ck = None
    if args.resume:
        resume_ck = torch.load(args.resume, map_location=device, weights_only=False)
        model = NeuralLexer(LexerConfig(**resume_ck['config'])).to(device)
        model.load_state_dict(resume_ck['model_state_dict'])
        print(f">> resumed {args.resume} (was weighted {100 * resume_ck['weighted']:.2f}%)")
    else:
        model = NeuralLexer(LexerConfig(film_rank=args.film_rank, erase_rank=args.erase_rank,
                                        dim=args.dim,
                                        embed_dim=args.embed_dim,
                                        head_hidden=args.head_hidden)).to(device)
    size = model.size_report()
    print(f">> model: {size['total_parameters']:,} params, "
          f"{size['packed_kb']:.2f} KB packed (gpu-lexer: 41,321 params, 31.0 KB)")

    bench_root = Path(args.real_bench_root) if args.real_bench_root else None
    npm_dist = Path(__file__).resolve().parents[1] / 'demo/node_modules/gpu-lexer/dist/index.js'
    npm_manifest = npm_dist.parent.parent / 'package.json'
    run_manifest = {
        'format': 'lex-training-run-v1',
        'args': vars(args),
        'model': size,
        'dataset': {
            'total_tokens': meta['total_tokens'],
            'files': meta['files'],
            'languages_target': meta['languages_target'],
            'meta_sha256': sha256_file(Path(args.dataset) / 'meta.json'),
        },
        'training_languages': active_names,
        'excluded_but_evaluated': excluded,
        'initial_language_probabilities': {
            lang: float(prob) for lang, prob in zip(TARGET_LANGUAGES, desired) if prob > 0
        },
        'benchmark': {
            'verification_sha256': sha256_file(
                bench_root / 'packages/training/data/generated/shards/verification.jsonl.gz')
                if bench_root else None,
            'popularity_sha256': sha256_file(
                bench_root / 'packages/training/data/language-popularity.json')
                if bench_root else None,
            'npm_version': (json.loads(npm_manifest.read_text()).get('version')
                            if npm_manifest.exists() else None),
            'npm_dist_sha256': sha256_file(npm_dist),
        },
    }
    with open(os.path.join(args.out_dir, 'run_manifest.json'), 'w') as handle:
        json.dump(run_manifest, handle, indent=2)

    cw = class_weights(meta['label_counts'], device)
    criterion = BoundaryWeightedCrossEntropy(cw, boundary_boost=args.boundary_boost)

    teacher = None
    if args.teacher_checkpoint:
        teacher_ck = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
        teacher = NeuralLexer(LexerConfig(**teacher_ck['config'])).to(device)
        teacher.load_state_dict(teacher_ck['model_state_dict'])
        teacher.eval()
        teacher.set_quant(False)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        print(f'>> distilling from {args.teacher_checkpoint} '
              f'(weight {args.distill_weight}, temperature {args.distill_temperature})')

    # Training-only auxiliary heads. Their parameters are optimized alongside the
    # model but are never part of it.
    lang_head = nn.Linear(model.cfg.dim * 2, len(TARGET_LANGUAGES)).to(device)
    lang_criterion = nn.CrossEntropyLoss()
    struct_head = nn.Linear(model.cfg.dim, NUM_STRUCT_BITS).to(device)
    struct_criterion = nn.BCEWithLogitsLoss()

    steps_per_epoch = args.max_steps or len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(lang_head.parameters()) + list(struct_head.parameters()),
        lr=args.lr, weight_decay=0.01)
    if resume_ck is not None:
        # No ramp-up: the loaded model is already past warmup, and a fresh
        # OneCycleLR climbing back to max_lr would shove it away from wherever
        # QAT had settled before it gets a chance to re-anneal back down.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.15)

    if resume_ck is None:
        best = -1.0
    elif real_bench is not None:
        # A resumed checkpoint's stored 'weighted' is a val-split score, not
        # comparable to real-bench selection; re-baseline against the real
        # bench once instead of starting from an unrelated number.
        best = real_bench.evaluate(model, device)['weighted']
        print(f'>> resumed checkpoint scores {100 * best:.2f}% on the real bench')
    else:
        best = resume_ck['weighted']
    history = []
    for epoch in range(1, args.epochs + 1):
        quant = not args.full_precision and (resume_ck is not None or epoch > args.warmup_epochs)
        model.train()
        model.set_quant(quant)
        t0 = time.time()
        run_loss = 0.0
        run_aux = 0.0
        run_struct = 0.0
        aux_hit = aux_tot = 0
        struct_hit = struct_tot = 0
        hit = tot = 0
        for step, batch in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            feats, labels, valid, lang = to_device(batch, device)
            want_sig = args.lang_loss > 0
            want_struct = args.struct_loss > 0
            out = model(feats, valid, return_signature=want_sig or want_struct)
            logits, sig, token_repr = out if (want_sig or want_struct) else (out, None, None)
            loss = criterion(logits, labels)
            if teacher is not None and args.distill_weight > 0:
                with torch.no_grad():
                    teacher_logits = teacher(feats, valid)
                temp = args.distill_temperature
                soft = F.kl_div(
                    F.log_softmax(logits / temp, dim=-1),
                    F.softmax(teacher_logits / temp, dim=-1),
                    reduction='none').sum(-1)
                mask = labels != MASK
                distill = soft[mask].mean() * (temp * temp)
                loss = loss + args.distill_weight * distill
            if want_sig:
                aux = lang_criterion(lang_head(sig), lang)
                loss = loss + args.lang_loss * aux
                run_aux += aux.item()
                aux_hit += (lang_head(sig).argmax(-1) == lang).sum().item()
                aux_tot += lang.numel()
            if want_struct:
                struct_target = structural_targets(feats, labels)
                struct_logits = struct_head(token_repr)
                m3 = valid.unsqueeze(-1).expand_as(struct_target)
                struct = struct_criterion(struct_logits[m3], struct_target[m3])
                loss = loss + args.struct_loss * struct
                run_struct += struct.item()
                with torch.no_grad():
                    struct_hit += (((struct_logits > 0) == (struct_target > 0.5)) & m3).sum().item()
                    struct_tot += m3.sum().item()
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

        val = evaluate(model, val_loader, device, criterion,
                       quantized=not args.full_precision)
        mode = 'FP teacher' if args.full_precision else ('1-bit QAT' if quant else 'FP warmup')
        # The auxiliary language accuracy is reported because the FiLM
        # conditioning downstream is only as good as the signature it reads.
        aux_str = (f' lang {100 * aux_hit / max(1, aux_tot):.1f}%'
                   if aux_tot else '')
        struct_str = (f' struct {100 * struct_hit / max(1, struct_tot):.1f}%'
                      if struct_tot else '')

        bench_result = None
        bench_str = ''
        if real_bench is not None and (quant or args.full_precision):
            bench_result = real_bench.evaluate(model, device)
            bench_str = (f' | REAL-BENCH {100 * bench_result["weighted"]:.2f}% '
                         f'micro {100 * bench_result["micro"]:.2f}% '
                         f'boundary {100 * bench_result["boundary_f1"]:.2f}%')

        print(f'Epoch {epoch:2d}/{args.epochs} [{mode}] {time.time() - t0:.0f}s '
              f'train_loss {run_loss / steps_per_epoch:.4f} '
              f'train_acc {100 * hit / max(1, tot):.1f}%{aux_str}{struct_str} | '
                       f'val_loss {val["loss"]:.4f} '
                      f'WEIGHTED {100 * val["weighted"]:.2f}% micro {100 * val["micro"]:.2f}% '
                      f'boundary {100 * val["boundary_f1"]:.2f}%{bench_str}',
              flush=True)
        history.append({'epoch': epoch, 'quant': quant,
                        **{k: v for k, v in val.items() if k != 'per_lang'},
                        **({'real_bench_weighted': bench_result['weighted'],
                            'real_bench_micro': bench_result['micro']}
                           if bench_result is not None else {})})

        calibration_start = max(1, math.ceil(args.epochs * (1.0 - args.calibration_fraction)))
        if args.sampler == 'tempered' and epoch == calibration_start:
            desired = sampling_probabilities(real_bench, 1.0, args.tail_floor,
                                             args.include_excluded_languages)
            base_per_window_weight = window_weights(desired)
            train_sampler.weights = torch.from_numpy(base_per_window_weight).double()
            print('   sampler: switched to natural benchmark calibration weights')

        if args.mine_weak_languages:
            lang_index = {l: i for i, l in enumerate(TARGET_LANGUAGES)}
            mult = np.ones(len(TARGET_LANGUAGES))
            for lang, acc in val['per_lang'].items():
                i = lang_index.get(lang)
                if i is not None:
                    mult[i] = np.clip(2.0 - 2.0 * acc, 0.5, 3.0)
            new_weight = base_per_window_weight * mult[ds['train'].langs]
            train_sampler.weights = torch.from_numpy(new_weight).double()
            active = set(active_names)
            worst = sorted(
                ((lang, acc) for lang, acc in val['per_lang'].items() if lang in active),
                key=lambda kv: kv[1])[:5]
            print(f"   mining: boosted active languages "
                  f"{', '.join(f'{l} ({100 * a:.0f}%)' for l, a in worst)}")

        # Only checkpoint from the quantized regime: a full-precision model that
        # scores well says nothing about what ships. Selection uses the real
        # bench when available -- that is the number that actually gets
        # published -- and falls back to the internal val split otherwise.
        selection_score = bench_result['weighted'] if bench_result is not None else val['weighted']
        if (quant or args.full_precision) and selection_score > best:
            best = selection_score
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'weighted': val['weighted'], 'micro': val['micro'],
                        'per_lang': val['per_lang'], 'per_class': val['per_class'],
                        'real_bench_weighted': bench_result['weighted'] if bench_result else None,
                        'real_bench_micro': bench_result['micro'] if bench_result else None,
                        'train_args': vars(args),
                        'config': vars(model.cfg)},
                       os.path.join(args.out_dir, 'best_model.pt'))
            label = 'real-bench' if bench_result is not None else 'val'
            print(f'   -> saved best ({label} {100 * best:.2f}%)')

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
    label = 'real-bench weighted accuracy' if real_bench is not None else 'weighted agreement with Shiki (val)'
    print(f">> best {label}: {100 * best:.2f}% (gpu-lexer's own Prism.js reference: 84.19%)")


if __name__ == '__main__':
    main()
