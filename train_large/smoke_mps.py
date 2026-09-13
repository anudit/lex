"""Matched, bounded architecture screening on real cached train/val windows.

No teacher, no test-split access, no production weight replacement. Stored NPZ
members are memory-mapped so sampling does not materialize the 18 GB archive.
Results measure early supervised token learning, not final browser accuracy.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data_pipeline as dp
from languages import TARGET_LANGUAGES, weights
from model import LexerConfig, NeuralLexer
from labels import MASK
from research_variants import BASE as RESEARCH_BASE, VARIANTS, ResearchLexer


CANDIDATES = {
    'legacy': dict(binary_ste=False),
    'baseline': {},
    'head2': dict(head_bits=2),
    'mixed': dict(input_bits=3, head_bits=2, output_bits=3),
    'mixed_erase': dict(input_bits=3, head_bits=2, output_bits=3, erase_rank=32),
    'mixed_kernel': dict(input_bits=3, head_bits=2, output_bits=3, kernel_size=7),
    'context': dict(input_bits=3, head_bits=2, output_bits=3, erase_rank=32, kernel_size=7),
    'row_head2': dict(embedding_scale='row', head_bits=2),
    'film128': dict(film_rank=128),
}


def mmap_member(path: Path, key: str) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(key + '.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError('Smoke loader requires the existing uncompressed NPZ cache')
    with path.open('rb') as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        name_len, extra_len = struct.unpack_from('<HH', header, 26)
        handle.seek(name_len + extra_len, 1)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f'Unsupported NPY header {version}')
        offset = handle.tell()
    return np.memmap(path, dtype=dtype, mode='r', offset=offset,
                     shape=shape, order='F' if fortran else 'C')


def subset(path: Path, per_lang: int, seed: int):
    langs = mmap_member(path, 'lang')
    offsets = mmap_member(path, 'offsets')
    rng = np.random.default_rng(seed)
    ids = np.concatenate([rng.choice(np.flatnonzero(langs == lang),
                                    min(per_lang, int((langs == lang).sum())), replace=False)
                          for lang in np.unique(langs)])
    # Copy only selected windows; preserve the original full 512-token context.
    selected = {k: [] for k in (*dp.FEATURE_KEYS, 'label')}
    for key in selected:
        array = mmap_member(path, key)
        selected[key] = np.concatenate([array[offsets[i]:offsets[i + 1]] for i in ids]).copy()
    lengths = offsets[ids + 1] - offsets[ids]
    ds = dp.LexerDataset(selected, np.r_[0, lengths.cumsum()], np.array(langs[ids]), 512)
    return ds, ids.tolist()


@torch.inference_mode()
def evaluate(model, ds, device, batch_size):
    model.eval()
    model.set_quant(True)
    hits = np.zeros(len(TARGET_LANGUAGES), dtype=np.int64)
    totals = hits.copy()
    class_hits = np.zeros(9, dtype=np.int64)
    class_totals = class_hits.copy()
    for batch in DataLoader(ds, batch_size=batch_size):
        feats = {k: batch[k].to(device) for k in dp.FEATURE_KEYS}
        pred = model(feats, batch['valid'].to(device)).argmax(-1).cpu().numpy()
        gold = batch['label'].numpy()
        valid = gold != MASK
        correct = (gold == pred) & valid
        np.add.at(hits, batch['lang'].numpy(), correct.sum(1))
        np.add.at(totals, batch['lang'].numpy(), valid.sum(1))
        class_totals += np.bincount(gold[valid], minlength=9)
        class_hits += np.bincount(gold[correct], minlength=9)
    scores = hits / np.maximum(totals, 1)
    weight = weights()
    return dict(weighted=sum(weight[l] * scores[i] for i, l in enumerate(TARGET_LANGUAGES)),
                micro=float(hits.sum() / max(totals.sum(), 1)),
                macro=float(scores.mean()), covered_languages=int((totals > 0).sum()),
                per_class=(class_hits / np.maximum(class_totals, 1)).tolist(),
                per_lang={l: float(scores[i]) for i, l in enumerate(TARGET_LANGUAGES)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, default=Path(__file__).parent / 'corpus/dataset')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--candidates', nargs='+', choices=[*CANDIDATES, *VARIANTS], default=list(CANDIDATES))
    ap.add_argument('--steps', type=int, default=160)
    ap.add_argument('--warmup', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--train-per-lang', type=int, default=16)
    ap.add_argument('--val-per-lang', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20260912)
    ap.add_argument('--data-seed', type=int, default=None,
                    help='fix sampled windows and batches across initialization seeds')
    args = ap.parse_args()
    if not 0 < args.warmup < args.steps:
        ap.error('Require 0 < warmup < steps')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; refusing to silently benchmark CPU')
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device('mps')
    torch.set_num_threads(4)
    # MPS embedding backward uses index_put accumulation, which has no fully
    # deterministic implementation. Seed everything and record this limitation.
    torch.use_deterministic_algorithms(True, warn_only=True)
    print('Loading stratified windows from stored NPZ members', flush=True)
    data_seed = args.seed if args.data_seed is None else args.data_seed
    dataset_manifest = json.loads((args.dataset / 'language_manifest.json').read_text())
    if tuple(sorted(item['id'] for item in dataset_manifest['languages'])) != TARGET_LANGUAGES:
        raise ValueError('Cached language IDs differ from the current training manifest')
    train_ds, train_ids = subset(args.dataset / 'train.npz', args.train_per_lang, data_seed)
    val_ds, val_ids = subset(args.dataset / 'val.npz', args.val_per_lang, data_seed + 1)
    # Every candidate gets the exact same sampled index sequence, with tempered
    # language weighting. All windows of a language have equal probability.
    counts = np.bincount(train_ds.langs, minlength=len(TARGET_LANGUAGES))
    lw = np.array([weights()[l] ** 0.5 for l in TARGET_LANGUAGES])
    p = lw[train_ds.langs] / counts[train_ds.langs]
    p /= p.sum()
    order = np.random.default_rng(data_seed + 2).choice(len(train_ds),
                (args.steps, args.batch_size), p=p)
    meta = json.loads((args.dataset / 'meta.json').read_text())
    manifest = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    torch=torch.__version__, device='mps', distillation=False,
                    auxiliary_losses={'language': 0.15, 'structure': 0.2},
                    research_variants={name: VARIANTS[name] for name in args.candidates if name in VARIANTS},
                    deterministic='warn_only: MPS index accumulation is nondeterministic',
                    dataset_meta_sha256=hashlib.sha256((args.dataset / 'meta.json').read_bytes()).hexdigest(),
                    train_ids=train_ids, val_ids=val_ids,
                    batch_order_sha256=hashlib.sha256(order.tobytes()).hexdigest())
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    # Import the shared loss without invoking either CLI entry point.
    import importlib.util
    spec = importlib.util.spec_from_file_location('_smoke_trainer', Path(__file__).parents[1] / 'train/train.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    results = []
    torch.manual_seed(args.seed)
    # Freeze the original initialization even when the production default changes.
    common = NeuralLexer(LexerConfig()).state_dict()
    for name in args.candidates:
        variant = VARIANTS.get(name, {})
        torch.manual_seed(args.seed)
        cfg = LexerConfig(**(RESEARCH_BASE | variant.get('config', {}) if name in VARIANTS else CANDIDATES[name]))
        model = ResearchLexer(cfg, repeats=variant.get('repeats', 1),
                              full_precision=variant.get('full_precision', False),
                              embedding_rms_clip=variant.get('embedding_rms_clip', False))
        state = model.state_dict()
        model.load_state_dict({k: common[k] if k in common and common[k].shape == v.shape else v
                               for k, v in state.items()})
        packed = model.size_report()['packed_bytes']
        assert packed <= 111000, (name, packed)
        model.to(device)
        from labels import CLASS_NAMES
        freq = torch.tensor([max(1, meta['label_counts'].get(n, 0)) for n in CLASS_NAMES], dtype=torch.float32)
        cw = (freq.sum() / freq) ** variant.get('class_exponent', 0.5)
        criterion = trainer.BoundaryWeightedCrossEntropy((cw / cw.mean()).to(device))
        torch.manual_seed(args.seed + 10)
        lang_head = torch.nn.Linear(cfg.dim * 2, len(TARGET_LANGUAGES)).to(device)
        struct_head = torch.nn.Linear(cfg.dim, trainer.NUM_STRUCT_BITS).to(device)
        predictor = torch.nn.Linear(cfg.dim, cfg.dim).to(device) if variant.get('jepa') else None
        ema = copy.deepcopy(model).eval() if predictor is not None else None
        if ema is not None:
            ema.requires_grad_(False)
        parameters = list(model.parameters()) + list(lang_head.parameters()) + list(struct_head.parameters())
        if predictor is not None:
            parameters += list(predictor.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=0.003, weight_decay=0.01)
        mask_rng = torch.Generator().manual_seed(args.seed + 20)
        records = []
        start = time.monotonic()
        storage = (f'{4 * sum(p.numel() for p in model.parameters())} FP32 bytes (diagnostic only)'
                   if model.full_precision else f'{packed} packed bytes')
        print(f'{name}: {storage}; {args.steps} steps, {args.warmup} FP', flush=True)
        for step, indices in enumerate(order, 1):
            model.train()
            model.set_quant(step > args.warmup)
            batch = torch.utils.data.default_collate([train_ds[int(i)] for i in indices])
            feats, label, valid, _ = trainer.to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, signature, token_repr = model(feats, valid, return_signature=True)
            loss = criterion(logits, label)
            loss = loss + 0.15 * torch.nn.functional.cross_entropy(lang_head(signature), batch['lang'].to(device))
            target = trainer.structural_targets(feats, label)
            mask = valid.unsqueeze(-1).expand_as(target)
            loss = loss + 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(
                struct_head(token_repr)[mask], target[mask])
            if ema is not None and step > args.warmup:
                # Predict contextual EMA targets at masked word positions.
                # This is a small data2vec/JEPA-inspired auxiliary experiment,
                # not an implementation of the published transformer systems.
                selected = ((torch.rand(label.shape, generator=mask_rng) < 0.15).to(device)
                            & valid & (feats['kind'] == 0))
                masked = {k: torch.where(selected, torch.zeros_like(v), v)
                          for k, v in feats.items()}
                # Remove direct lexical identity at the selected word positions.
                masked['kind'] = torch.where(selected, torch.ones_like(feats['kind']), feats['kind'])
                _, _, predicted = model(masked, valid, return_signature=True)
                ema.set_quant(True)
                with torch.no_grad():
                    _, _, latent = ema(feats, valid, return_signature=True)
                    latent = F.layer_norm(latent, (cfg.dim,))
                latent_loss = F.smooth_l1_loss(predictor(predicted), latent, reduction='none').mean(-1)
                loss = loss + variant['jepa'] * (latent_loss * selected).sum() / selected.sum().clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    for target_param, source_param in zip(ema.parameters(), model.parameters()):
                        target_param.lerp_(source_param, 0.01)
            progress = max(0, step - args.warmup) / (args.steps - args.warmup)
            for group in optimizer.param_groups:
                group['lr'] = 0.0003 + 0.0027 * (1 + np.cos(np.pi * progress)) / 2
            if step % 200 == 0:
                value = float(loss.detach())
                if not np.isfinite(value):
                    raise RuntimeError(f'Nonfinite loss: {name} step {step}')
                print(f'{name} step {step}: loss={value:.4f} elapsed={time.monotonic()-start:.1f}s', flush=True)
            if step in (args.warmup, args.steps // 2, args.steps):
                metrics = evaluate(model, val_ds, device, args.batch_size)
                records.append(dict(step=step, **metrics))
                print(f'{name} val {step}: weighted={metrics["weighted"]:.4%} micro={metrics["micro"]:.4%}', flush=True)
        torch.mps.synchronize()
        elapsed = time.monotonic() - start
        model.cpu()
        inference_special = model.repeats != 1 or model.full_precision or model.embedding_rms_clip
        torch.save(dict(config=asdict(cfg), model_state_dict=model.state_dict(),
                        research_variant=variant,
                        requires_research_loader=inference_special), args.out / f'{name}.pt')
        result = dict(name=name, packed_bytes=None if model.full_precision else packed,
                      fp32_weight_bytes=4 * sum(p.numel() for p in model.parameters()) if model.full_precision else None,
                      seconds=elapsed, records=records, variant=variant)
        results.append(result)
        (args.out / 'results.json').write_text(json.dumps(results, indent=2))
        del optimizer, model, ema, predictor, parameters
        torch.mps.empty_cache()
    print(json.dumps([{k: v for k, v in r.items() if k != 'records'} | {'weighted': r['records'][-1]['weighted']}
                      for r in results], indent=2), flush=True)


if __name__ == '__main__':
    main()
