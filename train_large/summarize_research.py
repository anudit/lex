"""Summarize completed smoke runs without mixing different corpus protocols."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from languages import weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    args = ap.parse_args()
    groups = defaultdict(list)
    protocols = set()
    for root in args.runs:
        manifest = json.loads((root / 'manifest.json').read_text())
        config = manifest['args']
        protocols.add((manifest['dataset_meta_sha256'], config['steps'], config['warmup'],
                       config['batch_size'], tuple(manifest['val_ids']),
                       manifest['batch_order_sha256']))
        for result in json.loads((root / 'results.json').read_text()):
            groups[result['name']].append((config['seed'], result))
    if len(protocols) != 1:
        raise ValueError('These runs use different datasets, batches, validation windows or schedules')
    rows = []
    for name, results in groups.items():
        metrics = [result['records'][-1] for _, result in results]
        rows.append(dict(name=name, seeds=[seed for seed, _ in results],
                         packed_bytes=results[0][1]['packed_bytes'],
                         weighted_mean=float(np.mean([r['weighted'] for r in metrics])),
                         weighted_std=float(np.std([r['weighted'] for r in metrics])),
                         micro_mean=float(np.mean([r['micro'] for r in metrics])),
                         macro_mean=float(np.mean([r['macro'] for r in metrics])),
                         seconds_mean=float(np.mean([r['seconds'] for _, r in results])),
                         class_recall_mean=np.mean([r['per_class'] for r in metrics], axis=0).tolist()))
    rows.sort(key=lambda r: -r['weighted_mean'])
    print(json.dumps(rows, indent=2))
    if 'r_control' in groups:
        per_lang = groups['r_control'][0][1]['records'][-1]['per_lang']
        deficit = sorted(((lang, weights()[lang] * (1 - acc), acc) for lang, acc in per_lang.items()),
                         key=lambda x: -x[1])[:12]
        print('Largest weighted error contributions in control:', json.dumps(deficit))


if __name__ == '__main__':
    main()
