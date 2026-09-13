"""Export isolated smoke checkpoints and build real-browser parity fixtures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import LexerConfig, NeuralLexer
import export
import tokenizer
from reference_large import Reference
from wgsl_large import generate, pipeline_order
from test_architecture import CASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    suite = []
    for checkpoint in sorted(args.checkpoints.glob('*.pt')):
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck.get('requires_research_loader'):
            print(f'Skipping {checkpoint.name}: experimental inference graph')
            continue
        model = NeuralLexer(LexerConfig(**ck['config'])).eval()
        model.load_state_dict(ck['model_state_dict'])
        target = args.out / checkpoint.stem
        report = export.export_model(model, str(target))
        (target / 'shader.wgsl').write_text(generate(report['meta']))
        meta, planes, f16 = export.load_weights(str(target))
        reference = Reference(str(target))
        cases = []
        for code in CASES:
            arrays = tokenizer.tokens_to_arrays(tokenizer.tokenize(code))
            table = reference.W('embedding.table')
            emb = np.zeros((len(arrays['kind']), model.cfg.embed_dim), dtype=np.float32)
            for field, offset in meta['field_offsets'].items():
                value = table[offset + arrays[field]]
                if field in ('hash1', 'hash2'):
                    value = value * (arrays['kind'] == 0)[:, None]
                emb += value
            for bit in range(8):
                emb += ((arrays['flags'] >> bit) & 1)[:, None] * table[meta['flag_offset'] + bit]
            emb = emb @ reference.W('embedding.up').T + reference.B('embedding.up')
            cases.append(dict(code=code, classes=reference.forward(arrays).argmax(-1).tolist(),
                              embedding=emb.ravel().tolist()))
        suite.append(dict(name=checkpoint.stem, planes=planes.tolist(), fp=f16.astype(np.float32).tolist(),
                          steps=pipeline_order(model.cfg.n_layers), cases=cases))
    (args.out / 'suite.json').write_text(json.dumps(suite))
    print(f'Prepared {len(suite)} candidate exports in {args.out}')


if __name__ == '__main__':
    main()
