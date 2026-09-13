"""Export a large-model checkpoint to the shared packed binary format."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

from model import LexerConfig, NeuralLexer

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'train'


def main() -> None:
    parser = argparse.ArgumentParser(description='Export a train_large checkpoint')
    parser.add_argument('--checkpoint', default='./checkpoints_student/best_model.pt')
    parser.add_argument('--out', default='./checkpoints_student')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('requires_research_loader'):
        raise ValueError('This experimental checkpoint needs its research inference graph; production export is unsupported')
    config = LexerConfig(**checkpoint.get('config', {}))
    model = NeuralLexer(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load the shared serializer only after the large model owns the `model`
    # module name expected by export.py.
    spec = importlib.util.spec_from_file_location('_lex_shared_export', BASE_DIR / 'export.py')
    assert spec and spec.loader
    exporter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exporter
    spec.loader.exec_module(exporter)

    report = exporter.export_model(model, args.out)
    print(
        f">> exported {report['kb']:.2f} KiB to {Path(args.out).resolve()} "
        f"(round-trip max error {report['max_abs_error']:.2e})"
    )


if __name__ == '__main__':
    main()
