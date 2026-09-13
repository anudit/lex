"""Fast invariants for the large training target; no corpus is required."""

from __future__ import annotations

import tempfile
import re
from pathlib import Path

import torch

import data_pipeline
import tokenizer
from languages import (
    DEFAULT_TOTAL_TOKENS, MIN_TOKENS_PER_LANG, TARGET_LANGUAGES,
    token_budgets, weights,
)
from model import FIELD_SIZES, LexerConfig, NeuralLexer


def _rle_length(value: str) -> int:
    return sum(int(count) for count in re.findall(r'\d:(\d+)', value))


def main() -> None:
    assert len(TARGET_LANGUAGES) == 185
    assert abs(sum(weights().values()) - 1.0) < 1e-12
    budgets = token_budgets()
    assert sum(budgets.values()) == DEFAULT_TOTAL_TOKENS
    assert min(budgets.values()) >= MIN_TOKENS_PER_LANG
    assert tuple(FIELD_SIZES) == tuple(key for key in data_pipeline.FEATURE_KEYS if key != 'flags')

    code = "def f(x):\n    return x + 1  # comment\n"
    tokens = tokenizer.tokenize(code)
    arrays = tokenizer.tokens_to_arrays(tokens)
    features = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in arrays.items()
    }
    valid = torch.ones((1, len(tokens)), dtype=torch.bool)
    model = NeuralLexer()
    model.set_quant(True)
    with torch.no_grad():
        logits = model(features, valid)
    assert logits.shape == (1, len(tokens), 9)
    assert torch.isfinite(logits).all()

    size = model.size_report()
    assert size['packed_bytes'] == 110_352, size
    assert size['packed_bytes'] <= 111_000, size

    # Import through the shared exporter to verify that the wider feature table
    # and all four layers are represented by the binary format.
    import export
    with tempfile.TemporaryDirectory() as directory:
        report = export.export_model(model, directory)
        assert report['bytes'] == size['packed_bytes']
        assert Path(directory, 'weights.meta.json').exists()

    # Exercise both ground-truth engines and require exact character coverage.
    from build_labels import run_shard
    from languages import teacher
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        samples = {
            'javascript': 'const answer = "wide: 🚀"; // value\n',
            'abnf': 'rule = 1*ALPHA / "literal"\n',
        }
        entries = []
        for language, source in samples.items():
            path = root / f'{language}.txt'
            path.write_text(source)
            engine, target = teacher(language)
            entries.append({
                'sourceLanguage': language, 'teacher': engine,
                'language': target, 'path': str(path),
            })
        ok, dropped = run_shard(0, entries, root)
        assert (ok, dropped) == (2, 0)
        rows = [line.rstrip('\n').split('\t', 1) for line in (root / 'labels.0.tsv').open()]
        assert len(rows) == 2
        for path, encoded in rows:
            assert _rle_length(encoded) == len(Path(path).read_text())
        encoded_by_language = {
            Path(path).stem: encoded for path, encoded in rows
        }
        # Highlight.js 11 emitter nodes expose `scope`, not `kind`. Exact
        # length alone would let an all-PLAIN label stream pass unnoticed.
        highlight_rle = encoded_by_language['abnf']
        assert any(
            run.split(':', 1)[0] not in {'0', '9'}
            for run in highlight_rle.split(',')
        ), highlight_rle

    print('PASS: 185 languages')
    print(f'PASS: {sum(budgets.values()):,} tokens, floor {min(budgets.values()):,}')
    print(f'PASS: {size["total_parameters"]:,} params, {size["packed_kb"]:.2f} KiB')
    print(f'PASS: forward {tuple(logits.shape)} and export round-trip')
    print('PASS: exact-length Shiki and semantic Highlight.js labels')


if __name__ == '__main__':
    main()
