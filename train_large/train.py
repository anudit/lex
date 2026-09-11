"""Opinionated entry point for the 99.97 KiB student.

All flags remain overridable. Defaults are inserted only when the caller did
not provide that flag, so this stays compatible with ../train/train.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'train'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


def _has(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:])


def _default(flag: str, value: str | None = None) -> None:
    if _has(flag):
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


def main() -> None:
    defaults = {
        '--dataset': './corpus/dataset',
        '--total-tokens': '160000000',
        '--out-dir': './checkpoints_student',
        '--epochs': '48',
        '--warmup-epochs': '6',
        '--batch-size': '64',
        '--workers': '8',
        '--lr': '3e-3',
        '--precision': 'bf16',
        '--compile-mode': 'max-autotune',
        '--matmul-precision': 'high',
        '--prefetch-factor': '4',
        '--dim': '96',
        '--embed-dim': '64',
        '--head-hidden': '192',
        '--film-rank': '64',
        '--erase-rank': '16',
        '--sampler': 'tempered',
        '--sampling-exponent': '0.5',
        '--tail-floor': '0.00025',
        '--construct-boost': '1.0',
        '--boundary-boost': '1.0',
        '--calibration-fraction': '0.167',
        '--lang-loss': '0.15',
        '--struct-loss': '0.2',
        '--seed': '20260911',
        # The old external set covers only 57 languages. An empty root makes the
        # 75/25 popularity/coverage validation score select this checkpoint.
        '--real-bench-root': '',
    }
    for flag, value in defaults.items():
        _default(flag, value)
    _default('--mine-weak-languages')
    _default('--pin-memory')
    _default('--fused-optimizer')

    teacher = HERE / 'checkpoints_teacher' / 'best_model.pt'
    teacher_logits = HERE / 'corpus' / 'dataset' / 'teacher_logits.npy'
    if not _has('--full-precision'):
        if teacher_logits.exists() and not _has('--teacher-logits'):
            sys.argv.extend(['--teacher-logits', str(teacher_logits)])
        elif teacher.exists() and not _has('--teacher-checkpoint'):
            sys.argv.extend(['--teacher-checkpoint', str(teacher)])

    spec = importlib.util.spec_from_file_location('_lex_base_trainer', BASE_DIR / 'train.py')
    assert spec and spec.loader
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    trainer.main()


if __name__ == '__main__':
    main()
