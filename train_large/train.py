"""Opinionated entry point for the 101,107-byte large-v2 student.

All flags remain overridable. Defaults are inserted only when the caller did
not provide that flag, so this stays compatible with ../train/train.py.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'train'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


def _has(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:])


def _default(flag: str, value: str | None = None) -> None:
    if _has(flag) or (value is None and _has('--no-' + flag.removeprefix('--'))):
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


def main() -> None:
    from model import student_config
    student = student_config()
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    defaults = {
        '--dataset': './corpus/dataset_v2',
        '--total-tokens': '160000000',
        '--out-dir': './checkpoints_student',
        # Both 48-epoch runs selected their final epoch with val loss still
        # falling, so the QAT phase was cut short rather than converged.
        '--epochs': '64',
        '--warmup-epochs': '6',
        # Keep the historical global batch at 64 under torchrun. --batch-size
        # is per process/GPU in the shared DDP trainer.
        '--batch-size': str(max(1, 64 // world_size)),
        '--workers': '4' if world_size > 1 else '8',
        '--lr': '3e-3',
        '--precision': 'bf16',
        '--compile-mode': 'none',
        '--matmul-precision': 'high',
        '--prefetch-factor': '4',
        '--dim': '96',
        '--embed-dim': '64',
        '--head-hidden': '192',
        '--film-rank': '64',
        '--erase-rank': str(student.erase_rank),
        '--n-layers': str(student.n_layers),
        '--kernel-size': str(student.kernel_size),
        '--embed-bits': str(student.embed_bits),
        '--head-bits': str(student.head_bits),
        '--input-bits': str(student.input_bits),
        '--output-bits': str(student.output_bits),
        '--embedding-scale': student.embedding_scale,
        '--sampler': 'tempered',
        '--sampling-exponent': '0.5',
        '--tail-floor': '0.00025',
        '--construct-boost': '1.0',
        '--boundary-boost': '1.0',
        '--class-weight-exponent': '0.25',
        '--calibration-fraction': '0.167',
        '--lang-loss': '0.15',
        '--struct-loss': '0.2',
        # 0.2 was set when logits came from a teacher scoring its own training
        # windows. Cross-fitted (held-out) logits are worth leaning on harder.
        '--distill-weight': '0.5',
        '--distill-temperature': '2.0',
        '--seed': '20260911',
        '--weight-budget': '111000',
        # The old external set covers only 57 languages. An empty root makes the
        # 75/25 popularity/coverage validation score select this checkpoint.
        '--real-bench-root': '',
        # Port lex-lite's whitespace-free representation, remove context views
        # duplicated by the signature, and QAT the scalar state to 8 bits.
        '--feature-version': str(student.feature_version),
        '--ctx-views': student.ctx_views,
        '--scalar-bits': str(student.scalar_bits),
    }
    for flag, value in defaults.items():
        _default(flag, value)
    _default('--mine-weak-languages')
    _default('--binary-ste')
    _default('--pin-memory')
    _default('--fused-optimizer')

    teacher = HERE / 'checkpoints_teacher' / 'best_model.pt'
    teacher_logits = HERE / 'corpus' / 'dataset_v2' / 'teacher_logits.npy'
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
