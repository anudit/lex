"""Train the unquantized teacher used to distill the shipping student.

The teacher never ships, so it is sized for accuracy rather than bytes: wider
than the student, with dropout and heavier weight decay because the
192-wide teacher overfit after epoch 20 (val loss rising while train accuracy
climbed). The natural-popularity calibration phase is off -- the teacher must be
good on every language, and in the 192-wide run that phase lifted train
accuracy 2 points while val got worse.

For cross-fitted distillation, train one teacher per fold:

    train_teacher.py --teacher-folds 2 --teacher-fold 0 --out-dir ./checkpoints_teacher_f0
    train_teacher.py --teacher-folds 2 --teacher-fold 1 --out-dir ./checkpoints_teacher_f1

then pass both checkpoints to cache_teacher_logits.py.

Width, not depth, provisionally: on MPS a 256-wide 4-layer teacher (2.44M
params) stepped 1.5x slower than the 192-wide one, while 6 layers cost 5x --
the dilation-16/32 depthwise convs dominate there. Training runs on the RTX
box, so run time_teacher.py on it before choosing `--n-layers`.
"""

from __future__ import annotations

import os
import sys

from train import main

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '1'))

DEFAULTS = (
    ('--out-dir', './checkpoints_teacher'),
    ('--epochs', '24'),
    # --batch-size and --workers are per process/GPU under torchrun.
    ('--batch-size', str(max(1, 32 // WORLD_SIZE))),
    ('--workers', '4' if WORLD_SIZE > 1 else '8'),
    ('--lr', '2e-3'),
    ('--precision', 'bf16'),
    ('--compile-mode', 'none'),
    ('--matmul-precision', 'high'),
    ('--prefetch-factor', '4'),
    ('--dim', '256'),
    ('--embed-dim', '128'),
    ('--head-hidden', '512'),
    ('--film-rank', '128'),
    ('--erase-rank', '64'),
    ('--n-layers', '4'),
    ('--dropout', '0.1'),
    ('--weight-decay', '0.05'),
    ('--calibration-fraction', '0'),
    ('--kernel-size', '7'),
    # Teacher is entirely FP; student bit allocation and byte cap do not apply.
    ('--input-bits', '1'),
    ('--head-bits', '1'),
    ('--output-bits', '1'),
    ('--weight-budget', '0'),
    ('--teacher-checkpoint', ''),
    ('--teacher-logits', ''),
)

for flag, value in DEFAULTS:
    if not any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:]):
        sys.argv.extend([flag, value])
if '--full-precision' not in sys.argv:
    sys.argv.append('--full-precision')
if '--pin-memory' not in sys.argv and '--no-pin-memory' not in sys.argv:
    sys.argv.append('--pin-memory')
if '--fused-optimizer' not in sys.argv and '--no-fused-optimizer' not in sys.argv:
    sys.argv.append('--fused-optimizer')

if __name__ == '__main__':
    main()
