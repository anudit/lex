"""Train the unquantized teacher used to distill the shipping student."""

from __future__ import annotations

import sys

from train import main

DEFAULTS = (
    ('--out-dir', './checkpoints_teacher'),
    ('--epochs', '24'),
    ('--batch-size', '32'),
    ('--workers', '8'),
    ('--lr', '2e-3'),
    ('--precision', 'bf16'),
    ('--compile-mode', 'max-autotune'),
    ('--matmul-precision', 'high'),
    ('--prefetch-factor', '4'),
    ('--dim', '192'),
    ('--embed-dim', '96'),
    ('--head-hidden', '384'),
    ('--film-rank', '96'),
    ('--erase-rank', '32'),
    ('--teacher-checkpoint', ''),
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
