"""
Post-training pipeline: export, bundle, verify, report.

Run after training finishes. Each step is a precondition for the next, so the
script stops at the first failure rather than producing a half-updated package —
several bugs this session came from a bundle and a fixture drifting out of sync.

    uv run python finalize.py

Browser-side checks (the WGSL kernel against the numpy reference, and the
head-to-head benchmark) still need `bun run dev` in demo/ and a WebGPU browser;
this prints the URLs at the end.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(desc: str, cmd: list[str], quiet_filter: str | None = None) -> str:
    print(f'\n=== {desc} ===')
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    lines = [l for l in out.splitlines()
             if 'index_reduce' not in l and 'per_group' not in l]
    if quiet_filter:
        lines = [l for l in lines if quiet_filter in l]
    print('\n'.join(lines[-40:]))
    if proc.returncode != 0:
        sys.exit(f'\nFAILED: {desc} (exit {proc.returncode})')
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', default='./checkpoints')
    ap.add_argument('--lex-src', default='../lex/src')
    ap.add_argument('--skip-eval', action='store_true')
    args = ap.parse_args()

    py = [sys.executable]
    run('export the trained checkpoint',
        py + ['export.py', '--checkpoint', f'{args.checkpoint_dir}/best_model.pt',
              '--out', args.checkpoint_dir])
    run('regenerate shader fixtures',
        py + ['make_refcases.py', '--weights', args.checkpoint_dir])
    run('bundle weights and shader into lex',
        py + ['bundle_lex.py', '--checkpoint-dir', args.checkpoint_dir,
              '--lex-src', args.lex_src])
    run('test suite', py + ['test_all.py'])

    ckpt = f'{args.checkpoint_dir}/best_model.pt'
    if not args.skip_eval:
        run('held-out test split', py + ['eval.py', '--checkpoint', ckpt,
                                         '--split', 'test', '--workers', '0'])
        run('error breakdown', py + ['diagnose.py', '--checkpoint', ckpt,
                                     '--workers', '0'])
        run('FiLM conditioning', py + ['film_check.py', '--checkpoint', ckpt,
                                       '--workers', '0'])
        run('points remaining to 95%', py + ['headroom_to_95.py', '--checkpoint', ckpt])

    print('\n=== browser checks (need `bun run dev` in demo/) ===')
    print('  shader vs numpy reference : http://localhost:5173/validate.html')
    print('  head-to-head benchmark    : http://localhost:5173/')


if __name__ == '__main__':
    main()
