"""Resumably prepare all 193 languages and publish the finished cache."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_PATH = HERE / 'prepare_upload_state.json'


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {'format': 'lex-large-prepare-v1', 'completed': {}}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2) + '\n')


def github_environment() -> dict[str, str]:
    env = dict(os.environ)
    preferred = [str(Path.home() / '.bun' / 'bin'), '/opt/homebrew/bin', '/usr/local/bin']
    env['PATH'] = ':'.join([*preferred, env.get('PATH', '/usr/bin:/bin')])
    if env.get('GITHUB_TOKEN'):
        return env
    gh = shutil.which('gh', path=env['PATH'])
    if gh:
        result = subprocess.run(
            [gh, 'auth', 'token'], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            env['GITHUB_TOKEN'] = result.stdout.strip()
            print('Using the authenticated GitHub CLI token for discovery.', flush=True)
    return env


def run_stage(name: str, command: list[str], state: dict, env: dict[str, str]) -> None:
    if name in state['completed']:
        print(f'==> SKIP {name} (completed {state["completed"][name]})', flush=True)
        return
    print(f'==> START {name}: {" ".join(command)}', flush=True)
    subprocess.run(command, cwd=HERE, env=env, check=True)
    state['completed'][name] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f'==> DONE {name}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-id', default='anudit/lex-large')
    parser.add_argument('--total-tokens', type=int, default=160_000_000)
    parser.add_argument('--fetch-workers', type=int, default=6)
    parser.add_argument('--label-shards', type=int, default=8)
    parser.add_argument('--upload-workers', type=int, default=4)
    parser.add_argument('--private', action='store_true')
    args = parser.parse_args()

    state = load_state()
    env = github_environment()
    python = sys.executable
    stages = [
        ('bun-install', ['bun', 'install', '--frozen-lockfile']),
        ('language-manifest', ['bun', 'run', 'sync-languages']),
        ('repository-discovery', [python, '-u', 'discover_repos.py',
                                  '--per-language', '24', '--min-stars', '5']),
        ('reuse-small-corpus', [python, '-u', 'reuse_small_corpus.py',
                                '--source', '../train/corpus/raw',
                                '--out', './corpus/raw']),
        ('fixtures', [python, '-u', 'bootstrap_fixtures.py']),
        ('source-corpus', [python, '-u', 'fetch_corpus.py',
                           '--out', './corpus/raw',
                           '--total-tokens', str(args.total_tokens),
                           '--headroom', '3.0',
                           '--workers', str(args.fetch_workers)]),
        ('labels', [python, '-u', 'build_labels.py',
                    '--raw', './corpus/raw', '--out', './corpus/labels',
                    '--shards', str(args.label_shards)]),
        ('dataset-cache', [python, '-u', 'build_dataset.py',
                           '--labels', './corpus/labels',
                           '--cache', './corpus/dataset',
                           '--total-tokens', str(args.total_tokens),
                           '--seq-len', '512', '--min-len', '32', '--max-dup', '3']),
        ('setup-tests', [python, '-u', 'test_setup.py']),
    ]
    for name, command in stages:
        run_stage(name, command, state, env)

    upload = [python, '-u', 'upload_dataset.py',
              '--folder', './corpus/dataset', '--repo-id', args.repo_id,
              '--workers', str(args.upload_workers)]
    if args.private:
        upload.append('--private')
    run_stage('huggingface-upload', upload, state, env | {'HF_XET_HIGH_PERFORMANCE': '1'})


if __name__ == '__main__':
    main()
