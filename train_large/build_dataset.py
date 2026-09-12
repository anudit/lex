from __future__ import annotations

import argparse

import data_pipeline
from languages import DEFAULT_TOTAL_TOKENS, MIN_TOKENS_PER_LANG, TARGET_LANGUAGES


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the 185-language cached dataset')
    parser.add_argument('--labels', default='./corpus/labels')
    parser.add_argument('--cache', default='./corpus/dataset')
    parser.add_argument('--total-tokens', type=int, default=DEFAULT_TOTAL_TOKENS)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--min-len', type=int, default=32)
    parser.add_argument('--max-dup', type=int, default=3)
    parser.add_argument('--allow-underfilled', action='store_true')
    args = parser.parse_args()
    datasets, meta = data_pipeline.build(
        label_dir=args.labels,
        cache=args.cache,
        total_tokens=args.total_tokens,
        seq_len=args.seq_len,
        min_len=args.min_len,
        max_dup=args.max_dup,
    )
    for split, dataset in datasets.items():
        print(f'{split:6s} {len(dataset):8,d} windows')
    underfilled = {
        language: meta['tokens_per_lang'].get(language, 0)
        for language in TARGET_LANGUAGES
        if meta['tokens_per_lang'].get(language, 0) < MIN_TOKENS_PER_LANG
    }
    if underfilled and not args.allow_underfilled:
        detail = ', '.join(
            f'{language}={count:,}' for language, count in
            sorted(underfilled.items(), key=lambda item: item[1])[:20])
        raise SystemExit(
            f'{len(underfilled)} languages are below the {MIN_TOKENS_PER_LANG:,}-token '
            f'floor ({detail}). Add source data or pass --allow-underfilled for a smoke run.')
    if meta['total_tokens'] < int(args.total_tokens * 0.95) and not args.allow_underfilled:
        raise SystemExit(
            f'dataset contains {meta["total_tokens"]:,} tokens, below 95% of the '
            f'{args.total_tokens:,} target; fetch more source data or use '
            f'--allow-underfilled only for a smoke run')


if __name__ == '__main__':
    main()
