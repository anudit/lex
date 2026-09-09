"""
Per-language corpus health, for deciding whether a weak language is a model
problem or a data problem.

Four signals, in the order they usually explain a failure:
  * tokens per file -- a budget met by a handful of huge files teaches one house
    style. This is what made markdown score 15% on unseen repositories.
  * top-repo share -- what fraction of a language's tokens come from its single
    largest source. Many files from one project is still one distribution:
    plaintext was 400k tokens of SPDX licence templates spread over hundreds of
    files, so tokens-per-file looked healthy while the model learned that a bare
    English word on its own line is a keyword. It scored ~100% on validation,
    which drew from the same source, and 34% on real word lists.
  * label distribution -- a language whose tokens are almost all one class is
    easy; one that disagrees with the corpus-wide mix may be mislabelled.
  * mask rate -- how often Shiki leaves a token's first character uncovered, i.e.
    how far the two tokenizers disagree on boundaries. High values mean the
    labels themselves are noisy and no model will fit them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import data_pipeline as dp
import tokenizer
from labels import CLASS_NAMES, MASK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='./corpus/labels')
    ap.add_argument('--raw', default='./corpus/raw')
    ap.add_argument('--languages', default='')
    args = ap.parse_args()

    want = {l.strip() for l in args.languages.split(',') if l.strip()}
    stats: dict[str, dict] = {}

    for shard in sorted(Path(args.labels).glob('labels.*.tsv')):
        for line in shard.open():
            path, _, rle = line.rstrip('\n').partition('\t')
            if not rle:
                continue
            lang = Path(path).parent.name
            if want and lang not in want:
                continue
            try:
                code = Path(path).read_text(encoding='utf-8')
            except OSError:
                continue
            s = stats.setdefault(lang, {'files': 0, 'tokens': 0, 'masked': 0,
                                        'labels': Counter(), 'repos': Counter()})
            # Files are named "<owner>__<repo>__<path>"; the prefix identifies
            # the source project.
            stem = Path(path).name.split('__')
            repo = '__'.join(stem[:2]) if len(stem) > 2 else '?'
            char_cls = dp.decode_rle(rle, len(code))
            toks = tokenizer.tokenize(code)
            lab = dp.align(toks, char_cls)
            # Only count tokens that carry visible text; whitespace is masked by
            # design and would swamp the real disagreement rate.
            visible = np.array([t.kind not in (1, 2) for t in toks])
            s['files'] += 1
            s['tokens'] += int(visible.sum())
            s['repos'][repo] += int(visible.sum())
            s['masked'] += int(((lab == MASK) & visible).sum())
            s['labels'].update(lab[lab >= 0].tolist())

    print(f'{"language":12s} {"files":>6s} {"tokens":>10s} {"tok/file":>9s} '
          f'{"top repo":>9s} {"mask%":>6s}   dominant classes')
    flagged = []
    for lang, s in sorted(stats.items(), key=lambda kv: -kv[1]['tokens'] / max(1, kv[1]['files'])):
        tot = sum(s['labels'].values())
        top = ', '.join(f'{CLASS_NAMES[k]} {100 * v / max(1, tot):.0f}%'
                        for k, v in s['labels'].most_common(3))
        share = (s['repos'].most_common(1)[0][1] / max(1, s['tokens'])) if s['repos'] else 0
        mark = ' *' if share > 0.5 else '  '
        if share > 0.5:
            flagged.append((lang, share, s['repos'].most_common(1)[0][0]))
        print(f'{lang:12s} {s["files"]:6,d} {s["tokens"]:10,d} '
              f'{s["tokens"] / max(1, s["files"]):9,.0f} '
              f'{100 * share:8.0f}%{mark}'
              f'{100 * s["masked"] / max(1, s["tokens"]):5.1f}%   {top}')
    if flagged:
        print('\n* over half this language\'s tokens come from one project, so'
              '\n  validation drawn from the same source will not reveal the gap:')
        for lang, share, repo in flagged:
            print(f'    {lang:12s} {100 * share:.0f}% from {repo}')


if __name__ == '__main__':
    main()
