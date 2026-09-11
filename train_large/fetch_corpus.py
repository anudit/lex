"""Large-corpus entry point using the compact fetcher's diversity safeguards."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from languages import DEFAULT_TOTAL_TOKENS, LANGUAGE_META

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'train'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


def _has(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:])


def _default(flag: str, value: str) -> None:
    if not _has(flag):
        sys.argv.extend([flag, value])


def main() -> None:
    _default('--out', './corpus/raw')
    _default('--total-tokens', str(DEFAULT_TOTAL_TOKENS))
    _default('--headroom', '3.0')
    _default('--workers', '6')

    spec = importlib.util.spec_from_file_location('_lex_compact_fetch', BASE_DIR / 'fetch_corpus.py')
    assert spec and spec.loader
    fetcher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fetcher
    spec.loader.exec_module(fetcher)
    for language, meta in LANGUAGE_META.items():
        values = tuple(ext.lower() for ext in meta.get('extensions', ()) if ext.startswith('.'))
        # A missing extension is kept out of automatic archive extraction. Its
        # grammar remains in the manifest and should be seeded through fixtures.
        if values:
            fetcher.LANG_EXTENSIONS[language] = values
        filenames = tuple(name.lower() for name in meta.get('filenames', ()))
        if filenames:
            fetcher.FILENAME_MATCH[language] = filenames
    fetcher.main()


if __name__ == '__main__':
    main()
