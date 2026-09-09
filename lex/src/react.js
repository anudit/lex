// React bindings. Written with createElement rather than JSX so the package
// ships as plain ESM with no build step and no bundler assumptions; React itself
// is a peer dependency, so the core engine stays dependency-free.

import { createElement, useEffect, useMemo, useRef, useState } from 'react';
import { createLexer, isSupported, CLASS_NAMES } from './index.js';

/**
 * Highlight `code` and return spans, re-running when the code changes.
 * Returns `{spans, ready, error}`; `spans` is null until the first result, so a
 * caller can render unhighlighted text immediately instead of flashing empty.
 */
export function useLex(code, options = {}) {
  const [state, setState] = useState({ spans: null, ready: false, error: null });
  const seq = useRef(0);

  useEffect(() => {
    if (!isSupported()) {
      setState({ spans: null, ready: true, error: new Error('WebGPU unavailable') });
      return undefined;
    }
    const id = ++seq.current;
    let cancelled = false;
    createLexer(options)
      .then((lexer) => lexer.highlight(code))
      .then((spans) => {
        // Drop results from superseded renders; highlight() is async and a fast
        // typist can outrun it.
        if (!cancelled && id === seq.current) {
          setState({ spans, ready: true, error: null });
        }
      })
      .catch((error) => {
        if (!cancelled && id === seq.current) setState({ spans: null, ready: true, error });
      });
    return () => { cancelled = true; };
  }, [code]);

  return state;
}

/** Split source into renderable pieces, including the gaps between spans. */
export function toPieces(code, spans) {
  if (!spans) return [{ type: null, text: code }];
  const pieces = [];
  let pos = 0;
  for (const s of spans) {
    if (s.start > pos) pieces.push({ type: null, text: code.slice(pos, s.start) });
    pieces.push({ type: s.type, text: code.slice(s.start, s.end) });
    pos = s.end;
  }
  if (pos < code.length) pieces.push({ type: null, text: code.slice(pos) });
  return pieces;
}

/**
 * <Lex code="..." /> renders a <pre><code> with one span per highlighted run.
 * Classes are `lex-<name>`; the package ships a stylesheet but any theme that
 * targets those nine names works.
 */
export function Lex({ code, className, style, prefix = 'lex-', as = 'pre', ...rest }) {
  const { spans, error } = useLex(code);
  const pieces = useMemo(() => toPieces(code, spans), [code, spans]);
  return createElement(
    as,
    {
      className: ['lex', className].filter(Boolean).join(' '),
      style,
      'data-lex-error': error ? String(error.message) : undefined,
      ...rest,
    },
    createElement(
      'code',
      null,
      ...pieces.map((p, i) =>
        p.type
          ? createElement('span', { key: i, className: prefix + p.type }, p.text)
          : p.text),
    ),
  );
}

export { CLASS_NAMES, isSupported };
