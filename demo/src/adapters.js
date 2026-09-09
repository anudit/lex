// Normalizes every highlighter onto the same nine classes so their outputs are
// comparable: plain, comment, string, number, keyword, type, function, constant,
// operator.
//
// Comparison is per non-whitespace *character*, not per token. Each library
// tokenizes differently -- Prism keeps `foo.bar` whole where Shiki splits it --
// so any token-level comparison would measure tokenizer disagreement rather than
// highlighting disagreement.

import { classifyScopes, CLASS, CLASS_NAMES } from '../../train/scope_map.mjs';

export { CLASS, CLASS_NAMES };
export const PLAIN = CLASS.plain;

/** Fill a per-character class array from {type,start,end} spans. */
export function fillFromSpans(code, spans) {
  const out = new Uint8Array(code.length).fill(PLAIN);
  for (const s of spans) {
    const cls = CLASS[s.type];
    if (cls === undefined) continue;
    for (let i = s.start; i < s.end && i < code.length; i++) out[i] = cls;
  }
  return out;
}

// ---------------------------------------------------------------- shiki (ref)

export function makeShikiAdapter(highlighter) {
  return {
    name: 'Shiki',
    reference: true,
    supports: (lang) => highlighter.getLoadedLanguages().includes(lang)
      || ['txt', 'text', 'plaintext'].includes(lang),
    classes(code, lang) {
      const out = new Uint8Array(code.length).fill(PLAIN);
      const res = highlighter.codeToTokens(code, {
        lang, theme: 'github-dark', includeExplanation: true,
      });
      let pos = 0;
      for (const line of res.tokens) {
        for (const token of line) {
          const parts = token.explanation?.length
            ? token.explanation
            : [{ content: token.content, scopes: [] }];
          for (const part of parts) {
            const text = part.content;
            if (!text.length) continue;
            if (!code.startsWith(text, pos)) {
              const idx = code.indexOf(text, pos);
              if (idx === -1) return out;
              pos = idx;
            }
            const cls = classifyScopes(part.scopes.map((s) => s.scopeName));
            out.fill(cls, pos, pos + text.length);
            pos += text.length;
          }
        }
        if (code[pos] === '\r') pos += 1;
        if (code[pos] === '\n') pos += 1;
      }
      return out;
    },
  };
}

// ------------------------------------------------------------------ sugar-high

// sugar-high's own vocabulary. It folds numeric literals into `class`, so the
// numeric test has to run before the type test.
const NUMERIC = /^[+-]?(0[xXbBoO][0-9a-fA-F_]+|\d[\d_]*(\.[\d_]*)?([eE][+-]?\d+)?)[uUlLfFdDnN]*$/;
const SCREAMING = /^[A-Z][A-Z0-9_]*$/;

function sugarType(type, text, applied) {
  switch (type) {
    case 'comment': return CLASS.comment;
    case 'string': case 'jsxliterals': return CLASS.string;
    case 'sign': return CLASS.operator;
    case 'entity': return CLASS.function;
    case 'keyword': return CLASS.keyword;
    case 'class':
      if (NUMERIC.test(text)) return CLASS.number;
      if (SCREAMING.test(text) && text.length > 1) return CLASS.constant;
      return CLASS.type;
    case 'identifier': case 'property':
      if (applied) return CLASS.function;
      if (SCREAMING.test(text) && text.length > 1) return CLASS.constant;
      return CLASS.plain;
    default: return CLASS.plain;
  }
}

export function makeSugarHighAdapter(parse, languages) {
  const byId = new Map(languages.map((l) => [l.id, l.config]));
  return {
    name: 'Sugar High',
    supports: (lang) => byId.has(lang),
    classes(code, lang) {
      const out = new Uint8Array(code.length).fill(PLAIN);
      const config = byId.get(lang);
      if (!config) return out;
      const parsed = parse(code, config);
      let pos = 0;
      for (const line of parsed.lines) {
        for (const t of line.tokens) {
          const v = t.value;
          if (!v.length) continue;
          if (!code.startsWith(v, pos)) {
            const idx = code.indexOf(v, pos);
            if (idx === -1 || idx - pos > 8) return out;
            pos = idx;
          }
          if (t.type !== 'space' && t.type !== 'break') {
            let probe = pos + v.length;
            while (probe < code.length && (code[probe] === ' ' || code[probe] === '\t')) probe++;
            out.fill(sugarType(t.type, v, code[probe] === '('), pos, pos + v.length);
          }
          pos += v.length;
        }
        if (code[pos] === '\r') pos += 1;
        if (code[pos] === '\n') pos += 1;
      }
      return out;
    },
  };
}

// ----------------------------------------------------------------------- prism

const PRISM_MAP = {
  comment: CLASS.comment, prolog: CLASS.comment, doctype: CLASS.comment,
  cdata: CLASS.comment,
  string: CLASS.string, 'template-string': CLASS.string, char: CLASS.string,
  regex: CLASS.string, 'attr-value': CLASS.string, 'string-property': CLASS.string,
  number: CLASS.number,
  keyword: CLASS.keyword, atrule: CLASS.keyword, important: CLASS.keyword,
  tag: CLASS.keyword, selector: CLASS.keyword, rule: CLASS.keyword,
  'class-name': CLASS.type, builtin: CLASS.type, 'attr-name': CLASS.type,
  namespace: CLASS.type, symbol: CLASS.type, entity: CLASS.type,
  function: CLASS.function, 'function-variable': CLASS.function, method: CLASS.function,
  boolean: CLASS.constant, constant: CLASS.constant, null: CLASS.constant,
  operator: CLASS.operator, punctuation: CLASS.operator, url: CLASS.string,
  variable: CLASS.plain, parameter: CLASS.plain, property: CLASS.plain,
  'property-access': CLASS.plain, 'maybe-class-name': CLASS.type,
  inserted: CLASS.plain, deleted: CLASS.plain,
};

export function makePrismAdapter(Prism, langMap) {
  return {
    name: 'Prism.js',
    supports: (lang) => !!Prism.languages[langMap[lang]],
    classes(code, lang) {
      const out = new Uint8Array(code.length).fill(PLAIN);
      const grammar = Prism.languages[langMap[lang]];
      if (!grammar) return out;
      let pos = 0;
      const walk = (tokens, inherited) => {
        for (const t of tokens) {
          if (typeof t === 'string') { pos += t.length; continue; }
          const cls = PRISM_MAP[t.type] ?? inherited;
          if (typeof t.content === 'string') {
            if (cls !== undefined) out.fill(cls, pos, pos + t.content.length);
            pos += t.content.length;
          } else if (Array.isArray(t.content)) {
            walk(t.content, cls);
          } else {
            walk([t.content], cls);
          }
        }
      };
      walk(Prism.tokenize(code, grammar), undefined);
      return out;
    },
  };
}

// ---------------------------------------------------------- span-based engines

export function makeSpanAdapter(name, highlight, supports = () => true) {
  return {
    name,
    supports,
    async classesAsync(code) {
      return fillFromSpans(code, await highlight(code));
    },
  };
}

// ------------------------------------------------------------------- scoring

/**
 * Fraction of non-whitespace characters on which `got` matches `ref`.
 * Whitespace is excluded because every library agrees on it trivially and
 * including it would inflate every score toward the whitespace ratio.
 */
export function agreement(code, ref, got) {
  let hit = 0;
  let total = 0;
  for (let i = 0; i < code.length; i++) {
    const c = code.charCodeAt(i);
    if (c === 32 || c === 9 || c === 10 || c === 13) continue;
    total++;
    if (ref[i] === got[i]) hit++;
  }
  return { hit, total };
}
