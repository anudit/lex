// TextMate scope -> gpu-lexer's 9 syntax classes.
//
// Shiki is the normalization reference for this project, so this table defines
// what "correct" means for every downstream label. Rules are longest-prefix,
// evaluated against the token's scope stack from innermost scope outward: the
// most specific scope that matches anything wins. That ordering matters --
// `keyword.operator.arithmetic` must beat the bare `keyword` rule, and a
// `punctuation.definition.string` inside `string.quoted.double` must stay STRING.

export const CLASS = {
  plain: 0, comment: 1, string: 2, number: 3, keyword: 4,
  type: 5, function: 6, constant: 7, operator: 8,
};
export const CLASS_NAMES = [
  'plain', 'comment', 'string', 'number', 'keyword',
  'type', 'function', 'constant', 'operator',
];

// Ordered most-specific-first. First matching prefix on a scope wins.
const RULES = [
  // --- comments -------------------------------------------------------------
  ['comment', CLASS.comment],
  ['punctuation.definition.comment', CLASS.comment],

  // --- strings --------------------------------------------------------------
  // Escapes and interpolation punctuation read as string in every theme; the
  // interpolated *expression* carries its own scopes and is handled normally.
  ['constant.character.escape', CLASS.string],
  ['punctuation.definition.string', CLASS.string],
  ['string.regexp', CLASS.string],
  ['string', CLASS.string],
  ['meta.embedded.line.regexp', CLASS.string],
  ['text.html.basic', CLASS.plain],
  // Markdown inline code and link text/title read as string, matching every
  // other "quoted-looking content" rule above.
  ['markup.inline.raw', CLASS.string],
  ['markup.underline.link', CLASS.string],

  // --- numbers --------------------------------------------------------------
  ['constant.numeric', CLASS.number],

  // --- constants ------------------------------------------------------------
  ['constant.language', CLASS.constant],
  ['constant.other.caps', CLASS.constant],
  ['variable.other.constant', CLASS.constant],
  ['support.constant', CLASS.constant],
  ['constant.other.symbol', CLASS.constant],
  ['constant.other', CLASS.constant],

  // --- operators (must precede the generic keyword rule) ---------------------
  ['keyword.operator', CLASS.operator],
  // Markdown's heading/bold/italic markers carry a punctuation.definition.*
  // scope as the *innermost* scope of the whole marked-up span (Shiki does not
  // split the marker from its content into separate tokens the way it does for
  // strings/comments), so without these exceptions the generic `punctuation`
  // rule below would claim the entire "# Heading" or "**bold**" span, not just
  // the `#`/`**` marks.
  ['punctuation.definition.heading', CLASS.keyword],
  ['punctuation.definition.bold', CLASS.plain],
  ['punctuation.definition.italic', CLASS.plain],
  ['punctuation.definition.raw', CLASS.string],
  ['punctuation', CLASS.operator],

  // --- keywords -------------------------------------------------------------
  ['keyword', CLASS.keyword],
  ['markup.heading', CLASS.keyword],
  ['storage.modifier', CLASS.keyword],
  ['storage.type.function', CLASS.keyword],
  ['storage.type.class', CLASS.keyword],
  ['storage.type.struct', CLASS.keyword],
  ['storage.type.enum', CLASS.keyword],
  ['storage.type.interface', CLASS.keyword],
  ['storage.type.annotation', CLASS.keyword],
  // Primitive/built-in type names (`int`, `bool`, `String`) read as types.
  ['storage.type.primitive', CLASS.type],
  ['storage.type.built-in', CLASS.type],
  ['storage.type.numeric', CLASS.type],
  ['storage.type', CLASS.keyword],
  ['storage', CLASS.keyword],

  // --- functions ------------------------------------------------------------
  ['entity.name.function', CLASS.function],
  ['support.function', CLASS.function],
  ['meta.function-call.generic', CLASS.function],
  ['entity.name.method', CLASS.function],
  ['variable.function', CLASS.function],

  // --- types ----------------------------------------------------------------
  ['entity.name.type', CLASS.type],
  ['entity.name.class', CLASS.type],
  ['entity.name.namespace', CLASS.type],
  ['entity.other.inherited-class', CLASS.type],
  ['support.type', CLASS.type],
  ['support.class', CLASS.type],
  ['entity.name.tag', CLASS.keyword],
  ['entity.other.attribute-name', CLASS.type],
  ['entity.name', CLASS.type],

  // --- everything identifier-shaped ----------------------------------------
  ['support.other.variable', CLASS.plain],
  ['variable', CLASS.plain],
  ['source', CLASS.plain],
  ['text', CLASS.plain],
  ['meta', CLASS.plain],
];

// Longest-prefix index: map a scope to its class by walking dotted prefixes down
// from the full scope name, which is cheaper and more predictable than scanning
// the rule list per token.
const EXACT = new Map();
for (const [prefix, cls] of RULES) {
  // Later rules must not clobber an earlier, more specific entry.
  if (!EXACT.has(prefix)) EXACT.set(prefix, cls);
}

const cache = new Map();

function classifyScope(scope) {
  const hit = cache.get(scope);
  if (hit !== undefined) return hit;
  let cls = -1;
  let s = scope;
  while (s.length) {
    const found = EXACT.get(s);
    if (found !== undefined) { cls = found; break; }
    const dot = s.lastIndexOf('.');
    if (dot === -1) break;
    s = s.slice(0, dot);
  }
  cache.set(scope, cls);
  return cls;
}

/**
 * Classify one grammar token from its scope stack (outermost first, as Shiki
 * emits it). Walks inward-out so the most specific scope decides.
 */
export function classifyScopes(scopes) {
  for (let i = scopes.length - 1; i >= 0; i--) {
    const cls = classifyScope(scopes[i]);
    if (cls !== -1) return cls;
  }
  return CLASS.plain;
}
