// Ported verbatim from gpu-lexer's packages/training/src/classes.js (taxonomy
// version 5), so the "gpu-lexer verification corpus" benchmark in benchmark.js
// can score every engine -- including lex-lite -- under gpu-lexer's own
// scope-to-class taxonomy and its own confidence-gated denominator, instead of
// lex's scope_map.mjs. That is what makes that one corpus's numbers directly
// comparable to gpu-lexer's published correctness figures; every other corpus
// in this demo keeps using lex's own ground truth (train/scope_map.mjs), which
// is documented as *the* definition of correct for lex's own reported numbers.
//
// Re-sync by re-copying that file if gpu-lexer bumps its taxonomy version.

const auxiliaryRules = [
  /(?:^|\.)comment(?:\.|$)/,
  /(?:^|\.)(?:string|regexp)(?:\.|$)/,
  /(?:^|\.)(?:preprocessor|directive|at-rule)(?:\.|$)/,
  /(?:^|\.)(?:tag|markup|attribute-name|attribute-value)(?:\.|$)/,
  /(?:^|\.)(?:embedded|template\.expression)(?:\.|$)/,
  /(?:^|\.)(?:selector|attribute-name\.class|attribute-name\.id)(?:\.|$)/,
  /(?:^|\.)(?:property-name)(?:\.|$)/,
  /(?:^|\.)(?:property-value|declaration-value)(?:\.|$)/,
  /(?:^|\.)(?:property-access|object-member|member|variable\.other\.property|support\.variable\.property)(?:\.|$)/,
  /(?:^|\.)(?:select|insert|update|delete|create|alter|from|where|join|clause)(?:\.|$)/,
];

const rules = [
  ["comment", /^comment\./],
  ["string", /^(?:string|constant\.other\.symbol|constant\.regexp|markup\.inline\.raw|markup\.underline\.link)\./],
  ["number", /^constant\.numeric\./],
  ["operator", /^keyword\.operator(?:\.|$)/],
  ["keyword", /^(?:keyword(?!\.operator(?:\.|$))|storage|modifier|control|markup\.heading|entity\.name\.tag)(?:\.|$)/],
  ["type", /^(?:entity\.name\.(?:type|class|interface|struct|enum)|support\.(?:type|class))\./],
  ["function", /^(?:entity\.name\.(?:function|method)|support\.function)\./],
  ["constant", /^(?:constant\.(?:language|other)|support\.constant)(?:\.|$)/],
  // Generic punctuation not already claimed by an outer scope is operator-like.
  ["operator", /^punctuation(?:\.|$)/],
];

/** Collapse nested TextMate scopes into gpu-lexer's display taxonomy. */
export function classFromScopes(scopes) {
  scopes = normalizeScopes(scopes);
  if (scopes.some((scope) => /^punctuation\.definition\.template-expression(?:\.|$)/.test(scope))) {
    return 'operator';
  }
  scopes = withoutOuterTemplateString(scopes);
  for (const [name, pattern] of rules) {
    if (scopes.some((scope) => pattern.test(scope))) return name;
  }
  return 'plain';
}

export function auxiliaryFromScopes(scopes) {
  scopes = normalizeScopes(scopes);
  scopes = withoutOuterTemplateString(scopes);
  let bits = 0;
  for (let index = 0; index < auxiliaryRules.length; index++) {
    if (scopes.some((scope) => auxiliaryRules[index].test(scope))) bits |= 1 << index;
  }
  return bits;
}

/** 1 = unambiguous, 0.5 = scopes matched more than one class (or unknown/unparsed),
 * 0.25 = invalid. gpu-lexer's own correctness.js truncates confidence into a
 * Uint8Array, so only exact confidence===1 tokens survive into its denominator.
 */
export function confidenceFromScopes(scopes) {
  scopes = withoutOuterTemplateString(normalizeScopes(scopes));
  if (scopes.some((scope) => /^invalid(?:\.|$)/.test(scope))) return 0.25;
  const matches = new Set();
  for (const [name, pattern] of rules) {
    if (scopes.some((scope) => pattern.test(scope))) matches.add(name);
  }
  if (matches.size > 1) return 0.5;
  if (scopes.some((scope) => /(?:^|\.)(?:unknown|unparsed|illegal)(?:\.|$)/.test(scope))) return 0.5;
  return 1;
}

export function normalizeScopes(scopes) {
  return [...new Set(scopes
    .filter((scope) => typeof scope === 'string')
    .map((scope) => scope.trim().toLowerCase())
    .filter(Boolean))];
}

function withoutOuterTemplateString(scopes) {
  const embedded = scopes.some((scope) => /^meta\.(?:embedded|template\.expression)(?:\.|$)/.test(scope));
  return embedded ? scopes.filter((scope) => !/^string\.template(?:\.|$)/.test(scope)) : scopes;
}
