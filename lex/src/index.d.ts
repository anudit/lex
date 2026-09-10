// SyntaxClassName, SyntaxSpan, and `parse` are identical to gpu-lexer's
// index.d.ts on purpose -- see the comment at the top of index.js.

type SyntaxClassName =
  | "plain"
  | "comment"
  | "string"
  | "number"
  | "keyword"
  | "type"
  | "function"
  | "constant"
  | "operator";

interface SyntaxSpan {
  type: SyntaxClassName;
  start: number;
  end: number;
}

export declare function parse(
  code: string,
): Promise<SyntaxSpan[]>;

// Everything below is lex's own lower-level surface, not part of gpu-lexer's
// API: an explicit session (rather than the implicit shared one `parse`
// uses), per-token classes instead of merged spans, and a debug-stage hook.

export declare const CLASS_NAMES: SyntaxClassName[];

export declare function isSupported(): boolean;

export declare function createLexer(options?: { shared?: boolean }): Promise<Lexer>;

export declare class Lexer {
  highlight(code: string): Promise<SyntaxSpan[]>;
  classify(code: string): Promise<{ tokens: unknown; classes: Uint32Array }>;
  debugStage(code: string, stage: number): Promise<Float32Array>;
  readonly stages: string[];
  destroy(): void;
}
