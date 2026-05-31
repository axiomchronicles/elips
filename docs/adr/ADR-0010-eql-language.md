# ADR-0010: EQL as a parsed language

**Status:** Accepted
**Date:** 2024-08-01

## Context
ELIPS needs an expressive, tool-friendly query surface (CLI, logs, bindings)
beyond a method-chaining SDK.

## Decision
Provide EQL — a first-class language with a lexer, recursive-descent parser, AST,
and executor. Statements: `seek` (KNN + filter + threshold + rank_by + project),
`fetch`, `scan`, `place`, `erase`. The `where` clause reuses the `Filter` tree,
keeping one predicate engine across the SDK and the language.

## Consequences
- A portable, inspectable query string works identically from C++, Python, and CLI.
- Clear extension point for planner/optimizer rules later.
- A grammar is more to maintain than a pure fluent API; mitigated by reusing `Filter`.

## Alternatives Considered
- **Fluent SDK only:** no portable query representation for the CLI/tools.
- **SQL dialect:** familiar but a poor fit for vector-first semantics.
