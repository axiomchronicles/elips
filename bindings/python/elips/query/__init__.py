r"""EQL: the ELIPS query language.

Helpers for validating, tokenizing, and parsing EQL, plus the statement types a
parse produces. Useful when a query arrives from somewhere other than Python
code -- a saved report, a config file, an HTTP request -- and you want to check
or rewrite it before running it.

Executing a query is :meth:`elips.Engine.query`; this package is only about the
text of one.

Examples::

    >>> from elips.query import parse, validate
    >>> validate('seek in docs nearest $q top 5 yield') is None
    True
    >>> statement = parse('seek in docs nearest $q top 5 yield')
    >>> statement.vault, statement.top
    ('docs', 5)
"""

from __future__ import annotations

from elips._native import (
    DeleteStatement,
    FetchStatement,
    InsertStatement,
    ParseError,
    ScanStatement,
    SearchStatement,
    Token,
    TokenKind,
    VectorRef,
    parse_eql as parse,
    tokenize_eql as tokenize,
    validate_eql as validate,
)

__all__ = [
    "DeleteStatement",
    "FetchStatement",
    "InsertStatement",
    "ParseError",
    "ScanStatement",
    "SearchStatement",
    "Token",
    "TokenKind",
    "VectorRef",
    "parse",
    "tokenize",
    "validate",
]
