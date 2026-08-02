r"""Query construction: metadata filters applied during search and scan.

:class:`~elips.Filter` is the native predicate builder. It is re-exported here
rather than wrapped -- the native fluent API is already the right shape, and a
Python wrapper around it would add a boundary crossing per predicate without
adding anything a caller wants.

Examples::

    >>> from elips.search import Filter
    >>> recent = Filter.compare("year", "ge", 2023)
    >>> recent.matches({"year": 2024})
    True
    >>> public_and_recent = recent.and_(Filter.compare("public", "eq", True))
    >>> public_and_recent.matches({"year": 2024, "public": False})
    False
"""

from __future__ import annotations

from elips._native import Comparator, Filter, QueryPlan, QueryStrategy, Result

__all__ = ["Comparator", "Filter", "QueryPlan", "QueryStrategy", "Result"]
