r"""The database API: a typed facade over one native database handle."""

from __future__ import annotations

from elips.database.engine import DEFAULT_TEXT_SLOT, Engine

__all__ = ["DEFAULT_TEXT_SLOT", "Engine"]
