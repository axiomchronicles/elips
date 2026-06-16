from __future__ import annotations

from typing import Optional

from ..core import Config, Database, Vault
from .arena import Arena
from .typing import Embedder

DEFAULT_TEXT_SLOT = "__elips_text__"


class Engine:
    r"""Engine(db, *, default_embedder=None) -> Engine

    High-level wrapper around :class:`elips.core.Database`.

    Args:
        db (Database): Open ELIPS database handle.
        default_embedder (Embedder, optional): Batch embedder used by arenas
            when the database does not have a native text embedder. Default:
            ``None``.

    Examples::

        >>> import elips
        >>> engine = elips.connect(":memory:", dimension=2)
        >>> engine.config.dimension_val
        2
        >>> engine.close()
    """

    def __init__(
        self,
        db: Database,
        *,
        default_embedder: Optional[Embedder] = None,
    ) -> None:
        self._db = db
        self._default_embedder = default_embedder

    @property
    def raw(self) -> Database:
        r"""raw -> Database

        Return the underlying low-level database handle.
        """

        return self._db

    @property
    def config(self) -> Config:
        r"""config -> Config

        Return the effective database configuration.
        """

        return self._db.config

    def arena(
        self,
        name: str,
        *,
        embedder: Optional[Embedder] = None,
        text_slot: str = DEFAULT_TEXT_SLOT,
    ) -> Arena:
        r"""arena(name, *, embedder=None, text_slot="__elips_text__") -> Arena

        Create a typed high-level arena wrapper for a named vault.

        Args:
            name (str): Vault name.
            embedder (Embedder, optional): Arena-specific embedder override.
                Default: ``None``.
            text_slot (str, optional): Reserved compatibility argument from the
                early wrapper design. The current document-aware runtime stores
                text on :class:`elips.DocumentAttachment` rather than mirroring
                it into metadata. Default: ``"__elips_text__"``.

        Returns:
            Arena: High-level arena wrapper.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> arena.name
            'documents'
            >>> engine.close()
        """

        return Arena(
            self._db,
            name,
            embedder=embedder if embedder is not None else self._default_embedder,
            text_slot=text_slot,
        )

    def checkpoint(self) -> None:
        r"""checkpoint() -> None

        Flush the database state to durable storage.
        """

        self._db.checkpoint()

    def compact(self) -> None:
        r"""compact() -> None

        Rebuild indexes and compact the persistent layout.
        """

        self._db.compact()

    def close(self) -> None:
        r"""close() -> None

        Close the underlying database handle.
        """

        self._db.close()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["DEFAULT_TEXT_SLOT", "Engine"]
