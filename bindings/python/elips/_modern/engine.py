r"""Typed database facade over the native :class:`elips.Database` handle.

:class:`Engine` is a context manager that hands out :class:`~elips.Arena`
collections and exposes WAL replay, checkpointing, and transactions.

Examples::

    >>> import elips
    >>> with elips.connect(":memory:", dimension=2) as engine:
    ...     arena = engine.arena("documents")
    ...     arena.count()
    0
"""

from __future__ import annotations

import os
from types import TracebackType

from elips._modern.arena import Arena
from elips._modern.models import WalRecord
from elips._modern.typing import Embedder
from elips._native import Config, Database, replay_wal

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
        default_embedder: Embedder | None = None,
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
        embedder: Embedder | None = None,
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

    def vacuum(self) -> None:
        r"""vacuum() -> None

        Reclaim index space held by deleted records across every arena.

        Deletes leave tombstones so graph navigation stays intact. Each arena
        compacts itself once tombstones pass its configured
        ``compaction_ratio``, but after a bulk delete it is worth reclaiming
        immediately. Unlike :meth:`compact`, this does not rewrite the on-disk
        snapshot and works on in-memory databases.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> key = arena.write(text="alpha note")
            >>> _ = arena.discard([key])
            >>> engine.vacuum()
            >>> arena.pending_removals
            0
            >>> engine.close()
        """

        self._db.vacuum()

    def vault_names(self) -> list[str]:
        r"""vault_names() -> list[str]

        Return the names of every arena that currently exists.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> _ = engine.arena("documents").write(text="alpha note")
            >>> engine.vault_names()
            ['documents']
            >>> engine.close()
        """

        return self._db.list_vaults()

    def pending_writes(self) -> list[WalRecord]:
        r"""pending_writes() -> list[WalRecord]

        Read this database's write-ahead log without mutating anything.

        Returns every record the log acknowledged, in log order. Records inside
        an unterminated transaction are omitted, matching what recovery would
        apply, and a corrupt tail is dropped rather than raised. Returns an
        empty list for in-memory databases, and after :meth:`checkpoint`, which
        truncates the log.

        Returns:
            list[WalRecord]: Acknowledged log records.

        Examples::

            >>> import elips, tempfile
            >>> path = tempfile.mkdtemp()
            >>> engine = elips.connect(path, dimension=2)
            >>> _ = engine.arena("documents").write(vector=[1.0, 0.0])
            >>> [record.op for record in engine.pending_writes()]
            ['insert']
            >>> engine.close()
        """

        path = self._db.path
        if not path or path == ":memory:":
            return []
        wal_path = os.path.join(path, "wal.log")
        if not os.path.exists(wal_path):
            return []
        return [WalRecord.from_entry(entry) for entry in replay_wal(wal_path)]

    def close(self) -> None:
        r"""close() -> None

        Close the underlying database handle.

        Checkpoints, releases the cross-process lock, and seals every arena so
        later writes raise instead of silently failing to persist.
        """

        self._db.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["DEFAULT_TEXT_SLOT", "Engine"]
