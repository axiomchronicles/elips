r"""Typed database facade over the native :class:`elips.Database` handle.

:class:`Engine` is a context manager that hands out :class:`~elips.Collection`
handles and exposes WAL replay, checkpointing, and transactions.

Examples::

    >>> import elips
    >>> with elips.connect(":memory:", dimension=2) as db:
    ...     docs = db.collection("documents")
    ...     docs.count()
    0
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Self

from elips._internal.protocols import Embedder
from elips._native import Config, Database, replay_wal
from elips.collection.collection import Collection
from elips.records.models import WalRecord

DEFAULT_TEXT_SLOT = "__elips_text__"


class Engine:
    r"""Engine(db, *, default_embedder=None) -> Engine

    High-level wrapper around :class:`~elips.Database`.

    Args:
        db (Database): Open ELIPS database handle.
        default_embedder (Embedder, optional): Batch embedder used by
            collections when the database does not have a native text embedder.
            Default: ``None``.

    Examples::

        >>> import elips
        >>> db = elips.connect(":memory:", dimension=2)
        >>> db.config.dimension_val
        2
        >>> db.close()
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
    def native(self) -> Database:
        r"""native -> Database

        The underlying native handle: the explicit escape hatch.

        Everything the high-level API does goes through this object, so
        reaching for it is supported rather than a workaround. It is spelled
        out rather than exposed implicitly so that dropping to the low-level
        API is a visible decision at the call site.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> vault = db.native.vault("documents")
            >>> vault.count()
            0
            >>> db.close()
        """

        return self._db

    #: Prior name for :attr:`native`.
    raw = native

    @property
    def config(self) -> Config:
        r"""config -> Config

        Return the effective database configuration.
        """

        return self._db.config

    def collection(
        self,
        name: str,
        *,
        embedder: Embedder | None = None,
        text_slot: str = DEFAULT_TEXT_SLOT,
    ) -> Collection:
        r"""collection(name, *, embedder=None, text_slot="__elips_text__") -> Collection

        Open a named collection, creating it on first write.

        There is no separate "create collection" step: vaults are created
        lazily, so naming one that does not exist yet is not an error.

        Args:
            name (str): Collection name.
            embedder (Embedder, optional): Per-collection embedder override.
                Default: ``None``.
            text_slot (str, optional): Reserved compatibility argument from the
                early wrapper design. The current document-aware runtime stores
                text on :class:`elips.DocumentAttachment` rather than mirroring
                it into metadata. Default: ``"__elips_text__"``.

        Returns:
            Collection: Typed collection wrapper.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> docs.name
            'documents'
            >>> db.close()
        """

        return Collection(
            self._db,
            name,
            embedder=embedder if embedder is not None else self._default_embedder,
            text_slot=text_slot,
        )

    #: Prior name for :meth:`collection`.
    arena = collection

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

        Reclaim index space held by deleted records across every collection.

        Deletes leave tombstones so graph navigation stays intact. Each collection
        compacts itself once tombstones pass its configured
        ``compaction_ratio``, but after a bulk delete it is worth reclaiming
        immediately. Unlike :meth:`compact`, this does not rewrite the on-disk
        snapshot and works on in-memory databases.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> key = docs.write(text="alpha note")
            >>> _ = docs.discard([key])
            >>> db.vacuum()
            >>> docs.pending_removals
            0
            >>> db.close()
        """

        self._db.vacuum()

    def collection_names(self) -> list[str]:
        r"""collection_names() -> list[str]

        Return the names of every collection that currently exists.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> _ = db.collection("documents").add(text="alpha note")
            >>> db.collection_names()
            ['documents']
            >>> db.close()
        """

        return self._db.list_vaults()

    #: Prior name for :meth:`collection_names`.
    vault_names = collection_names

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
            >>> db = elips.connect(path, dimension=2)
            >>> _ = db.collection("documents").add(vector=[1.0, 0.0])
            >>> [record.op for record in db.pending_writes()]
            ['insert']
            >>> db.close()
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

        Checkpoints, releases the cross-process lock, and seals every collection so
        later writes raise instead of silently failing to persist.
        """

        self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["DEFAULT_TEXT_SLOT", "Engine"]
