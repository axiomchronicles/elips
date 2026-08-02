r"""Durability: the write-ahead log and the snapshots it folds into.

:func:`replay_wal` reads an acknowledged log without opening the database,
which is what crash forensics needs: it answers "what did the database
actually acknowledge before it died?" without mutating anything.

Checkpointing and compaction are lifecycle operations on an open handle, so
they live on :class:`~elips.Engine` rather than here.

Examples::

    >>> import elips, tempfile
    >>> path = tempfile.mkdtemp()
    >>> db = elips.connect(path, dimension=2)
    >>> _ = db.collection("documents").add(vector=[1.0, 0.0])
    >>> [record.op for record in db.pending_writes()]
    ['insert']
    >>> db.close()
"""

from __future__ import annotations

from elips._native import WalEntry, WalOp, replay_wal
from elips.records.models import WalRecord

__all__ = ["WalEntry", "WalOp", "WalRecord", "replay_wal"]
