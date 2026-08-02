r"""Typing helpers for the modern ELIPS API.

Examples::

    >>> from collections.abc import Sequence
    >>> from elips.modern import Embedder
    >>> def toy_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
    ...     return [[float(len(text)), 0.0] for text in texts]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from elips.typing import BatchRecord, RecordInputDict

if TYPE_CHECKING:
    from elips.records.models import RecordInput


@runtime_checkable
class Embedder(Protocol):
    r"""Embedder(texts) -> vectors

    Protocol for batch embedders accepted by :func:`connect` and
    :meth:`elips.Collection.write_many`.

    Args:
        texts (Sequence[str]): Text strings to embed.

    Returns:
        Sequence[Sequence[float]]: One embedding vector per input text.

    Examples::
        >>> from collections.abc import Sequence
        >>> def toy_embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...     return [[1.0, 0.0] for _ in texts]
    """

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


RecordInputLike: TypeAlias = "RecordInput | RecordInputDict | BatchRecord"

__all__ = ["Embedder", "RecordInputLike"]
