"""Type aliases for the ELIPS Python SDK.

These aliases are provided for IDE convenience and documentation. They do not
add runtime behavior; all types map directly to the C++ extension types.

Usage::

    from elips.types import MetaValue, Vector, PayloadLike

    def embed(text: str) -> Vector:
        ...
"""

from typing import Iterable, List, Mapping, Sequence, Tuple, Union

#: A metadata value: integer, float, boolean, or string.
MetaValue = Union[bool, int, float, str]

#: A vector: any sequence of 32-bit floats.
Vector = Sequence[float]

#: A metadata payload: mapping of string keys to typed values.
PayloadLike = Mapping[str, MetaValue]

#: One batch record for ``place_many()``.
#: Must have at least ``vector``; ``data`` and ``id`` are optional.
BatchRecord = Mapping[str, Union[Vector, PayloadLike, str, None]]

#: A fetch result: dict with ``id``, ``vector``, and ``data`` keys.
FetchResult = Mapping[str, Union[str, Vector, PayloadLike, None]]

#: A scan result: dict with ``id`` and ``data`` keys.
ScanResult = Mapping[str, Union[str, PayloadLike]]

#: A query binding: mapping of variable names to vectors.
QueryBindings = Mapping[str, Vector]

#: A record for place_many: id, vector, and payload dicts.
RecordDict = Mapping[str, Union[str, Vector, PayloadLike, None]]

__all__ = [
    "MetaValue",
    "Vector",
    "PayloadLike",
    "BatchRecord",
    "FetchResult",
    "ScanResult",
    "QueryBindings",
    "RecordDict",
]