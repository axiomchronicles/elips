r"""Vector compression: codec selection and its parameters.

Selecting a codec does not compress anything by itself. Product quantization
cannot encode the first record -- a codebook has to be learned from real data
first -- so compression is an explicit transition: configure the codec when
opening, then call :meth:`elips.Collection.compress` once the collection holds
representative data.

:meth:`QuantParams.code_bytes` reports the compressed width before any training
happens, which is what sizing a deployment needs.

Examples::

    >>> from elips.quantization import Codec, QuantParams
    >>> params = QuantParams(codec="pq", pq_dim=24)
    >>> params.code_bytes(768)          # 768 fp32 -> 24 bytes
    24
    >>> params.codec_enum == Codec.pq
    True
"""

from __future__ import annotations

from elips._native import Codec, QuantParams

__all__ = ["Codec", "QuantParams"]
