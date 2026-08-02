r"""Connection helpers that compose a native :class:`elips.Config`.

:func:`connect` takes keyword arguments and builds the config for you;
:func:`connect_with_config` accepts a pre-built config for callers that need
options the keyword form does not cover.

Examples::

    >>> from elips import connect
    >>> db = connect(":memory:", dimension=2, metric="cosine")
    >>> db.vault_names()
    []
    >>> db.close()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elips._internal.protocols import Embedder
from elips._native import (
    Config,
    LocalEmbedderConfig,
    QuantParams,
    open_with_config as _open_with_config,
)
from elips.database.engine import Engine
from elips.typing import AccessModeName, CodecName, IndexName, MetricName

if TYPE_CHECKING:
    from elips._native import GpuConfig


def _attach_embedder(
    config: Config,
    *,
    embedder: Embedder | LocalEmbedderConfig | None,
    embedder_provider: str,
    embedder_model: str,
    embedder_revision: str,
    dimension: int = 0,
) -> Embedder | None:
    runtime_embedder: Embedder | None = None
    if isinstance(embedder, LocalEmbedderConfig):
        config.local_text_embedder(embedder)
    elif embedder is not None:
        config.text_embedder(
            embedder,
            provider=embedder_provider,
            model=embedder_model,
            revision=embedder_revision,
            dimension=dimension,
        )
        runtime_embedder = embedder
    return runtime_embedder


def connect(
    path: str,
    *,
    dimension: int = 0,
    metric: MetricName = "cosine",
    index: IndexName = "graph",
    access_mode: AccessModeName = "read_write",
    segmented_storage: bool = True,
    metadata_acceleration: bool = True,
    embedder: Embedder | LocalEmbedderConfig | None = None,
    embedder_provider: str = "python",
    embedder_model: str = "callable",
    embedder_revision: str = "",
    use_default_text_embedder: bool = True,
    quantization: QuantParams | CodecName | None = None,
    gpu: GpuConfig | None = None,
) -> Engine:
    r"""connect(path, *, dimension=0, metric="cosine", index="graph", access_mode="read_write", segmented_storage=True, metadata_acceleration=True, embedder=None, embedder_provider="python", embedder_model="callable", embedder_revision="", use_default_text_embedder=True, quantization=None, gpu=None) -> Engine

    Open a high-level :class:`Engine` wrapper.

    Args:
        path (str): Filesystem path or ``":memory:"``.
        dimension (int, optional): Vector dimension for new databases. Default:
            ``0``.
        metric ("cosine" | "euclidean" | "dot_product", optional): Similarity
            metric. Default: ``"cosine"``.
        index ("graph" | "exact", optional): Index backend. Default:
            ``"graph"``.
        access_mode ("read_write" | "read_only", optional): Database access
            mode. Default: ``"read_write"``.
        segmented_storage (bool, optional): Whether to use segmented
            persistence. Default: ``True``.
        metadata_acceleration (bool, optional): Whether to enable the metadata
            accelerator. Default: ``True``.
        embedder (Embedder or LocalEmbedderConfig, optional): Python batch
            embedder or native local embedder config. Default: ``None``.
        embedder_provider (str, optional): Provider metadata for Python
            embedders. Default: ``"python"``.
        embedder_model (str, optional): Model metadata for Python embedders.
            Default: ``"callable"``.
        embedder_revision (str, optional): Revision metadata for Python
            embedders. Default: ``""``.
        use_default_text_embedder (bool, optional): Whether to auto-attach the
            built-in local embedder for new databases. Default: ``True``.
        quantization (QuantParams or str, optional): Vector compression codec,
            either a :class:`QuantParams` or a bare name such as ``"sq8"``.
            Selecting one does not compress anything by itself -- call
            :meth:`elips.Collection.compress` once the arena holds representative
            data. Default: ``None``.
        gpu (GpuConfig, optional): GPU runtime configuration. Default:
            ``None``.

        Returns:
            Engine: High-level engine wrapper.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> key = docs.write(text="alpha note", meta={"kind": "design"})
            >>> isinstance(key, str)
            True
            >>> db.close()
    """

    config = (
        Config()
        .dimension(dimension)
        .metric(metric)
        .index(index)
        .access_mode(access_mode)
        .segmented_storage(segmented_storage)
        .metadata_acceleration(metadata_acceleration)
        .auto_text_embedder(use_default_text_embedder)
    )

    if quantization is not None:
        config.quantization(
            quantization
            if isinstance(quantization, QuantParams)
            else QuantParams(codec=quantization)
        )

    if gpu is not None:
        if not hasattr(config, "gpu"):
            raise ValueError("gpu config requires ELIPS to be built with GPU bindings")
        config.gpu(gpu)

    runtime_embedder = _attach_embedder(
        config,
        embedder=embedder,
        embedder_provider=embedder_provider,
        embedder_model=embedder_model,
        embedder_revision=embedder_revision,
        dimension=dimension,
    )
    return Engine(_open_with_config(path, config), default_embedder=runtime_embedder)


def connect_with_config(
    path: str,
    config: Config,
    *,
    embedder: Embedder | LocalEmbedderConfig | None = None,
    embedder_provider: str = "python",
    embedder_model: str = "callable",
    embedder_revision: str = "",
) -> Engine:
    r"""connect_with_config(path, config, *, embedder=None, embedder_provider="python", embedder_model="callable", embedder_revision="") -> Engine

    Open an :class:`Engine` wrapper from a prepared :class:`elips.Config`.

    Args:
        path (str): Filesystem path or ``":memory:"``.
        config (Config): Prepared configuration builder.
        embedder (Embedder or LocalEmbedderConfig, optional): Optional runtime
            embedder added only when ``config`` does not already contain one.
            Default: ``None``.
        embedder_provider (str, optional): Provider metadata for Python
            embedders. Default: ``"python"``.
        embedder_model (str, optional): Model metadata for Python embedders.
            Default: ``"callable"``.
        embedder_revision (str, optional): Revision metadata for Python
            embedders. Default: ``""``.

    Returns:
        Engine: High-level engine wrapper.

    Examples::

        >>> import elips
        >>> config = elips.Config().dimension(2).metric("cosine")
        >>> db = elips.connect_with_config(":memory:", config)
        >>> db.config.metric_val
        'cosine'
        >>> db.close()
    """

    runtime_embedder: Embedder | None = None
    if embedder is not None and not config.has_text_embedder:
        runtime_embedder = _attach_embedder(
            config,
            embedder=embedder,
            embedder_provider=embedder_provider,
            embedder_model=embedder_model,
            embedder_revision=embedder_revision,
        )
    elif embedder is not None and not isinstance(embedder, LocalEmbedderConfig):
        runtime_embedder = embedder

    return Engine(
        _open_with_config(path, config),
        default_embedder=runtime_embedder,
    )


__all__ = ["connect", "connect_with_config"]
