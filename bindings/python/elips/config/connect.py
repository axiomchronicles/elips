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

import os
import warnings
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


def _build_gpu_config(device: str, *, fallback: bool) -> GpuConfig:
    r"""Parse a device string into a native :class:`GpuConfig`.

    Grammar (case-insensitive)::

        "cpu"              -> CpuOnly
        "auto"             -> Auto (GPU if present, else logged CPU fallback)
        "metal"            -> Specific metal
        "cuda:N" / "hip:N" -> Specific backend, device index N
        "cuda" / "hip"     -> Specific backend, any index

    A named backend that is absent raises ``BackendUnavailableError`` at open
    time unless ``fallback`` is true, which demotes to CPU with a warning.
    """
    from elips._native import GpuConfig, GpuPolicy

    spec = device.strip().lower()
    cfg = GpuConfig()
    cfg.fallback = fallback

    if spec in ("cpu", ""):
        cfg.policy = GpuPolicy.cpu_only
        return cfg
    if spec == "auto":
        cfg.policy = GpuPolicy.auto
        return cfg

    backend, _, index = spec.partition(":")
    if backend not in ("metal", "cuda", "hip"):
        raise ValueError(
            f"device: unknown backend {device!r}; expected one of "
            "'cpu', 'auto', 'metal', 'cuda[:N]', 'hip[:N]'"
        )
    cfg.policy = GpuPolicy.specific
    cfg.preferred_backend = backend
    if index:
        try:
            cfg.device_index = int(index)
        except ValueError as exc:
            raise ValueError(
                f"device: {device!r} has a non-integer device index"
            ) from exc
    return cfg


def _wanted_gpu(gpu: GpuConfig) -> bool:
    """True when this config asked for a GPU (anything but CPU-only)."""
    from elips._native import GpuPolicy

    return gpu.policy != GpuPolicy.cpu_only


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
    device: str | None = None,
    fallback: bool = False,
) -> Engine:
    r"""connect(path, *, dimension=0, metric="cosine", index="graph", access_mode="read_write", segmented_storage=True, metadata_acceleration=True, embedder=None, embedder_provider="python", embedder_model="callable", embedder_revision="", use_default_text_embedder=True, quantization=None, gpu=None) -> Engine

    Open a high-level :class:`Engine` wrapper.

    .. deprecated:: 1.2
        Use :class:`elips.Client` (ephemeral) or :class:`elips.PersistentClient`
        (on disk) instead. ``connect`` remains as a working shim.

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
        device (str, optional): Device string -- ``"cpu"``, ``"auto"``,
            ``"metal"``, ``"cuda:N"``, or ``"hip:N"`` -- compiled into a
            :class:`GpuConfig`. Consulted (with ``ELIPS_DEVICE`` as the
            environment fallback) only when ``gpu=`` is not given. A named GPU
            backend that is absent raises :class:`BackendUnavailableError`
            unless ``fallback=True``. Default: ``None``.
        fallback (bool, optional): Allow a logged demotion to CPU when the
            requested GPU backend is unavailable. Default: ``False``.

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

    # GPU resolution priority: explicit ``gpu=`` wins, then ``device=``, then
    # the ELIPS_DEVICE environment override.
    has_gpu_bindings = hasattr(config, "gpu")
    resolved_gpu: GpuConfig | None = None

    if gpu is not None:
        if not has_gpu_bindings:
            raise ValueError("gpu config requires ELIPS to be built with GPU bindings")
        config.gpu(gpu)
        resolved_gpu = gpu
    else:
        dev = device if device is not None else os.environ.get("ELIPS_DEVICE")
        if dev is not None:
            if has_gpu_bindings:
                resolved_gpu = _build_gpu_config(dev, fallback=fallback)
                config.gpu(resolved_gpu)
            elif dev.strip().lower() not in ("cpu", "auto", ""):
                # CPU-only build: a named GPU backend can only run on CPU.
                msg = (
                    f"device {dev!r} requested but ELIPS was built without "
                    "GPU support"
                )
                if not fallback:
                    from elips.exceptions import BackendUnavailableError

                    raise BackendUnavailableError(msg)
                warnings.warn(f"{msg}; running on CPU", stacklevel=2)

    runtime_embedder = _attach_embedder(
        config,
        embedder=embedder,
        embedder_provider=embedder_provider,
        embedder_model=embedder_model,
        embedder_revision=embedder_revision,
        dimension=dimension,
    )
    engine = Engine(_open_with_config(path, config), default_embedder=runtime_embedder)

    # No-silent-fallback: when a GPU was requested but the engine opened on CPU
    # (an allowed/Auto demotion), surface a Python warning. The native layer
    # also logs the fallback to stderr.
    if resolved_gpu is not None and _wanted_gpu(resolved_gpu):
        info = engine.native.gpu_info()
        if getattr(info, "backend", "cpu") == "cpu":
            warnings.warn(
                f"GPU backend {resolved_gpu.preferred_backend or 'auto'!r} "
                "requested but unavailable; running on CPU",
                stacklevel=2,
            )
    return engine


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

    .. deprecated:: 1.2
        Use :class:`elips.Client` / :class:`elips.PersistentClient`. This
        remains as a working shim.

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
