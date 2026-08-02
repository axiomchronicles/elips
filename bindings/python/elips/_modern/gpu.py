r"""Typed GPU accelerator facade for the modern ELIPS API.

The low-level bindings expose GPU control as free functions plus a
:class:`elips.GpuDevice` handle. This module wraps them in the same
dataclass-and-context-manager style as the rest of the modern API::

    >>> import elips
    >>> specs = elips.accelerators()
    >>> all(spec.name and spec.backend for spec in specs)
    True

Nothing here requires a GPU to import. :func:`accelerators` returns an empty
list and :func:`accelerator` returns ``None`` on CPU-only machines, so callers
branch on the result rather than catching an exception.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING

from ..types import DistanceMatrix, MetricName, TopKResult, VectorBatch

if TYPE_CHECKING:
    from ..core import GpuConfig, GpuDevice, GpuDeviceInfo

try:  # pragma: no cover - depends on build flags
    from ..core import (
        GpuConfig as _GpuConfig,
        gpu_can_fit_index as _can_fit,
        gpu_devices as _devices,
        gpu_select as _select,
    )

    _GPU_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only build
    _GPU_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class AcceleratorSpec:
    r"""AcceleratorSpec(*, name, backend, index, memory_bytes, free_memory_bytes, unified_memory, supports_fp16, raw) -> AcceleratorSpec

    Immutable description of one detected GPU.

    Args:
        name (str): Device name, e.g. ``"Apple M4"``.
        backend (str): Backend that drives it: ``"metal"``, ``"cuda"``,
            ``"hip"``, ``"sycl"``, or ``"vulkan"``.
        index (int): Device ordinal, for multi-GPU hosts.
        memory_bytes (int): Total device memory.
        free_memory_bytes (int): Device memory currently free.
        unified_memory (bool): Whether host and device share one address space,
            as on Apple silicon. When true, transfers are far cheaper and
            larger-than-VRAM indexes become practical.
        supports_fp16 (bool): Whether half-precision search is available.
        raw (GpuDeviceInfo): The underlying low-level device info, for the
            fields this summary omits.

    Examples::

        >>> import elips
        >>> specs = elips.accelerators()
        >>> all(spec.memory_gb > 0 for spec in specs)
        True
    """

    name: str
    backend: str
    index: int
    memory_bytes: int
    free_memory_bytes: int
    unified_memory: bool
    supports_fp16: bool
    raw: GpuDeviceInfo

    @property
    def memory_gb(self) -> float:
        r"""memory_gb -> float

        Total device memory in GiB.
        """

        return self.memory_bytes / (1024.0**3)

    def can_fit(
        self,
        n_vectors: int,
        dimension: int,
        *,
        config: GpuConfig | None = None,
    ) -> bool:
        r"""can_fit(n_vectors, dimension, *, config=None) -> bool

        Whether an index of this shape is expected to fit in device memory.

        Check this before a long ingest rather than discovering the ceiling
        part-way through a multi-hour build.

        Args:
            n_vectors (int): Number of vectors the index will hold.
            dimension (int): Vector dimensionality.
            config (GpuConfig, optional): Configuration whose precision and pool
                settings affect the estimate. Default: ``None``.

        Returns:
            bool: True when the index should fit.

        Examples::

            >>> import elips
            >>> specs = elips.accelerators()
            >>> all(isinstance(spec.can_fit(1000, 128), bool) for spec in specs)
            True
        """

        if not _GPU_AVAILABLE:
            return False
        return _can_fit(
            self.raw,
            n_vectors,
            dimension,
            config if config is not None else _GpuConfig(),
        )

    @classmethod
    def from_info(cls, info: GpuDeviceInfo) -> AcceleratorSpec:
        r"""from_info(info) -> AcceleratorSpec

        Summarize a low-level :class:`elips.GpuDeviceInfo`.
        """

        return cls(
            name=info.name,
            backend=info.backend,
            index=info.device_index,
            memory_bytes=info.total_memory_bytes,
            free_memory_bytes=info.free_memory_bytes,
            unified_memory=info.has_unified_memory,
            supports_fp16=info.supports_fp16,
            raw=info,
        )


class Accelerator:
    r"""Accelerator(device) -> Accelerator

    A live GPU handle for standalone kernel work.

    Wraps :class:`elips.GpuDevice`, which owns the backend independently of any
    :class:`elips.Engine`. Use it as a context manager so the backend and its
    memory pool are released deterministically::

        with elips.accelerator() as gpu:
            distances = gpu.distances(queries, corpus, metric="cosine")
            indices, values = gpu.nearest(distances, top=10)

    Raw device allocation is deliberately unavailable: it takes a
    caller-supplied byte count against a raw pointer, where a wrong value
    corrupts device memory instead of raising. Every method here derives its
    sizes from the sequences passed in.

    Args:
        device (GpuDevice): Low-level handle from :func:`elips.gpu_select`.
    """

    def __init__(self, device: GpuDevice) -> None:
        self._device = device

    @property
    def raw(self) -> GpuDevice:
        r"""raw -> GpuDevice

        The underlying low-level device handle.
        """

        return self._device

    @property
    def spec(self) -> AcceleratorSpec:
        r"""spec -> AcceleratorSpec

        A typed summary of the device this handle drives.
        """

        return AcceleratorSpec.from_info(self._device.device_info)

    @property
    def backend(self) -> str:
        r"""backend -> str

        Backend name, e.g. ``"metal"`` or ``"cuda"``.
        """

        return self._device.backend

    @property
    def idle(self) -> bool:
        r"""idle -> bool

        Whether all submitted work has completed.
        """

        return self._device.idle

    @property
    def closed(self) -> bool:
        r"""closed -> bool

        Whether :meth:`close` has run.
        """

        return self._device.closed

    def reserve(self, pool_bytes: int = 0) -> None:
        r"""reserve(pool_bytes=0) -> None

        Size the device memory pool up front.

        Doing this once avoids the suballocator having to grow mid-run, which is
        the difference between a steady ingest and a stuttering one.

        Args:
            pool_bytes (int, optional): Pool size in bytes. ``0`` uses 80% of
                device memory. Default: ``0``.
        """

        self._device.memory.initialize(pool_bytes)

    def distances(
        self,
        queries: VectorBatch,
        corpus: VectorBatch,
        *,
        metric: MetricName = "cosine",
    ) -> DistanceMatrix:
        r"""distances(queries, corpus, *, metric="cosine") -> DistanceMatrix

        Compute every pairwise distance between two batches, on the GPU.

        Args:
            queries (Sequence[Sequence[float]]): Query vectors, uniform length.
            corpus (Sequence[Sequence[float]]): Corpus vectors, same dimension.
            metric ("cosine" | "euclidean" | "dot_product", optional):
                Similarity metric. Default: ``"cosine"``.

        Returns:
            DistanceMatrix: ``len(queries)`` rows of ``len(corpus)`` distances,
            ordering-normalized so smaller always means closer.

        Raises:
            DimensionMismatch: If query and corpus dimensions differ.
            StorageError: If the GPU kernel fails, or the handle is closed.

        Examples::

            >>> import elips
            >>> gpu = elips.accelerator()
            >>> if gpu is not None:
            ...     with gpu:
            ...         rows = gpu.distances([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
            ...         print(len(rows), len(rows[0]))
            1 2
        """

        return self._device.compute_distances(queries, corpus, metric)

    def nearest(self, distances: DistanceMatrix, *, top: int) -> TopKResult:
        r"""nearest(distances, *, top) -> TopKResult

        Select the ``top`` closest entries per row, on the GPU.

        Args:
            distances (DistanceMatrix): Rows of distances, typically from
                :meth:`distances`.
            top (int): Results per row; must not exceed the row width.

        Returns:
            TopKResult: An ``(indices, values)`` pair of parallel row lists,
            ascending by distance.

        Raises:
            ValueError: If ``top`` exceeds the row width.
            StorageError: If the GPU kernel fails, or the handle is closed.
        """

        return self._device.top_k(distances, top)

    def search(
        self,
        queries: VectorBatch,
        corpus: VectorBatch,
        *,
        top: int,
        metric: MetricName = "cosine",
    ) -> TopKResult:
        r"""search(queries, corpus, *, top, metric="cosine") -> TopKResult

        Brute-force nearest-neighbour search over an in-memory corpus.

        Combines :meth:`distances` and :meth:`nearest` into the exhaustive-search
        path: exact results, no index to build, and no recall loss. Sensible up
        to corpora that fit comfortably in device memory; beyond that, build a
        real index by passing a ``gpu`` config to :func:`elips.connect`.

        Args:
            queries (Sequence[Sequence[float]]): Query vectors.
            corpus (Sequence[Sequence[float]]): Corpus vectors to search.
            top (int): Neighbours to return per query.
            metric ("cosine" | "euclidean" | "dot_product", optional):
                Similarity metric. Default: ``"cosine"``.

        Returns:
            TopKResult: ``(indices, values)``, where each index refers to a row
            of ``corpus``.

        Examples::

            >>> import elips
            >>> gpu = elips.accelerator()
            >>> if gpu is not None:
            ...     with gpu:
            ...         corpus = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
            ...         idx, _ = gpu.search([[1.0, 0.0]], corpus, top=2)
            ...         print(idx[0][0])
            0
        """

        return self.nearest(self.distances(queries, corpus, metric=metric), top=top)

    def synchronize(self) -> None:
        r"""synchronize() -> None

        Block until all submitted device work completes.
        """

        self._device.synchronize()

    def memory_usage(self) -> tuple[int, int, int]:
        r"""memory_usage() -> tuple[int, int, int]

        Return ``(used, available, peak)`` pool bytes.

        ``available`` counts reachable bytes -- the free list plus uncommitted
        pool headroom -- so it never over-reports what a subsequent allocation
        could actually obtain.
        """

        memory = self._device.memory
        return (
            memory.bytes_used,
            memory.bytes_available,
            memory.peak_bytes_used,
        )

    def kernel_timings(self, limit: int = 100) -> Sequence[object]:
        r"""kernel_timings(limit=100) -> Sequence[KernelTiming]

        Return up to ``limit`` recent kernel timings, newest last.

        Args:
            limit (int, optional): Maximum entries to return. Default: ``100``.
        """

        return self._device.profiler.recent_timings(limit)

    def close(self) -> None:
        r"""close() -> None

        Release the backend and its memory pool. Idempotent.
        """

        self._device.close()

    def __enter__(self) -> Accelerator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        if self._device.closed:
            return "<Accelerator closed>"
        spec = self.spec
        return f"<Accelerator {spec.name!r} backend={spec.backend}>"


def accelerators() -> list[AcceleratorSpec]:
    r"""accelerators() -> list[AcceleratorSpec]

    Describe every GPU this build can drive.

    Returns:
        list[AcceleratorSpec]: One entry per usable device. Empty on CPU-only
        machines and on builds compiled without a GPU backend, which is a normal
        outcome rather than an error.

    Examples::

        >>> import elips
        >>> specs = elips.accelerators()
        >>> isinstance(specs, list)
        True
    """

    if not _GPU_AVAILABLE:
        return []
    return [AcceleratorSpec.from_info(info) for info in _devices()]


def accelerator(config: GpuConfig | None = None) -> Accelerator | None:
    r"""accelerator(config=None) -> Optional[Accelerator]

    Select and initialize a GPU for standalone kernel work.

    Args:
        config (GpuConfig, optional): Selection policy and tuning. The default
            picks the best available device. Default: ``None``.

    Returns:
        Accelerator or None: A live handle, or ``None`` when no compatible
        device exists. Branch on the result; do not assume a device is present.

    Examples::

        >>> import elips
        >>> gpu = elips.accelerator()
        >>> if gpu is None:
        ...     ran_on_gpu = False           # CPU-only machine: expected
        ... else:
        ...     with gpu:
        ...         ran_on_gpu = gpu.backend != ""
        >>> isinstance(ran_on_gpu, bool)
        True
    """

    if not _GPU_AVAILABLE:
        return None
    device = _select(config if config is not None else _GpuConfig())
    if device is None:
        return None
    return Accelerator(device)


__all__ = ["Accelerator", "AcceleratorSpec", "accelerator", "accelerators"]
