r"""The GPU accelerator facade.

Importing this package is always safe: the module degrades to an empty device
list when the extension was built without GPU bindings.
"""

from __future__ import annotations

from elips.gpu.accelerator import (
    Accelerator,
    AcceleratorSpec,
    accelerator,
    accelerators,
)

__all__ = ["Accelerator", "AcceleratorSpec", "accelerator", "accelerators"]
