"""Authored external embedding backend with a small inference memory budget."""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BoundedEmbeddingModel:
    def __init__(self, **options: Any) -> None:
        assert options["local_files_only"] is True
        assert options["providers"] == ["CPUExecutionProvider"]

    def embed(self, texts: list[str], *, batch_size: int = 256) -> Iterator[NDArray[np.float64]]:
        assert 1 <= batch_size <= 16, "Unbounded inference batches exceed the memory budget"
        for _ in texts:
            yield np.array([1.0, 0.0])
