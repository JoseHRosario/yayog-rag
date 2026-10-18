"""External local OCR boundary. Adapters must return verbatim OCR output."""

from pathlib import Path
from typing import Any, Protocol


class OCR(Protocol):
    @property
    def identity(self) -> dict[str, Any]: ...

    def recognize(self, image: Path) -> str: ...
