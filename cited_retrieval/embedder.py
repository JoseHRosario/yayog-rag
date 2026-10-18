from typing import Protocol


class Embedder(Protocol):
    @property
    def identity(self) -> dict[str, str]: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
