from collections.abc import Callable

from authored_embedder import AuthoredEmbedder


class CatalogChangingEmbedder(AuthoredEmbedder):
    def __init__(self, change: Callable[[], None]) -> None:
        super().__init__()
        self.change = change

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.change()
        return super().embed(texts)
