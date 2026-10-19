"""Mutate real inputs only across the external embedding boundary."""

from collections.abc import Callable

from authored_embedder import AuthoredEmbedder


class InputMutatingEmbedder(AuthoredEmbedder):
    def __init__(self, mutation: Callable[[], None], *, fail: bool = False) -> None:
        super().__init__()
        self.mutation = mutation
        self.fail = fail

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.mutation()
        if self.fail:
            raise RuntimeError("authored external failure")
        return super().embed(texts)
