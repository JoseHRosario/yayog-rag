"""Authored semantic vectors with a finite inference window, no live model."""


class WindowedEmbedder:
    def __init__(self) -> None:
        self.identity = {
            "model": "authored-window-vectors",
            "version": "1",
            "artifact_sha256": "b" * 64,
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0]
            if any(
                phrase in text[:100].casefold()
                for phrase in ("reduce the load", "lighter resistance")
            )
            else [0.0, 1.0]
            for text in texts
        ]
