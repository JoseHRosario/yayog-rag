"""Authored vectors at the external inference seam, never a production model."""


class AuthoredEmbedder:
    def __init__(self, fixed_vector: list[float] | None = None) -> None:
        self.fixed_vector = fixed_vector
        self.identity = {
            "model": "authored-test-vectors",
            "version": "1",
            "artifact_sha256": "a" * 64,
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fixed_vector is not None:
            return [self.fixed_vector.copy() for _ in texts]
        return [
            [1.0, 0.0] if any(w in t.lower() for w in ("sleep", "rest", "tired")) else [0.0, 1.0]
            for t in texts
        ]
