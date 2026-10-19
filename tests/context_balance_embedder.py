"""Deterministic inference-boundary double sensitive to visible topic dilution."""

import re


class ContextBalanceEmbedder:
    def __init__(self) -> None:
        self.identity = {
            "model": "authored-topic-counts",
            "version": "1",
            "artifact_sha256": "c" * 64,
        }
        self.inputs: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.inputs.extend(texts)
        vectors = []
        for text in texts:
            words = re.findall(r"\w+", text.casefold())
            wrist = sum(words.count(word) for word in ("wrist", "steady"))
            balance = sum(words.count(word) for word in ("balance", "stability"))
            vectors.append([float(wrist), float(balance), 0.0 if wrist or balance else 1.0])
        return vectors
