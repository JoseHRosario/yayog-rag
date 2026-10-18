from pathlib import Path

from authored_embedder import AuthoredEmbedder

from cited_retrieval import review


class ReviewChangingEmbedder(AuthoredEmbedder):
    def __init__(self, bundle: Path, region_id: str, revision: str) -> None:
        super().__init__()
        self.bundle, self.region_id, self.revision = bundle, region_id, revision

    def embed(self, texts: list[str]) -> list[list[float]]:
        review(self.bundle, self.region_id, self.revision, "rejected")
        return super().embed(texts)
