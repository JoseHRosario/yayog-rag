from pathlib import Path

import pytest

from cited_retrieval.local_embedder import LocalEmbedder


def test_missing_local_model_fails_closed_without_semantic_substitute(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_unavailable"):
        LocalEmbedder(tmp_path / "absent")
