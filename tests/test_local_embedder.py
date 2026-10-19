from pathlib import Path

import pytest

from cited_retrieval.local_embedder import LocalEmbedder


def test_missing_local_model_fails_closed_without_semantic_substitute(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_unavailable"):
        LocalEmbedder(tmp_path / "absent")


def test_local_embedding_bounds_inference_batches_without_losing_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bounded_embedding_model import BoundedEmbeddingModel

    from cited_retrieval import local_embedder

    for name in (
        "model.onnx",
        "tokenizer.json",
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        (tmp_path / name).write_text("authored model boundary fixture")
    monkeypatch.setattr(local_embedder, "TextEmbedding", BoundedEmbeddingModel)
    model = LocalEmbedder(tmp_path)
    identity = model.identity
    vectors = model.embed(["Authored training reference paragraph."] * 33)
    assert len(vectors) == 33
    assert vectors[0] == [1.0, 0.0]
    assert vectors[-1] == [1.0, 0.0]
    assert model.identity == identity
