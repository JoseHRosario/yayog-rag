from pathlib import Path

import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR
from review_changing_embedder import ReviewChangingEmbedder
from test_recovery import damaged_source

from cited_retrieval import ingest, recover, review, search


def test_review_revoked_during_query_inference_cannot_be_served(tmp_path: Path):
    source, bundle, index = (
        damaged_source(tmp_path),
        tmp_path / "bundle.json",
        tmp_path / "index.sqlite",
    )
    result = recover(source, bundle, 1, 1, AuthoredOCR())
    region = result["regions"][0]
    review(bundle, region["id"], region["revision"], "accepted")
    approval = {"sha256": result["source_sha256"], "pages": [1]}
    ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    changing_model = ReviewChangingEmbedder(bundle, region["id"], region["revision"])
    with pytest.raises(ValueError, match="stale_recovery"):
        search(index, "seconds", changing_model)


def test_review_revoked_during_rebuild_preserves_previous_index(tmp_path: Path):
    source, bundle, index = (
        damaged_source(tmp_path),
        tmp_path / "bundle.json",
        tmp_path / "index.sqlite",
    )
    result = recover(source, bundle, 1, 1, AuthoredOCR())
    region = result["regions"][0]
    review(bundle, region["id"], region["revision"], "accepted")
    approval = {"sha256": result["source_sha256"], "pages": [1]}
    ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    before = index.read_bytes()
    changing_model = ReviewChangingEmbedder(bundle, region["id"], region["revision"])
    with pytest.raises(ValueError, match="stale_recovery"):
        ingest(source, index, 1, 1, changing_model, approval=approval, recovery=bundle)
    assert index.read_bytes() == before
