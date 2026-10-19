"""Fail-closed opt-in image evidence through public source/index boundaries."""

from pathlib import Path
from typing import Any

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR
from input_mutating_embedder import InputMutatingEmbedder

from cited_retrieval import ingest, inspect_review, recover, report, review, search


def fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    source, bundle = tmp_path / "source.pdf", tmp_path / "bundle.json"
    with pymupdf.open() as drawing:
        page = drawing.new_page(width=400, height=200)
        page.insert_text((20, 60), "Rest for 12 seconds before resuming.")
        image = page.get_pixmap().tobytes("png")
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=200)
        page.insert_image(page.rect, stream=image)
        doc.save(source)
    result = recover(
        source,
        bundle,
        1,
        1,
        AuthoredOCR(),
        image_regions=[{"id": "one", "pdf_page": 1, "bbox": [10, 35, 350, 75]}],
    )
    return source, bundle, result


@pytest.mark.parametrize("mutation", ["source", "crop"])
@pytest.mark.parametrize("during_search", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_image_graph_mutation_during_embedding_fails_closed(
    tmp_path: Path, mutation: str, during_search: bool, fail: bool
) -> None:
    source, bundle, draft = fixture(tmp_path)
    region = draft["regions"][0]
    review(bundle, "one", region["revision"], "accepted")
    index = tmp_path / "index.sqlite"
    approval = {"sha256": draft["source_sha256"], "pages": [1]}
    ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    before = index.read_bytes()
    target = source if mutation == "source" else tmp_path / region["crop"]

    def mutate() -> None:
        target.write_bytes(target.read_bytes() + b"\n% authored mutation\n")

    with pytest.raises(ValueError, match="stale_recovery|stale_source|source_changed"):
        embedder = InputMutatingEmbedder(mutate, fail=fail)
        if during_search:
            search(index, "rest seconds", embedder)
        else:
            ingest(source, index, 1, 1, embedder, approval=approval, recovery=bundle)
    assert index.read_bytes() == before


@pytest.mark.parametrize("mode", ["duplicate", "overlap", "native"])
def test_image_configuration_rejects_native_shadow_and_overlap_before_writes(
    tmp_path: Path, mode: str
) -> None:
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text(
            (50, 70), "Healthy native prose remains owned by the native workflow."
        )
        doc.save(source)
    roi = {"id": "one", "pdf_page": 1, "bbox": [10, 200, 350, 250]}
    second = {**roi, "id": "one" if mode == "duplicate" else "two"}
    if mode == "native":
        second["bbox"] = [45, 50, 500, 80]
    engine = AuthoredOCR()
    bundle = tmp_path / "new" / "bundle.json"
    before = source.read_bytes()
    with pytest.raises(ValueError, match="invalid_image_regions"):
        recover(source, bundle, 1, 1, engine, image_regions=[roi, second])
    assert not bundle.parent.exists() and not engine.crops
    assert source.read_bytes() == before


@pytest.mark.parametrize("change", ["engine", "roi", "source"])
def test_changed_image_revision_resets_review_and_keeps_history(
    tmp_path: Path, change: str
) -> None:
    source, bundle, draft = fixture(tmp_path)
    region = draft["regions"][0]
    review(bundle, "one", region["revision"], "accepted")
    engine = AuthoredOCR()
    roi = {"id": "one", "pdf_page": 1, "bbox": [10, 35, 350, 75]}
    if change == "engine":
        engine.identity["version"] = "2"
    elif change == "roi":
        roi["bbox"] = [11, 35, 350, 75]
    else:
        source.write_bytes(source.read_bytes() + b"\n% authored new source revision\n")
    result = recover(source, bundle, 1, 1, engine, image_regions=[roi])
    revised = result["regions"][0]
    assert revised["state"] == "pending" and revised["revision"] != region["revision"]
    assert any(
        h["state"] == "accepted" and h["revision"] == region["revision"] for h in revised["history"]
    )
    assert inspect_review(bundle) == result


def test_index_cannot_replace_recovery_crop(tmp_path: Path) -> None:
    source, bundle, draft = fixture(tmp_path)
    region = draft["regions"][0]
    review(bundle, "one", region["revision"], "accepted")
    crop = tmp_path / region["crop"]
    before = crop.read_bytes()
    with pytest.raises(ValueError, match="unsafe_index_path"):
        ingest(
            source,
            crop,
            1,
            1,
            AuthoredEmbedder(),
            approval={"sha256": draft["source_sha256"], "pages": [1]},
            recovery=bundle,
        )
    assert crop.read_bytes() == before


def test_image_review_revocation_invalidates_index(tmp_path: Path) -> None:
    source, bundle, draft = fixture(tmp_path)
    region = draft["regions"][0]
    review(bundle, "one", region["revision"], "accepted")
    index = tmp_path / "index.sqlite"
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": draft["source_sha256"], "pages": [1]},
        recovery=bundle,
    )
    review(bundle, "one", region["revision"], "rejected")
    with pytest.raises(ValueError, match="stale_recovery"):
        report(index)
