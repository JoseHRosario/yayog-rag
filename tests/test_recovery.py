from pathlib import Path

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR

from cited_retrieval import ingest, inspect_review, inspect_source, recover, report, review, search


def damaged_source(tmp_path: Path) -> Path:
    path = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 70), "Rest and sleep support recovery after training.")
        font = page.insert_font(fontname="damaged", fontbuffer=pymupdf.Font("cjk").buffer)
        page.insert_text(
            (50, 140), "Hold for 12 seconds. Stop if pain develops.", fontname="damaged"
        )
        doc.xref_set_key(font, "ToUnicode", "null")
        # Image adjacent to, not replacing, instructions. Never mask the text.
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
        pix.clear_with(80)
        page.insert_image(pymupdf.Rect(300, 110, 350, 160), pixmap=pix)
        doc.save(path)
    return path


def test_targeted_recovery_retains_native_and_exact_ocr_with_inspectable_crop(tmp_path: Path):
    source = damaged_source(tmp_path)
    engine = AuthoredOCR("Hold for 12 seconds. Stop if pain develops.\n")
    bundle = tmp_path / "review" / "bundle.json"
    result = recover(source, bundle, 1, 1, engine)
    assert len(engine.crops) == 1
    assert result["source_sha256"]
    assert len(result["regions"]) == 1
    region = result["regions"][0]
    assert region["reason"] == "missing_unicode_map"
    assert region["ocr_text"] == "Hold for 12 seconds. Stop if pain develops.\n"
    assert region["state"] == "pending"
    assert region["original_text"] != region["ocr_text"]
    assert region["engine"] == engine.identity
    assert region["bbox"][1] > 100
    assert region["bbox"][3] < 160
    assert (bundle.parent / region["crop"]).read_bytes() == engine.crops[0]
    assert (
        result["blocks"][0]["lines"][0]["text"] == "Rest and sleep support recovery after training."
    )
    assert result["unresolved_regions"] == 1


def test_review_gates_ingestion_and_review_changes_invalidate_existing_index(tmp_path: Path):
    source = damaged_source(tmp_path)
    bundle, index = tmp_path / "bundle.json", tmp_path / "index.sqlite"
    recovered = recover(source, bundle, 1, 1, AuthoredOCR())
    approval = {"sha256": inspect_source(source, 1, 1)["sha256"], "pages": [1]}
    pending = ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    assert pending["recovery"]["unresolved_regions"] == 1
    evidence = search(index, "recovery", AuthoredEmbedder())
    assert evidence["coverage"]["unresolved_regions"] == 1
    assert evidence["coverage"]["pages"][0]["status"] == "partial"
    passages = evidence["passages"]
    assert [p["text"] for p in passages] == ["Rest and sleep support recovery after training."]
    region = recovered["regions"][0]
    with pytest.raises(ValueError, match="review_revision_mismatch"):
        review(bundle, region["id"], "wrong-revision", "accepted")
    review(bundle, region["id"], region["revision"], "accepted")
    assert inspect_review(bundle)["regions"][0]["state"] == "accepted"
    with pytest.raises(ValueError, match="stale_recovery"):
        report(index)
    ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    accepted = search(index, "pain seconds", AuthoredEmbedder())["passages"]
    recovered_passage = next(p for p in accepted if "12 seconds" in p["text"])
    assert recovered_passage["review_status"] == "accepted"
    assert recovered_passage["provenance"]["recovery_regions"][0]["revision"] == region["revision"]
    review(bundle, region["id"], region["revision"], "rejected")
    with pytest.raises(ValueError, match="stale_recovery"):
        search(index, "seconds", AuthoredEmbedder())
    with pytest.raises(ValueError, match="stale_recovery"):
        report(index)
    ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    assert all(
        "12 seconds" not in p["text"]
        for p in search(index, "seconds", AuthoredEmbedder())["passages"]
    )


def test_only_identical_bound_artifacts_retain_acceptance(tmp_path: Path):
    source = damaged_source(tmp_path)
    bundle = tmp_path / "bundle.json"
    original = recover(source, bundle, 1, 1, AuthoredOCR())
    region = original["regions"][0]
    review(bundle, region["id"], region["revision"], "accepted")
    identical = recover(source, bundle, 1, 1, AuthoredOCR())
    assert identical["regions"][0]["state"] == "accepted"
    changed = recover(
        source, bundle, 1, 1, AuthoredOCR("Hold for 13 seconds. Stop if pain develops.")
    )
    assert changed["regions"][0]["state"] == "pending"
    assert changed["regions"][0]["history"][1]["state"] == "accepted"
    with pytest.raises(ValueError, match="review_revision_mismatch"):
        review(bundle, region["id"], region["revision"], "accepted")
    current = changed["regions"][0]
    review(bundle, current["id"], current["revision"], "accepted")
    source.write_bytes(source.read_bytes() + b"\n% new artifact")
    with pytest.raises(ValueError, match="stale_recovery"):
        inspect_review(bundle)
    rebuilt = recover(
        source, bundle, 1, 1, AuthoredOCR("Hold for 13 seconds. Stop if pain develops.")
    )
    assert rebuilt["regions"][0]["state"] == "pending"


def test_changed_crop_invalidates_review_and_index(tmp_path: Path):
    source = damaged_source(tmp_path)
    bundle, index = tmp_path / "bundle.json", tmp_path / "index.sqlite"
    result = recover(source, bundle, 1, 1, AuthoredOCR())
    region = result["regions"][0]
    review(bundle, region["id"], region["revision"], "accepted")
    approval = {"sha256": result["source_sha256"], "pages": [1]}
    ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    (bundle.parent / region["crop"]).write_bytes(b"damaged crop")
    with pytest.raises(ValueError, match="stale_recovery"):
        inspect_review(bundle)
    with pytest.raises(ValueError, match="stale_recovery"):
        report(index)
    with pytest.raises(ValueError, match="stale_recovery"):
        search(index, "seconds", AuthoredEmbedder())
