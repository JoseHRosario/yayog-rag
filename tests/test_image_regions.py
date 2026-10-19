from pathlib import Path

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR

from cited_retrieval import ingest, inspect_review, recover, review, search
from cited_retrieval.exercises import draft_exercises


def test_image_only_roi_reaches_prose_and_source_selectors(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    # Rasterize independently authored text; source has no native glyphs.
    with pymupdf.open() as image_doc:
        page = image_doc.new_page(width=400, height=200)
        page.insert_text((20, 60), "Widget rest takes 12 seconds. Stop if the lamp flashes.")
        image = page.get_pixmap().tobytes("png")
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=200)
        page.insert_image(page.rect, stream=image)
        doc.save(source)
    with pymupdf.open(source) as doc:
        assert doc[0].get_text() == ""
    engine = AuthoredOCR("Widget rest takes 12 seconds. Stop if the lamp flashes.\n")
    bundle, index = tmp_path / "bundle.json", tmp_path / "index.sqlite"
    old = recover(source, bundle, 1, 1, engine)
    assert not old["regions"] and not engine.crops
    selector = {"pdf_page": 1, "bbox": [15, 40, 350, 70]}
    result = recover(source, bundle, 1, 1, engine, image_regions=[{"id": "label", **selector}])
    assert len(engine.crops) == 1
    region = result["regions"][0]
    assert region["state"] == "pending"
    approval = {"sha256": result["source_sha256"], "pages": [1]}
    with pytest.raises(ValueError, match="no_eligible_passages"):
        ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    review(bundle, region["id"], region["revision"], "accepted")
    imported = ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    assert imported["pages"][0]["status"] == "partial"
    passage = search(index, "Widget seconds", AuthoredEmbedder())["passages"][0]
    assert passage["text"] == engine.text.strip()
    assert passage["location"]["bbox"] == selector["bbox"]
    layout = {
        "records": [
            {
                "id": "widget",
                "coverage": "complete",
                "boundaries": [selector],
                "fields": {"name": [{**selector, "quote": "Widget"}], "instructions": [selector]},
            }
        ]
    }
    catalog = draft_exercises(source, tmp_path / "catalog.json", layout, 1, 1, recovery=bundle)
    assert catalog["records"][0]["coverage"] == "complete"
    assert catalog["records"][0]["instructions"][0]["text"] == engine.text
    changed = recover(
        source,
        bundle,
        1,
        1,
        AuthoredOCR("Widget rest takes 13 seconds."),
        image_regions=[{"id": "label", **selector}],
    )
    assert changed["regions"][0]["state"] == "pending"
    assert inspect_review(bundle)["regions"][0]["history"][1]["state"] == "accepted"


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "a", "pdf_page": True, "bbox": [1, 1, 10, 10]},
        {"id": "a", "pdf_page": 1, "bbox": [True, 1, 10, 10]},
        {"id": "a", "pdf_page": 1, "bbox": [float("nan"), 1, 10, 10]},
        {"id": "a", "pdf_page": 1, "bbox": [1, 1, float("inf"), 10]},
        {"id": "a", "pdf_page": 2, "bbox": [1, 1, 10, 10]},
        {"id": "a", "pdf_page": 1, "bbox": [-1, 1, 10, 10]},
        {"id": "a", "pdf_page": 1, "bbox": [10, 1, 10, 10]},
    ],
)
def test_invalid_rois_leave_existing_bundle_and_no_crops(
    tmp_path: Path, bad: dict[str, object]
) -> None:
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(source)
    target = tmp_path / "bundle.json"
    original = recover(source, target, 1, 1, AuthoredOCR())
    before = target.read_bytes()
    engine = AuthoredOCR()
    with pytest.raises(ValueError, match="invalid_image_regions"):
        recover(
            source,
            target,
            1,
            1,
            engine,
            image_regions=[{"id": "first", "pdf_page": 1, "bbox": [20, 20, 40, 40]}, bad],
        )
    assert not engine.crops and not (tmp_path / "crops").exists()
    assert target.read_bytes() == before
    assert inspect_review(target) == original


def test_pending_sibling_roi_withholds_complete_excerpt_and_crops_are_bound(tmp_path: Path) -> None:
    from cited_retrieval import draft_excerpts, inspect_excerpts, review_excerpt

    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        doc.new_page(width=400, height=200)
        doc.save(source)
    bundle = tmp_path / "bundle.json"
    rois = [
        {"id": "one", "pdf_page": 1, "bbox": [10, 10, 200, 40]},
        {"id": "two", "pdf_page": 1, "bbox": [10, 50, 200, 80]},
    ]
    recovered = recover(source, bundle, 1, 1, AuthoredOCR(), image_regions=rois)
    first = recovered["regions"][0]
    review(bundle, first["id"], first["revision"], "accepted")
    boundary = {"pdf_page": 1, "bbox": [10, 10, 200, 80]}
    layout = {
        "records": [
            {
                "id": "p",
                "coverage": "complete",
                "boundaries": [boundary],
                "selectors": [{k: r[k] for k in ("pdf_page", "bbox")} for r in rois],
            }
        ]
    }
    catalog = tmp_path / "excerpts.json"
    draft = draft_excerpts(source, catalog, layout, 1, 1, recovery=bundle)
    record = draft["records"][0]
    assert record["coverage"] == "partial"
    review_excerpt(catalog, "p", record["revision"], "accepted")
    assert not inspect_excerpts(catalog)["records"][0]["eligible"]
    second = recovered["regions"][1]
    review(bundle, second["id"], second["revision"], "accepted")
    ready = draft_excerpts(source, catalog, layout, 1, 1, recovery=bundle)["records"][0]
    assert ready["state"] == "pending"
    # Accepting two crops does not approve the unselected strip between them.
    assert ready["coverage"] == "partial"
    selected = [{k: r[k] for k in ("pdf_page", "bbox")} for r in rois]
    precise = {
        "records": [
            {"id": "p", "coverage": "complete", "boundaries": selected, "selectors": selected}
        ]
    }
    ready = draft_excerpts(source, catalog, precise, 1, 1, recovery=bundle)["records"][0]
    assert ready["coverage"] == "complete"
    review_excerpt(catalog, "p", ready["revision"], "accepted")
    index = tmp_path / "excerpts.sqlite"
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": recovered["source_sha256"], "pages": [1]},
        recovery=bundle,
        excerpts=catalog,
    )
    paragraph = search(index, "rest seconds", AuthoredEmbedder())["passages"][0]
    assert len(paragraph["excerpt"]["segments"]) == 2
    assert paragraph["excerpt"]["segments"][1]["location"]["bbox"] == rois[1]["bbox"]
    assert (
        paragraph["text"]
        == "Hold for 12 seconds. Stop if pain develops.\nHold for 12 seconds. Stop if pain develops."
    )
    crop = tmp_path / first["crop"]
    before = bundle.read_bytes()
    crop.write_bytes(crop.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="stale_recovery"):
        inspect_review(bundle)
    assert bundle.read_bytes() == before


def test_missing_image_index_policy_marker_fails_closed(tmp_path: Path) -> None:
    import hashlib
    import json
    import sqlite3

    source, bundle, index = (
        tmp_path / "source.pdf",
        tmp_path / "bundle.json",
        tmp_path / "index.sqlite",
    )
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(source)
    result = recover(
        source,
        bundle,
        1,
        1,
        AuthoredOCR(),
        image_regions=[{"id": "one", "pdf_page": 1, "bbox": [10, 10, 200, 40]}],
    )
    region = result["regions"][0]
    review(bundle, "one", region["revision"], "accepted")
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": result["source_sha256"], "pages": [1]},
        recovery=bundle,
    )
    # Author an obsolete external persistence format, with a valid manifest seal.
    with sqlite3.connect(index) as db:
        meta = json.loads(db.execute("SELECT json FROM metadata").fetchone()[0])
        meta.pop("manifest_sha256")
        meta["report"]["settings"].pop("image_recovery_version")
        encoded = json.dumps(meta, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        meta["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        db.execute("UPDATE metadata SET json=?", (json.dumps(meta),))
    from cited_retrieval import report

    with pytest.raises(ValueError, match="stale_settings"):
        report(index)
