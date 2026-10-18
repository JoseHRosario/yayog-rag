from pathlib import Path

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR

from cited_retrieval import (
    draft_exercises,
    ingest,
    inspect_exercises,
    inspect_source,
    recover,
    review,
    review_exercise,
    search,
)


def test_image_association_is_explicit_and_hyphen_join_requires_review(tmp_path: Path):
    source, catalog, index = [tmp_path / n for n in ("source.pdf", "catalog.json", "index.sqlite")]
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 60), "Pebble lift")
        page.insert_text((50, 100), "Move care-\nfully and stop before discomfort develops.")
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        pix.clear_with(80)
        xref = page.insert_image(pymupdf.Rect(50, 180, 100, 230), pixmap=pix)
        page.insert_image(pymupdf.Rect(300, 180, 350, 230), pixmap=pix)
        doc.save(source)
    layout = {
        "records": [
            {
                "id": "pebble",
                "coverage": "complete",
                "boundaries": [{"pdf_page": 1, "bbox": [40, 30, 400, 250]}],
                "images": [{"pdf_page": 1, "xref": xref, "bbox": [50, 180, 100, 230]}],
                "fields": {
                    "name": [{"pdf_page": 1, "bbox": [45, 40, 400, 65]}],
                    "instructions": [{"pdf_page": 1, "bbox": [45, 80, 400, 125]}],
                },
            }
        ]
    }
    drafted = draft_exercises(source, catalog, layout, 1, 1)
    record = drafted["records"][0]
    assert (
        record["instructions"][0]["text"]
        == "Move care-\nfully and stop before discomfort develops."
    )
    review_exercise(catalog, "pebble", record["revision"], "accepted")
    layout["records"][0]["fields"]["instructions"][0]["joins"] = ["care-\nfully"]
    joined = draft_exercises(source, catalog, layout, 1, 1)["records"][0]
    assert joined["state"] == "pending"
    assert (
        joined["instructions"][0]["text"] == "Move carefully and stop before discomfort develops."
    )
    assert joined["instructions"][0]["original_text"].startswith("Move care-\nfully")
    review_exercise(catalog, "pebble", joined["revision"], "accepted")
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": inspect_source(source, 1, 1)["sha256"], "pages": [1]},
        exercises=catalog,
    )
    images = search(index, "Pebble lift", AuthoredEmbedder())["passages"][0]["exercise"]["images"]
    assert len(images) == 1
    assert images[0] == {
        "source_id": drafted["binding"]["source_sha256"],
        "pdf_page": 1,
        "xref": xref,
        "bbox": [50, 180, 100, 230],
        "visual_understanding": False,
    }
    layout["records"][0]["images"][0]["xref"] = 99999
    with pytest.raises(ValueError, match="invalid_exercise_image"):
        draft_exercises(source, catalog, layout, 1, 1)


def test_unresolved_ocr_inside_boundary_withholds_whole_record_even_if_accepted(tmp_path: Path):
    source, bundle, catalog = [tmp_path / n for n in ("source.pdf", "bundle.json", "catalog.json")]
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 60), "Pebble lift")
        page.insert_text((50, 100), "Move slowly with a steady grip.")
        font = page.insert_font(fontname="damaged", fontbuffer=pymupdf.Font("cjk").buffer)
        page.insert_text((50, 140), "Stop if pain develops.", fontname="damaged")
        doc.xref_set_key(font, "ToUnicode", "null")
        doc.save(source)
    recovered = recover(source, bundle, 1, 1, AuthoredOCR("Stop if pain develops."))
    layout = {
        "records": [
            {
                "id": "pebble",
                "coverage": "complete",
                "boundaries": [{"pdf_page": 1, "bbox": [40, 30, 500, 160]}],
                "fields": {
                    "name": [{"pdf_page": 1, "bbox": [45, 40, 500, 65]}],
                    "instructions": [{"pdf_page": 1, "bbox": [45, 80, 500, 105]}],
                    "warnings": [{"pdf_page": 1, "bbox": [45, 120, 500, 145]}],
                },
            }
        ]
    }
    drafted = draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)
    record = drafted["records"][0]
    review_exercise(catalog, "pebble", record["revision"], "accepted")
    assert inspect_exercises(catalog)["records"][0]["eligible"] is False
    with pytest.raises(ValueError, match="no_eligible_passages"):
        ingest(
            source,
            tmp_path / "index.sqlite",
            1,
            1,
            AuthoredEmbedder(),
            approval={"sha256": drafted["binding"]["source_sha256"], "pages": [1]},
            recovery=bundle,
            exercises=catalog,
        )
    region = recovered["regions"][0]
    review(bundle, region["id"], region["revision"], "accepted")
    with pytest.raises(ValueError, match="stale_exercises"):
        inspect_exercises(catalog)
    record = draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)["records"][0]
    assert record["state"] == "pending"
    assert record["coverage"] == "complete"
    assert record["warnings"][0]["text"] == "Stop if pain develops."
