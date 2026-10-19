import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

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
    report,
    review,
    review_exercise,
    search,
)


def navigation_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    source, catalog = tmp_path / "navigation.pdf", tmp_path / "catalog.json"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 60), "Cloud press")
        page.insert_text((50, 100), "Wall press")
        page.insert_text((50, 140), "Movement directory")
        page = doc.new_page()
        page.insert_text((50, 100), "Lift slowly and keep the wrist steady.")
        page = doc.new_page()
        page.insert_text((50, 100), "Use the wall to reduce the load.")
        page.insert_text((50, 140), "Stop if discomfort develops during the movement.")
        doc.set_page_labels([{"startpage": 0, "prefix": "leaf-", "style": "D", "firstpagenum": 1}])
        doc.save(source)

    def selector(page: int, y: int) -> dict[str, Any]:
        return {"pdf_page": page, "bbox": [45, y - 20, 500, y + 5]}

    layout = {
        "records": [
            {
                "id": "cloud",
                "coverage": "complete",
                "aliases": ["sky push"],
                "boundaries": [
                    {"pdf_page": page, "bbox": [40, 30, 510, 160]} for page in (1, 2, 3)
                ],
                "fields": {
                    "name": [selector(1, 60)],
                    "chapter": [selector(1, 140)],
                    "instructions": [selector(2, 100)],
                    "warnings": [selector(3, 140)],
                },
                "variations": [
                    {
                        "id": "wall",
                        "relation": "easier",
                        "aliases": ["wall assist"],
                        "fields": {"name": [selector(1, 100)], "instructions": [selector(3, 100)]},
                    }
                ],
            }
        ]
    }
    draft_exercises(source, catalog, layout, 1, 3)
    record = inspect_exercises(catalog)["records"][0]
    review_exercise(catalog, "cloud", record["revision"], "accepted")
    return source, catalog, layout


@pytest.mark.parametrize(
    "query,node_id,page",
    [
        ("Cloud press", "cloud", 2),
        ("sky push", "cloud", 2),
        ("Wall press", "wall", 3),
        ("wall assist", "wall", 3),
    ],
)
def test_navigation_names_anchor_to_own_instructions(
    tmp_path: Path, query: str, node_id: str, page: int
) -> None:
    source, catalog, _ = navigation_fixture(tmp_path)
    before = catalog.read_bytes()
    accepted = inspect_exercises(catalog)
    index = tmp_path / "index.sqlite"
    imported = ingest(
        source,
        index,
        1,
        3,
        AuthoredEmbedder(),
        approval={"sha256": accepted["binding"]["source_sha256"], "pages": [1, 2, 3]},
        exercises=catalog,
    )
    assert report(index) == imported
    passage = search(index, query, AuthoredEmbedder())["passages"][0]
    record = accepted["records"][0]
    node = record if node_id == "cloud" else record["variations"][0]
    assert passage["provenance"]["matched_node_id"] == node_id
    assert passage["location"] == node["instructions"][0]["location"]
    assert passage["location"]["pdf_page"] == page
    assert passage["location"]["printed_label"] == f"leaf-{page}"
    assert passage["location"]["chapter"] == "Movement directory"
    assert passage["exercise"] == record
    assert node["citations"]["name"][0]["location"]["pdf_page"] == 1
    for text in ("Cloud press", "Wall press", "Lift slowly", "Use the wall", "Stop if discomfort"):
        assert text in passage["text"]
    assert catalog.read_bytes() == before


def seed_legacy_anchor_index(index: Path, marker: float | None) -> dict[str, str]:
    """Author a sealed old-format external input, not a business-function double."""

    def encode(value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    def digest(value: Any) -> str:
        return hashlib.sha256(encode(value).encode()).hexdigest()

    ids = {}
    with sqlite3.connect(index) as db:
        meta = json.loads(db.execute("SELECT json FROM metadata").fetchone()[0])
        binding = meta["report"]["exercises"]
        binding.pop("instruction_anchor_version", None)
        if marker is not None:
            binding["instruction_anchor_version"] = marker
        records = []
        for payload, vector in db.execute("SELECT json,vector FROM passages ORDER BY rowid"):
            passage = json.loads(payload)
            record = passage["exercise"]
            node = next(
                n
                for n in [record, *record["variations"]]
                if n["id"] == passage["provenance"]["matched_node_id"]
            )
            ids[node["id"]] = passage["passage_id"]
            passage["location"] = node["citations"]["name"][0]["location"]
            records.append([encode(passage), vector])
        db.execute("DELETE FROM passages")
        db.executemany("INSERT INTO passages VALUES (?,?)", records)
        meta["corpus_sha256"] = digest(records)
        meta["report"]["corpus_sha256"] = digest(records)
        meta.pop("manifest_sha256")
        meta["manifest_sha256"] = digest(meta)
        db.execute("UPDATE metadata SET json=?", (encode(meta),))
    return ids


@pytest.mark.parametrize("marker", [None, 0, 2, True, 1.0])
def test_legacy_exercise_index_requires_rebuild_without_new_catalog_review(
    tmp_path: Path, marker: float | None
) -> None:
    source, catalog, layout = navigation_fixture(tmp_path)
    accepted = inspect_exercises(catalog)
    before = catalog.read_bytes()
    index = tmp_path / "legacy.sqlite"
    approval = {"sha256": accepted["binding"]["source_sha256"], "pages": [1, 2, 3]}
    ingest(source, index, 1, 3, AuthoredEmbedder(), approval=approval, exercises=catalog)
    old_ids = seed_legacy_anchor_index(index, marker)
    for operation in (
        lambda: report(index),
        lambda: search(index, "Wall press", AuthoredEmbedder()),
    ):
        with pytest.raises(ValueError, match="^stale_settings$"):
            operation()
    assert inspect_exercises(catalog) == accepted
    assert catalog.read_bytes() == before
    rebuilt = ingest(source, index, 1, 3, AuthoredEmbedder(), approval=approval, exercises=catalog)
    assert rebuilt["exercises"]["instruction_anchor_version"] == 1
    assert report(index) == rebuilt
    for query, node_id, page in (("Cloud press", "cloud", 2), ("Wall press", "wall", 3)):
        passage = search(index, query, AuthoredEmbedder())["passages"][0]
        assert passage["passage_id"] == old_ids[node_id]
        assert passage["location"]["pdf_page"] == page
        assert passage["exercise"] == accepted["records"][0]
    assert draft_exercises(source, catalog, layout, 1, 3) == accepted
    assert (
        ingest(source, index, 1, 3, AuthoredEmbedder(), approval=approval, exercises=catalog)
        == rebuilt
    )


def test_anchor_policy_is_not_required_for_prose_or_programme_indexes(tmp_path: Path) -> None:
    from test_programmes import accept, build, schedule

    from cited_retrieval import draft_programmes

    prose = tmp_path / "prose.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text(
            (50, 60),
            "Cloud movement practice uses a steady wrist and a slow lift.\n"
            "Keep your balance and move gently throughout each repetition.\n"
            "Stop before discomfort develops and rest between movements.",
        )
        doc.save(prose)
    index = tmp_path / "prose.sqlite"
    imported = ingest(
        prose,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": inspect_source(prose, 1, 1)["sha256"], "pages": [1]},
    )
    assert "exercises" not in imported
    assert report(index) == imported
    assert search(index, "steady wrist", AuthoredEmbedder())["status"] == "evidence"

    exercise_dir = tmp_path / "exercises"
    exercise_dir.mkdir()
    _, exercises, _ = navigation_fixture(exercise_dir)
    source, catalog, layout, _ = schedule(tmp_path)
    draft_programmes(source, catalog, layout, 1, 2, exercise_catalogs=[exercises])
    accept(catalog)
    programme_index = build(source, catalog, tmp_path)
    programme_report = report(programme_index)
    assert "exercises" not in programme_report
    assert "instruction_anchor_version" not in programme_report["programmes"]
    assert "matching_window_version" not in programme_report["programmes"]
    result = search(
        programme_index,
        "Cloud programme",
        AuthoredEmbedder(),
        programme="Cloud programme",
        week=1,
        day=1,
    )
    assert result["status"] == "evidence"
    assert (
        result["passages"][0]["session"]["entries"][0]["exercise_reference"]["status"] == "resolved"
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
