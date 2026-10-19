"""Public exercise imports keep own instructions separate from shared guidance."""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from context_balance_embedder import ContextBalanceEmbedder

from cited_retrieval import (
    draft_exercises,
    ingest,
    inspect_exercises,
    report,
    review_exercise,
    search,
)


def context_fixture(
    tmp_path: Path, *, repetitions: int = 9
) -> tuple[Path, Path, dict[str, Any], str]:
    source, catalog = tmp_path / "context.pdf", tmp_path / "catalog.json"
    guidance = "\n".join(
        ["Balance practice develops control. Balance is useful throughout this category."]
        * repetitions
    )
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 60), "Cloud press")
        page.insert_text((50, 100), "Wall press")
        page.insert_text((50, 180), "Stone curl")
        page = doc.new_page()
        page.insert_text((50, 100), "Lift slowly and keep the wrist steady.")
        page.insert_text((50, 180), "Balance with your wrist.")
        page = doc.new_page()
        page.insert_text((50, 100), "Use the wall to reduce the load.")
        page.insert_text((50, 140), "Stop if discomfort develops during the movement.")
        page = doc.new_page()
        page.insert_text((50, 100), guidance)
        doc.save(source)

    def selector(page: int, y: int) -> dict[str, Any]:
        return {"pdf_page": page, "bbox": [45, y - 20, 550, y + 5]}

    references = [{"pdf_page": 4, "bbox": [45, 80, 550, 500]}]
    boundaries = [{"pdf_page": page, "bbox": [40, 30, 560, 520]} for page in (1, 2, 3, 4)]
    layout = {
        "records": [
            {
                "id": "cloud",
                "coverage": "complete",
                "boundaries": boundaries,
                "fields": {
                    "name": [selector(1, 60)],
                    "instructions": [selector(2, 100)],
                    "warnings": [selector(3, 140)],
                    "references": references,
                },
                "variations": [
                    {
                        "id": "wall",
                        "relation": "easier",
                        "fields": {
                            "name": [selector(1, 100)],
                            "instructions": [selector(3, 100)],
                            "references": references,
                        },
                    }
                ],
            },
            {
                "id": "stone",
                "coverage": "complete",
                "boundaries": boundaries,
                "fields": {"name": [selector(1, 180)], "instructions": [selector(2, 180)]},
            },
        ]
    }
    drafted = draft_exercises(source, catalog, layout, 1, 4)
    for record in drafted["records"]:
        review_exercise(catalog, record["id"], record["revision"], "accepted")
    return source, catalog, layout, guidance


def build_context(
    source: Path, catalog: Path, index: Path, model: ContextBalanceEmbedder
) -> dict[str, Any]:
    accepted = inspect_exercises(catalog)
    return ingest(
        source,
        index,
        1,
        4,
        model,
        approval={"sha256": accepted["binding"]["source_sha256"], "pages": [1, 2, 3, 4]},
        exercises=catalog,
    )


def test_shared_guidance_cannot_dilute_own_instruction_matching(tmp_path: Path) -> None:
    source, catalog, _, guidance = context_fixture(tmp_path)
    accepted = inspect_exercises(catalog)
    before = catalog.read_bytes()
    model = ContextBalanceEmbedder()
    index = tmp_path / "index.sqlite"
    imported = build_context(source, catalog, index, model)
    assert report(index) == imported
    passage = search(index, "Which movement uses a steady wrist?", model, limit=1)["passages"][0]
    assert passage["exercise"]["id"] == "cloud"
    assert passage["semantic_rank"] == 1
    assert passage["semantic_score"] == pytest.approx(1.0)
    assert passage["provenance"]["matched_node_id"] == "cloud"
    assert passage["location"] == accepted["records"][0]["instructions"][0]["location"]
    assert passage["location"]["pdf_page"] == 2
    assert "Balance practice" not in passage["retrieval_text"]
    assert "Stop if discomfort" in passage["retrieval_text"]
    assert passage["exercise"] == accepted["records"][0]
    # Literal source-field order is the existing complete-evidence contract.
    assert passage["text"] == (
        "Lift slowly and keep the wrist steady.\n\nCloud press\n\n"
        f"{guidance}\n\nStop if discomfort develops during the movement.\n\n"
        f"Use the wall to reduce the load.\n\nWall press\n\n{guidance}"
    )
    assert passage["exercise"]["difficulty"] is None
    assert passage["exercise"]["variations"][0]["difficulty"] is None
    assert catalog.read_bytes() == before


def test_shared_guidance_is_independently_searchable_with_complete_evidence(tmp_path: Path) -> None:
    source, catalog, _, guidance = context_fixture(tmp_path)
    model = ContextBalanceEmbedder()
    index = tmp_path / "index.sqlite"
    build_context(source, catalog, index, model)
    passage = search(index, "stability advice", model, limit=1)["passages"][0]
    assert passage["exercise"]["id"] == "cloud"
    assert passage["semantic_score"] == pytest.approx(1.0)
    assert passage["retrieval_text"] == guidance
    assert passage["location"]["pdf_page"] == 2
    assert "Lift slowly and keep the wrist steady." in passage["text"]
    assert "Use the wall to reduce the load." in passage["text"]
    assert "Stop if discomfort" in passage["text"]
    variation = search(index, "Wall press", model)["passages"][0]
    assert variation["provenance"]["matched_node_id"] == "wall"
    assert variation["location"]["pdf_page"] == 3
    assert variation["exercise"] == passage["exercise"]


def test_long_references_are_bounded_without_changing_own_matching_text(tmp_path: Path) -> None:
    source, catalog, layout, guidance = context_fixture(tmp_path, repetitions=24)
    model = ContextBalanceEmbedder()
    index = tmp_path / "long.sqlite"
    build_context(source, catalog, index, model)
    own = search(index, "Which movement uses a steady wrist?", model, limit=1)["passages"][0]
    assert own["exercise"]["id"] == "cloud"
    assert guidance in own["text"]
    assert all(len(text) <= 1200 for text in model.inputs)
    # Observe the actual inference boundary, not passage construction internals.
    reference_windows = [text for text in model.inputs if "Balance practice" in text]
    assert len(reference_windows) == 4  # Two chunks for parent and two for child.
    assert "".join(reference_windows[:2]) == guidance
    assert not any("wrist" in text.casefold() for text in reference_windows)
    for node in [layout["records"][0], *layout["records"][0]["variations"]]:
        node["fields"]["references"] = [{"pdf_page": 4, "bbox": [45, 80, 550, 105]}]
    drafted = draft_exercises(source, catalog, layout, 1, 4)
    record = drafted["records"][0]
    review_exercise(catalog, "cloud", record["revision"], "accepted")
    build_context(source, catalog, index, model)
    shorter = search(index, "Which movement uses a steady wrist?", model, limit=1)["passages"][0]
    assert shorter["retrieval_text"] == own["retrieval_text"]
    assert shorter["semantic_score"] == own["semantic_score"]


@pytest.mark.parametrize("partial", [False, True])
def test_reference_windows_do_not_bypass_withheld_or_ambiguous_identity(
    tmp_path: Path, partial: bool
) -> None:
    source, catalog, layout, _ = context_fixture(tmp_path)
    parent = layout["records"][0]
    if partial:
        parent["coverage"] = "partial"
    else:
        parent["variations"][0]["gaps"] = ["unverified warning"]
    layout["records"][1]["aliases"] = ["Cloud press"]
    drafted = draft_exercises(source, catalog, layout, 1, 4)
    for record in drafted["records"]:
        review_exercise(catalog, record["id"], record["revision"], "accepted")
    model = ContextBalanceEmbedder()
    index = tmp_path / "withheld.sqlite"
    build_context(source, catalog, index, model)
    assert not any("Balance practice" in text for text in model.inputs)
    withheld = search(index, "Wall press", model)
    assert withheld["status"] == "no_evidence"
    assert withheld["passages"] == []
    ambiguous = search(index, "Cloud press", model)
    assert ambiguous["status"] == "ambiguous"
    assert ambiguous["passages"] == []
    assert {candidate["id"] for candidate in ambiguous["candidates"]} == {"cloud", "stone"}
    assert "Stop if discomfort" not in str(ambiguous)


def seed_matching_policy(index: Path, marker: object) -> None:
    """Author obsolete external metadata with a valid integrity seal."""

    def encode(value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    with sqlite3.connect(index) as db:
        meta = json.loads(db.execute("SELECT json FROM metadata").fetchone()[0])
        binding = meta["report"]["exercises"]
        binding.pop("matching_window_version", None)
        if marker is not None:
            binding["matching_window_version"] = marker
        meta.pop("manifest_sha256")
        meta["manifest_sha256"] = hashlib.sha256(encode(meta).encode()).hexdigest()
        db.execute("UPDATE metadata SET json=?", (encode(meta),))


@pytest.mark.parametrize("marker", [None, 0, 2, True, 1.0, "1"])
def test_obsolete_matching_windows_require_exercise_only_rebuild(
    tmp_path: Path, marker: object
) -> None:
    source, catalog, layout, _ = context_fixture(tmp_path)
    accepted = inspect_exercises(catalog)
    before = catalog.read_bytes()
    model = ContextBalanceEmbedder()
    index = tmp_path / "legacy.sqlite"
    build_context(source, catalog, index, model)
    seed_matching_policy(index, marker)
    for operation in (
        lambda: report(index),
        lambda: search(index, "Which movement uses a steady wrist?", model),
    ):
        with pytest.raises(ValueError, match="^stale_settings$"):
            operation()
    assert inspect_exercises(catalog) == accepted
    assert catalog.read_bytes() == before
    rebuilt = build_context(source, catalog, index, model)
    assert rebuilt["exercises"]["matching_window_version"] == 1
    assert rebuilt["exercises"]["instruction_anchor_version"] == 1
    assert report(index) == rebuilt
    passage = search(index, "Which movement uses a steady wrist?", model, limit=1)["passages"][0]
    assert passage["exercise"] == accepted["records"][0]
    assert passage["location"]["pdf_page"] == 2
    assert draft_exercises(source, catalog, layout, 1, 4) == accepted
    assert build_context(source, catalog, index, model) == rebuilt
    repeated = search(index, "Which movement uses a steady wrist?", model, limit=1)["passages"][0]
    assert repeated == passage
