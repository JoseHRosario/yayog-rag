from pathlib import Path
from typing import Any

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder

from cited_retrieval import (
    draft_exercises,
    ingest,
    inspect_exercises,
    inspect_source,
    review_exercise,
    search,
)


def fixture(tmp_path: Path):
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        p = doc.new_page()
        p.insert_text((50, 60), "Cloud press")
        p.insert_text((50, 100), "Lift slowly and keep the wrist steady.")
        p.insert_text((50, 160), "Stone curl")
        p.insert_text((50, 200), "Curl gently without moving the elbow.")
        p = doc.new_page()
        p.insert_text((50, 60), "Stop if discomfort develops during the movement.")
        p.insert_text((50, 100), "Wall press")
        p.insert_text((50, 140), "Use the wall to reduce the load.")
        doc.save(source)

    def sel(page, y):
        return {"pdf_page": page, "bbox": [45, y - 20, 500, y + 5]}

    layout = {
        "records": [
            {
                "id": "cloud",
                "coverage": "complete",
                "aliases": ["sky push"],
                "boundaries": [
                    {"pdf_page": 1, "bbox": [40, 30, 510, 120]},
                    {"pdf_page": 2, "bbox": [40, 30, 510, 155]},
                ],
                "fields": {
                    "name": [sel(1, 60)],
                    "instructions": [sel(1, 100)],
                    "warnings": [sel(2, 60)],
                },
                "variations": [
                    {
                        "id": "wall",
                        "relation": "easier",
                        "fields": {"name": [sel(2, 100)], "instructions": [sel(2, 140)]},
                    }
                ],
            },
            {
                "id": "stone",
                "coverage": "complete",
                "boundaries": [{"pdf_page": 1, "bbox": [40, 130, 510, 220]}],
                "fields": {"name": [sel(1, 160)], "instructions": [sel(1, 200)]},
            },
        ]
    }
    catalog = tmp_path / "catalog.json"
    draft_exercises(source, catalog, layout, 1, 2)
    return source, catalog, layout


def clipped_image_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], int]:
    source, catalog, layout = fixture(tmp_path)
    with pymupdf.open(source) as doc:
        page = doc[0]
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        pix.clear_with(80)
        xref = page.insert_image(pymupdf.Rect(-20, 240, 80, 340), pixmap=pix)
        page.insert_image(pymupdf.Rect(-40, 400, 60, 500), xref=xref)
        page.insert_image(pymupdf.Rect(300, 240, 400, 340), xref=xref)
        pix.clear_with(160)
        other_xref = page.insert_image(pymupdf.Rect(450, 240, 500, 290), pixmap=pix)
        doc.saveIncr()
    layout["records"][0]["boundaries"][0]["bbox"] = [0, 30, 510, 500]
    image = {
        "pdf_page": 1,
        "xref": xref,
        "bbox": [0, 240, 80, 340],
        "placement_bbox": [-20, 240, 80, 340],
    }
    layout["records"][0]["images"] = [image]
    return source, catalog, layout, image, other_xref


@pytest.mark.parametrize("child", [False, True])
def test_exact_clipped_image_placement_survives_review_and_import(
    tmp_path: Path, child: bool
) -> None:
    source, catalog, layout, image, _ = clipped_image_fixture(tmp_path)
    if child:
        layout["records"][0]["variations"][0]["images"] = layout["records"][0].pop("images")
    drafted = draft_exercises(source, catalog, layout, 1, 2)
    expected = {
        "source_id": drafted["binding"]["source_sha256"],
        **image,
        "visual_understanding": False,
    }
    record = inspect_exercises(catalog)["records"][0]
    node = record["variations"][0] if child else record
    assert node["images"] == [expected]
    assert record["eligible"] is False
    accept(catalog)
    assert inspect_exercises(catalog)["records"][0]["eligible"] is True
    result = search(build(source, catalog, tmp_path), "Cloud press", AuthoredEmbedder())
    record = result["passages"][0]["exercise"]
    node = record["variations"][0] if child else record
    assert node["images"] == [expected]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("arbitrary_crop", "invalid_exercise_image"),
        ("near_visible_box", "invalid_exercise_image"),
        ("wrong_placement", "invalid_exercise_image"),
        ("near_placement", "invalid_exercise_image"),
        ("wrong_existing_xref", "invalid_exercise_image"),
        ("missing_xref", "invalid_exercise_image"),
        ("swapped_repeated_occurrence", "invalid_exercise_image"),
        ("swapped_visible_occurrence", "invalid_exercise_image"),
        ("outside_parent", "invalid_exercise_layout"),
        ("outside_child", "invalid_exercise_layout"),
        ("empty_visible", "invalid_exercise_layout"),
        ("offpage_visible", "invalid_exercise_layout"),
        ("empty_placement", "invalid_exercise_image"),
        ("infinite_placement", "invalid_exercise_image"),
        ("unclipped_opt_in", "invalid_exercise_image"),
        ("field_offpage", "invalid_exercise_layout"),
        ("boundary_offpage", "invalid_exercise_layout"),
    ],
)
def test_clipped_images_reject_nonexact_or_unowned_geometry(
    tmp_path: Path, change: str, error: str
) -> None:
    source, catalog, layout, image, other_xref = clipped_image_fixture(tmp_path)
    draft_exercises(source, catalog, layout, 1, 2)
    before = catalog.read_bytes()
    parent = layout["records"][0]
    if change == "arbitrary_crop":
        image["bbox"] = [10, 250, 70, 330]
    elif change == "near_visible_box":
        image["bbox"][2] = 79.99999
    elif change == "wrong_placement":
        image["placement_bbox"][0] = -30
    elif change == "near_placement":
        image["placement_bbox"][0] = -20.00001
    elif change == "wrong_existing_xref":
        image["xref"] = other_xref
    elif change == "missing_xref":
        image["xref"] = 99999
    elif change == "swapped_repeated_occurrence":
        image["placement_bbox"] = [-40, 400, 60, 500]
    elif change == "swapped_visible_occurrence":
        image["bbox"] = [0, 400, 60, 500]
    elif change == "outside_parent":
        parent["boundaries"][0]["bbox"][0] = 1
    elif change == "outside_child":
        child = parent["variations"][0]
        child["images"] = parent.pop("images")
        child["boundaries"] = [
            {"pdf_page": 1, "bbox": [1, 30, 510, 500]},
            parent["boundaries"][1],
        ]
    elif change == "empty_visible":
        image["bbox"] = [0, 240, 0, 340]
    elif change == "offpage_visible":
        image["bbox"] = [-1, 240, 80, 340]
    elif change == "empty_placement":
        image["placement_bbox"] = [0, 240, 0, 340]
    elif change == "infinite_placement":
        image["placement_bbox"] = list(pymupdf.INFINITE_RECT())
    elif change == "unclipped_opt_in":
        image["bbox"] = image["placement_bbox"] = [300, 240, 400, 340]
    elif change == "field_offpage":
        parent["fields"]["name"][0]["bbox"][0] = -1
    elif change == "boundary_offpage":
        parent["boundaries"][0]["bbox"][0] = -1
    with pytest.raises(ValueError, match=error):
        draft_exercises(source, catalog, layout, 1, 2)
    assert catalog.read_bytes() == before
    assert inspect_exercises(catalog)["records"][0]["images"][0]["bbox"] == [0, 240, 80, 340]


def test_old_image_format_stays_exact_and_does_not_acquire_placement_provenance(
    tmp_path: Path,
) -> None:
    source, catalog, layout, image, _ = clipped_image_fixture(tmp_path)
    image.pop("placement_bbox")
    # An intersection is not an original occurrence in the old format.
    with pytest.raises(ValueError, match="invalid_exercise_image"):
        draft_exercises(source, catalog, layout, 1, 2)
    image["bbox"] = [-20, 240, 80, 340]
    with pytest.raises(ValueError, match="invalid_exercise_layout"):
        draft_exercises(source, catalog, layout, 1, 2)
    image["bbox"] = [300, 240, 400, 340]
    drafted = draft_exercises(source, catalog, layout, 1, 2)
    expected = {
        "source_id": drafted["binding"]["source_sha256"],
        "pdf_page": 1,
        "xref": image["xref"],
        "bbox": [300, 240, 400, 340],
        "visual_understanding": False,
    }
    assert drafted["binding"]["settings"]["version"] == 1
    assert drafted["records"][0]["images"] == [expected]
    accept(catalog)
    accepted = inspect_exercises(catalog)["records"][0]
    repeated = draft_exercises(source, catalog, layout, 1, 2)["records"][0]
    assert repeated == accepted
    result = search(build(source, catalog, tmp_path), "Cloud press", AuthoredEmbedder())
    assert result["passages"][0]["exercise"]["images"] == [expected]


def test_changing_exact_clipped_occurrence_requires_new_review(tmp_path: Path) -> None:
    source, catalog, layout, image, _ = clipped_image_fixture(tmp_path)
    draft_exercises(source, catalog, layout, 1, 2)
    accept(catalog)
    old = inspect_exercises(catalog)["records"][0]
    image["placement_bbox"] = [-40, 400, 60, 500]
    image["bbox"] = [0, 400, 60, 500]
    record = draft_exercises(source, catalog, layout, 1, 2)["records"][0]
    assert record["revision"] != old["revision"]
    assert record["state"] == "pending"
    assert record["eligible"] is False
    assert record["history"] == old["history"]
    with pytest.raises(ValueError, match="review_revision_mismatch"):
        review_exercise(catalog, "cloud", old["revision"], "accepted")


def test_unnamed_variation_is_retrievable_without_inventing_a_name(tmp_path):
    source, catalog, layout = fixture(tmp_path)
    layout["records"][0]["variations"][0]["fields"].pop("name")
    draft_exercises(source, catalog, layout, 1, 2)
    accept(catalog)
    result = search(build(source, catalog, tmp_path), "Cloud press", AuthoredEmbedder())
    assert result["status"] == "evidence"
    assert result["passages"][0]["exercise"]["id"] == "cloud"
    variation = result["passages"][0]["exercise"]["variations"][0]
    assert variation["name"] is None
    assert variation["instructions"][0]["text"] == "Use the wall to reduce the load."


def test_specific_name_wins_over_a_shorter_alias_inside_it(tmp_path):
    source, catalog, layout = fixture(tmp_path)
    layout["records"][1]["aliases"] = ["press"]
    draft_exercises(source, catalog, layout, 1, 2)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    for query in ("How do I do the Cloud press?", "How do I do the Wall press?"):
        result = search(index, query, AuthoredEmbedder())
        assert result["status"] == "evidence"
        assert result["passages"][0]["exercise"]["id"] == "cloud"
    # Separate names in the question must not be collapsed to the longest one.
    compared = search(index, "Compare Cloud press with Stone curl", AuthoredEmbedder())
    assert compared["status"] == "ambiguous"
    assert {r["id"] for r in compared["candidates"]} == {"cloud", "stone"}


def test_known_withheld_name_does_not_fall_back_to_a_different_movement(tmp_path):
    source, catalog, _ = fixture(tmp_path)
    record = next(r for r in inspect_exercises(catalog)["records"] if r["id"] == "stone")
    review_exercise(catalog, record["id"], record["revision"], "accepted")
    index = build(source, catalog, tmp_path)
    result = search(index, "How do I do the Cloud press?", AuthoredEmbedder())
    assert result["status"] == "no_evidence"
    assert result["passages"] == []
    assert "Lift slowly" not in str(result)
    # Unaccepted operator aliases do not override a separately accepted identity.
    assert (
        search(index, "Stone curl", AuthoredEmbedder())["passages"][0]["exercise"]["id"] == "stone"
    )


def test_shared_alias_with_a_withheld_parent_is_ambiguous_without_its_instructions(tmp_path):
    source, catalog, layout = fixture(tmp_path)
    layout["records"][1]["aliases"] = ["sky push"]
    draft_exercises(source, catalog, layout, 1, 2)
    record = inspect_exercises(catalog)["records"][0]
    review_exercise(catalog, record["id"], record["revision"], "accepted")
    result = search(build(source, catalog, tmp_path), "sky push", AuthoredEmbedder())
    assert result["status"] == "ambiguous"
    assert result["passages"] == []
    assert {c["id"] for c in result["candidates"]} == {"cloud", "stone"}
    assert "Curl gently" not in str(result)


def test_variation_paraphrase_survives_finite_embedding_context(tmp_path):
    from windowed_embedder import WindowedEmbedder

    source, catalog, _ = fixture(tmp_path)
    accept(catalog)
    index = tmp_path / "windowed.sqlite"
    approval = {"sha256": inspect_source(source, 1, 2)["sha256"], "pages": [1, 2]}
    model = WindowedEmbedder()
    ingest(source, index, 1, 2, model, approval=approval, exercises=catalog)
    result = search(index, "lighter resistance", model)
    assert result["status"] == "evidence"
    assert [p["exercise"]["id"] for p in result["passages"]] == ["cloud"]
    assert result["passages"][0]["exercise"]["warnings"][0]["text"] == (
        "Stop if discomfort develops during the movement."
    )
    assert result["passages"][0]["location"]["pdf_page"] == 2


def test_exact_variation_returns_its_matching_window_and_source_anchor(tmp_path):
    source, catalog, _ = fixture(tmp_path)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    for model in (AuthoredEmbedder(), AuthoredEmbedder(fixed_vector=[0.0, -1.0])):
        result = search(index, "Wall press", model)
        passage = result["passages"][0]
        assert passage["exercise"]["id"] == "cloud"
        assert passage["provenance"]["matched_node_id"] == "wall"
        assert passage["location"]["pdf_page"] == 2
        assert "Use the wall to reduce the load." in passage["retrieval_text"]
        assert "Lift slowly and keep the wrist steady." in passage["text"]
        assert "Stop if discomfort develops" in passage["text"]


def test_incomplete_parent_never_leaks_through_search_coverage(tmp_path):
    source, catalog, layout = fixture(tmp_path)
    layout["records"][0]["coverage"] = "partial"
    layout["records"][0]["gaps"] = ["Lift slowly: operator notes must stay on the review surface"]
    draft_exercises(source, catalog, layout, 1, 2)
    accept(catalog)
    result = search(build(source, catalog, tmp_path), "Stone curl", AuthoredEmbedder())
    assert "Lift slowly" not in str(result)
    assert "Stop if discomfort" not in str(result)
    assert result["exercise_coverage"][0]["eligible"] is False


def test_alias_ambiguity_is_explicit_and_revision_change_resets_review(tmp_path):
    source, catalog, layout = fixture(tmp_path)
    layout["records"][1]["aliases"] = ["sky push"]
    draft_exercises(source, catalog, layout, 1, 2)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    result = search(index, "sky push", AuthoredEmbedder())
    assert result["status"] == "ambiguous"
    assert result["passages"] == []
    assert {r["id"] for r in result["candidates"]} == {"cloud", "stone"}
    revision = inspect_exercises(catalog)["records"][0]["revision"]
    layout["records"][0]["aliases"].append("new label")
    draft_exercises(source, catalog, layout, 1, 2)
    assert inspect_exercises(catalog)["records"][0]["state"] == "pending"
    with pytest.raises(ValueError, match="review_revision_mismatch"):
        review_exercise(catalog, "cloud", revision, "accepted")
    from cited_retrieval import report

    with pytest.raises(ValueError, match="stale_exercises"):
        report(index)
    with pytest.raises(ValueError, match="stale_exercises"):
        search(index, "sky push", AuthoredEmbedder())


def test_selectors_cannot_cross_parent_boundary_and_failed_draft_preserves_catalog(tmp_path):
    source, catalog, layout = fixture(tmp_path)
    before = catalog.read_bytes()
    layout["records"][0]["fields"]["instructions"][0]["bbox"] = [40, 80, 510, 205]
    with pytest.raises(ValueError, match="invalid_exercise_layout"):
        draft_exercises(source, catalog, layout, 1, 2)
    assert catalog.read_bytes() == before


def test_explicit_chapter_selector_is_retained_at_child_source_locations(tmp_path):
    source, catalog, layout = fixture(tmp_path)
    layout["records"][0]["fields"]["chapter"] = layout["records"][0]["fields"]["name"]
    draft_exercises(source, catalog, layout, 1, 2)
    accept(catalog)
    record = search(build(source, catalog, tmp_path), "Cloud press", AuthoredEmbedder())[
        "passages"
    ][0]["exercise"]
    assert record["warnings"][0]["location"]["chapter"] == "Cloud press"
    assert record["variations"][0]["instructions"][0]["location"]["chapter"] == "Cloud press"


def test_review_change_during_embedding_fails_query_and_atomic_rebuild(tmp_path):
    from catalog_changing_embedder import CatalogChangingEmbedder

    source, catalog, _ = fixture(tmp_path)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    before = index.read_bytes()
    record = inspect_exercises(catalog)["records"][0]

    def change():
        review_exercise(catalog, record["id"], record["revision"], "rejected")

    with pytest.raises(ValueError, match="stale_exercises"):
        search(index, "Cloud press", CatalogChangingEmbedder(change))
    accept(catalog)
    with pytest.raises(ValueError, match="stale_exercises"):
        ingest(
            source,
            index,
            1,
            2,
            CatalogChangingEmbedder(change),
            approval={"sha256": inspect_source(source, 1, 2)["sha256"], "pages": [1, 2]},
            exercises=catalog,
        )
    assert index.read_bytes() == before


def test_shared_line_fields_use_exact_unique_source_quotes(tmp_path):
    source, catalog, layout = fixture(tmp_path)
    selector = layout["records"][0]["fields"]["name"][0]
    layout["records"][0]["fields"]["muscles"] = [{**selector, "quote": "Cloud"}]
    layout["records"][0]["fields"]["difficulty"] = [{**selector, "quote": "press"}]
    record = draft_exercises(source, catalog, layout, 1, 2)["records"][0]
    assert record["muscles"] == "Cloud"
    assert record["difficulty"] == "press"
    assert record["citations"]["difficulty"][0]["quote"] == "press"
    layout["records"][0]["fields"]["difficulty"][0]["quote"] = "invented"
    before = catalog.read_bytes()
    with pytest.raises(ValueError, match="invalid_exercise_layout"):
        draft_exercises(source, catalog, layout, 1, 2)
    assert catalog.read_bytes() == before


def test_variation_boundary_cannot_select_a_sibling_region(tmp_path):
    source, catalog, layout = fixture(tmp_path)
    layout["records"][0]["variations"][0]["boundaries"] = [
        {"pdf_page": 2, "bbox": [40, 80, 510, 155]}
    ]
    layout["records"][0]["variations"][0]["fields"]["instructions"] = [
        {"pdf_page": 2, "bbox": [45, 40, 500, 65]}
    ]
    with pytest.raises(ValueError, match="invalid_exercise_layout"):
        draft_exercises(source, catalog, layout, 1, 2)


def accept(catalog):
    for record in inspect_exercises(catalog)["records"]:
        review_exercise(catalog, record["id"], record["revision"], "accepted")


def build(source, catalog, tmp_path):
    index = tmp_path / "index.sqlite"
    ingest(
        source,
        index,
        1,
        2,
        AuthoredEmbedder(),
        approval={"sha256": inspect_source(source, 1, 2)["sha256"], "pages": [1, 2]},
        exercises=catalog,
    )
    return index


def test_reviewed_parent_retains_continuation_variation_and_missing_attributes(tmp_path):
    source, catalog, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="no_eligible_passages"):
        build(source, catalog, tmp_path)
    accept(catalog)
    from cited_retrieval import report

    assert report(build(source, catalog, tmp_path))["pages"][1]["status"] == "imported"
    result = search(build(source, catalog, tmp_path), "sky push", AuthoredEmbedder())
    assert len(result["passages"]) == 1
    record = result["passages"][0]["exercise"]
    assert record["name"] == "Cloud press"
    assert record["category"] is None
    assert record["muscles"] is None
    assert record["difficulty"] is None
    assert record["warnings"][0]["text"] == "Stop if discomfort develops during the movement."
    assert record["warnings"][0]["location"]["pdf_page"] == 2
    assert record["variations"][0]["name"] == "Wall press"
    assert "Stone curl" not in result["passages"][0]["text"]
    assert (
        search(build(source, catalog, tmp_path), "Wall press", AuthoredEmbedder())["passages"][0][
            "exercise"
        ]["id"]
        == "cloud"
    )
