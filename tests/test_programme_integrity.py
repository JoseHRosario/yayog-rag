"""Public regressions for programme identity and dependency artifact safety."""

import copy

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR
from test_programmes import accept, build, schedule
from test_recovery import damaged_source

from cited_retrieval import (
    draft_exercises,
    draft_programmes,
    ingest,
    recover,
    review,
    review_exercise,
    review_programme,
    search,
)


@pytest.mark.parametrize("alias_collision", [False, True])
def test_pending_session_identity_blocks_a_definitive_answer(tmp_path, alias_collision):
    source, catalog, layout, _ = schedule(tmp_path)
    pending = copy.deepcopy(layout["records"][0])
    pending["id"] = "pending-correct-session"
    if alias_collision:
        layout["records"][0]["aliases"] = ["Shared programme"]
        pending["aliases"] = ["Shared programme"]
    layout["records"].append(pending)
    value = draft_programmes(source, catalog, layout, 1, 2)
    first = value["records"][0]
    review_programme(catalog, first["id"], first["revision"], "accepted")
    result = search(
        build(source, catalog, tmp_path),
        "",
        AuthoredEmbedder(),
        programme="Shared programme" if alias_collision else "Cloud programme",
        week=1,
        day=1,
    )
    assert result["status"] == "ambiguous"
    assert result["passages"] == []
    assert {c["id"] for c in result["candidates"]} == {"cloud-1", "pending-correct-session"}
    assert "Cloud press" not in str(result)


def test_missing_programme_on_an_unrelated_draft_does_not_break_strict_lookup(tmp_path):
    source, catalog, layout, _ = schedule(tmp_path)
    incomplete = copy.deepcopy(layout["records"][0])
    incomplete["id"] = "missing-identity"
    incomplete["fields"].pop("programme")
    layout["records"].append(incomplete)
    value = draft_programmes(source, catalog, layout, 1, 2)
    first = value["records"][0]
    review_programme(catalog, first["id"], first["revision"], "accepted")
    result = search(
        build(source, catalog, tmp_path),
        "",
        AuthoredEmbedder(),
        programme="Cloud programme",
        week=1,
        day=1,
    )
    assert result["status"] == "evidence"
    assert result["passages"][0]["session"]["id"] == "cloud-1"


def test_outside_glyph_damage_on_shared_line_does_not_veto_clean_cell(tmp_path):
    source, catalog, layout, _ = schedule(tmp_path)
    with pymupdf.open(source) as doc:
        doc.delete_page(0)
        page = doc.new_page(pno=0)
        page.insert_text((40, 50), "Cloud programme" + " " * 30 + "a" * 20)
        page.insert_text((40, 80), "Summary weeks 1-4")
        data = doc.tobytes()
    source.write_bytes(data)
    layout["records"][0]["boundaries"][0]["bbox"] = [30, 30, 195, 100]
    record = draft_programmes(source, catalog, layout, 1, 2)["records"][0]
    assert record["coverage"] == "complete"
    assert not record["gaps"]
    assert record["programme"].strip() == "Cloud programme"


@pytest.mark.parametrize("artifact", ["source", "recovery", "crop"])
@pytest.mark.parametrize("operation", ["import", "draft"])
def test_programme_outputs_cannot_replace_support_dependency_artifacts(
    tmp_path, artifact, operation
):
    support_dir = tmp_path / "support"
    support_dir.mkdir()
    support_source = damaged_source(support_dir)
    with pymupdf.open(support_source) as doc:
        doc[0].insert_text((50, 45), "Cloud press")
        data = doc.tobytes()
    support_source.write_bytes(data)
    bundle = support_dir / "recovery.json"
    recovered = recover(support_source, bundle, 1, 1, AuthoredOCR())
    region = recovered["regions"][0]
    review(bundle, region["id"], region["revision"], "accepted")
    exercises = support_dir / "exercises.json"
    exercise_layout = {
        "records": [
            {
                "id": "cloud",
                "coverage": "complete",
                "boundaries": [{"pdf_page": 1, "bbox": [40, 25, 500, 170]}],
                "fields": {
                    "name": [{"pdf_page": 1, "bbox": [45, 25, 500, 55]}],
                    "instructions": [{"pdf_page": 1, "bbox": [45, 120, 500, 160]}],
                },
            }
        ]
    }
    exercise = draft_exercises(support_source, exercises, exercise_layout, 1, 1, recovery=bundle)[
        "records"
    ][0]
    review_exercise(exercises, exercise["id"], exercise["revision"], "accepted")
    source, catalog, layout, _ = schedule(tmp_path)
    draft_programmes(source, catalog, layout, 1, 2, exercise_catalogs=[exercises])
    accept(catalog)
    target = {"source": support_source, "recovery": bundle, "crop": bundle.parent / region["crop"]}[
        artifact
    ]
    before = target.read_bytes()
    if operation == "draft":
        with pytest.raises(ValueError, match="unsafe_review_path"):
            draft_programmes(source, target, layout, 1, 2, exercise_catalogs=[exercises])
    else:
        from cited_retrieval import inspect_source

        approval = {"sha256": inspect_source(source, 1, 2)["sha256"], "pages": [1, 2]}
        with pytest.raises(ValueError, match="unsafe_index_path"):
            ingest(source, target, 1, 2, AuthoredEmbedder(), approval=approval, programmes=catalog)
    assert target.read_bytes() == before
