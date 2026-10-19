"""Authored schedules tested only through public catalog/import/search seams."""

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder

from cited_retrieval import (
    draft_programmes,
    ingest,
    inspect_programmes,
    inspect_source,
    review_programme,
    search,
)


def schedule(tmp_path, exercise_name="Cloud press"):
    source = tmp_path / "schedule.pdf"
    with pymupdf.open() as doc:
        p = doc.new_page()
        p.insert_text((40, 50), "Cloud programme")
        p.insert_text((40, 80), "Summary weeks 1-4")
        p = doc.new_page()
        for x, y, text in [
            (40, 50, "Weeks 1-2"),
            (220, 50, "Day 1"),
            (350, 50, "Day 2"),
            (420, 90, "Week 3"),
            (40, 90, "Strength"),
            (220, 90, "Alternate movements"),
            (40, 130, "Pair A"),
            (120, 130, exercise_name),
            (280, 130, "8 reps"),
            (380, 130, "30 seconds"),
            (40, 170, "Pair A"),
            (120, 170, "Stone curl"),
            (280, 170, "6 reps"),
            (380, 170, "60 seconds"),
            (40, 220, "Wall assistance"),
            (220, 220, "Slow variation"),
            (40, 280, "Alternate"),
            (40, 310, "Alternate each movement in the stated order."),
        ]:
            p.insert_text((x, y), text)
        p.insert_text((40, 740), "aaaaaaaaaaaaaaaaaaaa")
        doc.save(source)

    def sel(page, x, y, width=150):
        return {"pdf_page": page, "bbox": [x - 1, y - 13, x + width, y + 4], "cell": True}

    layout = {
        "records": [
            {
                "id": "cloud-1",
                "kind": "session",
                "coverage": "complete",
                "weeks": [1, 2],
                "day": 1,
                "boundaries": [
                    {"pdf_page": 1, "bbox": [30, 30, 550, 100]},
                    {"pdf_page": 2, "bbox": [30, 30, 550, 240]},
                ],
                "fields": {
                    "programme": [sel(1, 40, 50)],
                    "week_range": [sel(2, 40, 50)],
                    "day": [sel(2, 220, 50, 80)],
                    "category": [sel(2, 40, 90)],
                    "method": [sel(2, 220, 90)],
                },
                "entries": [
                    {
                        "pair": "A",
                        "fields": {
                            "pair": [sel(2, 40, 130, 60)],
                            "exercise": [sel(2, 120, 130)],
                            "reps": [sel(2, 280, 130, 80)],
                            "pause": [sel(2, 380, 130)],
                            "assistance": [sel(2, 40, 220)],
                            "variations": [sel(2, 220, 220)],
                        },
                    },
                    {
                        "pair": "A",
                        "fields": {
                            "pair": [sel(2, 40, 170, 60)],
                            "exercise": [sel(2, 120, 170)],
                            "reps": [sel(2, 280, 170, 80)],
                            "pause": [sel(2, 380, 170)],
                        },
                    },
                ],
            }
        ]
    }
    return source, tmp_path / "programmes.json", layout, sel


def accept(catalog):
    for r in inspect_programmes(catalog)["records"]:
        review_programme(catalog, r["id"], r["revision"], "accepted")


def build(source, catalog, tmp_path, model=None):
    index = tmp_path / "sessions.sqlite"
    ingest(
        source,
        index,
        1,
        2,
        model or AuthoredEmbedder(),
        approval={"sha256": inspect_source(source, 1, 2)["sha256"], "pages": [1, 2]},
        programmes=catalog,
    )
    return index


def test_review_bound_cells_preserve_crosspage_identity_and_order(tmp_path):
    source, catalog, layout, _ = schedule(tmp_path)
    r = draft_programmes(source, catalog, layout, 1, 2)["records"][0]
    assert not r["eligible"]
    assert inspect_source(source, 1, 2)["pages"][1]["reason"] == "suspicious_native"
    assert r["programme"] == "Cloud programme"
    assert r["entries"][0]["exercise"] == "Cloud press"
    assert r["entries"][0]["reps"] == "8 reps"
    assert r["entries"][0]["citations"]["exercise"][0]["location"]["pdf_page"] == 2
    accept(catalog)
    assert inspect_programmes(catalog)["records"][0]["eligible"]


def test_exact_session_ignores_similarity_and_returns_every_cell(tmp_path):
    source, catalog, layout, _ = schedule(tmp_path)
    draft_programmes(source, catalog, layout, 1, 2)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    result = search(
        index,
        "",
        AuthoredEmbedder(fixed_vector=[0.0, -1.0]),
        programme="Cloud programme",
        week=2,
        day=1,
    )
    assert result["status"] == "evidence"
    session = result["passages"][0]["session"]
    assert session["weeks"] == [1, 2]
    assert [e["exercise"] for e in session["entries"]] == ["Cloud press", "Stone curl"]
    assert [e["within_pair_order"] for e in session["entries"]] == [1, 2]
    assert [e["pause"] for e in session["entries"]] == ["30 seconds", "60 seconds"]
    assert session["entries"][1]["assistance"] is None
    assert session["entries"][0]["exercise_reference"]["status"] == "unresolved"
    assert "Wall assistance" in result["passages"][0]["text"]


@pytest.mark.parametrize(
    "filters",
    [
        {"week": 1},
        {"programme": "Cloud programme"},
        {"day": 1},
        {"programme": None, "week": 1, "day": 1},
        {"programme": "", "week": 1, "day": 1},
        {"programme": "Cloud programme", "week": 0, "day": 1},
        {"programme": "Cloud programme", "week": True, "day": 1},
        {"programme": "Cloud programme", "week": 1.0, "day": 1},
        {"programme": "Cloud programme", "week": 1, "day": None},
        {"programme": "Cloud programme", "week": 1, "day": False},
        {"programme": "Cloud programme", "week": 1, "day": 1.0},
        {"programme": "Cloud programme", "week": 1, "day": 8},
    ],
)
def test_partial_and_invalid_filters_never_broaden(tmp_path, filters):
    with pytest.raises(ValueError, match="invalid_session_filter"):
        search(tmp_path / "not-even-an-index", "Cloud", AuthoredEmbedder(), **filters)


def test_valid_but_unmatched_filters_never_fall_back(tmp_path):
    source, catalog, layout, _ = schedule(tmp_path)
    draft_programmes(source, catalog, layout, 1, 2)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    for programme, week, day in [
        ("Cloud", 1, 1),
        ("Cloud programme", 3, 1),
        ("Cloud programme", 1, 2),
    ]:
        result = search(
            index, "Cloud press", AuthoredEmbedder(), programme=programme, week=week, day=day
        )
        assert result["status"] == "no_evidence"
        assert result["passages"] == []


def test_references_and_both_conflict_claims_are_source_cited(tmp_path):
    from cited_retrieval import draft_exercises, review_exercise

    source, catalog, layout, sel = schedule(tmp_path)
    # A separate, independently reviewed supporting artifact, not a copied source passage.
    support_source = tmp_path / "support.pdf"
    with pymupdf.open() as doc:
        p = doc.new_page()
        p.insert_text((40, 50), "Cloud press")
        p.insert_text((40, 80), "Keep the wrist steady and lift slowly.")
        doc.save(support_source)
    exercises = tmp_path / "exercises.json"
    support = draft_exercises(
        support_source,
        exercises,
        {
            "records": [
                {
                    "id": "cloud",
                    "coverage": "complete",
                    "boundaries": [{"pdf_page": 1, "bbox": [30, 30, 550, 100]}],
                    "fields": {"name": [sel(1, 40, 50)], "instructions": [sel(1, 40, 80, 450)]},
                }
            ]
        },
        1,
        1,
    )
    r = support["records"][0]
    review_exercise(exercises, r["id"], r["revision"], "accepted")
    layout["records"][0]["conflicts"] = [{"claims": [[sel(1, 40, 80, 220)], [sel(2, 40, 50)]]}]
    layout["records"].append(
        {
            "id": "alternating",
            "kind": "method",
            "coverage": "complete",
            "aliases": ["Alternating", "Alternate movements"],
            "boundaries": [{"pdf_page": 2, "bbox": [30, 260, 550, 330]}],
            "fields": {"name": [sel(2, 40, 280)], "definition": [sel(2, 40, 310, 450)]},
        }
    )
    draft_programmes(source, catalog, layout, 1, 2, exercise_catalogs=[exercises])
    accept(catalog)
    index = build(source, catalog, tmp_path)
    result = search(index, "", AuthoredEmbedder(), programme="Cloud programme", week=1, day=1)
    session = result["passages"][0]["session"]
    ref = session["entries"][0]["exercise_reference"]
    assert ref["status"] == "resolved"
    assert ref["record"]["source_id"] == inspect_source(support_source, 1, 1)["sha256"]
    assert ref["record"]["instructions"][0]["location"]["pdf_page"] == 1
    assert session["entries"][1]["exercise_reference"]["status"] == "unresolved"
    method = session["method_reference"]
    assert method["status"] == "resolved"
    assert method["record"]["definition"] == "Alternate each movement in the stated order."
    assert method["record"]["citations"]["definition"][0]["location"]["pdf_page"] == 2
    claims = session["conflicts"][0]["claims"]
    assert [c[0]["text"] for c in claims] == ["Summary weeks 1-4", "Weeks 1-2"]
    assert [c[0]["location"]["pdf_page"] for c in claims] == [1, 2]
    assert session["weeks"] == [1, 2]


def test_empty_phantom_cell_withholds_session_and_preserves_existing_index(tmp_path):
    source, catalog, layout, sel = schedule(tmp_path)
    draft_programmes(source, catalog, layout, 1, 2)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    before = index.read_bytes()
    layout["records"][0]["entries"].append({"fields": {"exercise": [sel(2, 450, 200, 60)]}})
    draft_programmes(source, catalog, layout, 1, 2)
    accept(catalog)
    assert not inspect_programmes(catalog)["records"][0]["eligible"]
    with pytest.raises(ValueError, match="no_eligible_passages"):
        build(source, catalog, tmp_path)
    assert index.read_bytes() == before


def test_review_revocation_during_inference_invalidates_query_and_atomic_rebuild(tmp_path):
    from catalog_changing_embedder import CatalogChangingEmbedder

    from cited_retrieval import report

    source, catalog, layout, _ = schedule(tmp_path)
    draft_programmes(source, catalog, layout, 1, 2)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    before = index.read_bytes()
    r = inspect_programmes(catalog)["records"][0]

    def revoke():
        review_programme(catalog, r["id"], r["revision"], "rejected")

    with pytest.raises(ValueError, match="stale_programmes"):
        search(
            index,
            "Cloud",
            CatalogChangingEmbedder(revoke),
            programme="Cloud programme",
            week=1,
            day=1,
        )
    with pytest.raises(ValueError, match="stale_programmes"):
        report(index)
    accept(catalog)
    with pytest.raises(ValueError, match="stale_programmes"):
        build(source, catalog, tmp_path, CatalogChangingEmbedder(revoke))
    assert index.read_bytes() == before


def test_multicolumn_sessions_do_not_mix_week_day_or_category(tmp_path):
    import copy

    source, catalog, layout, sel = schedule(tmp_path)
    other = copy.deepcopy(layout["records"][0])
    other.update(id="cloud-2", weeks=[3], day=2)
    other["fields"]["day"] = [sel(2, 350, 50)]
    other["fields"]["week_range"] = [sel(2, 420, 90, 100)]
    other["entries"] = [other["entries"][1]]
    layout["records"].append(other)
    draft_programmes(source, catalog, layout, 1, 2)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    for week, day, expected in [(1, 1, "cloud-1"), (2, 1, "cloud-1"), (3, 2, "cloud-2")]:
        result = search(
            index,
            "Cloud",
            AuthoredEmbedder(fixed_vector=[0, -1]),
            programme="Cloud programme",
            week=week,
            day=day,
        )
        assert [p["session"]["id"] for p in result["passages"]] == [expected]
        assert result["passages"][0]["session"]["category"] == "Strength"


def test_multiple_support_catalogs_are_pinned_and_ambiguous_names_stay_unresolved(tmp_path):
    from test_exercises import accept as accept_exercises
    from test_exercises import fixture as exercise_fixture

    from cited_retrieval import inspect_exercises, report, review_exercise

    source, catalog, layout, _ = schedule(tmp_path)
    deps = []
    for name in ["first", "second"]:
        directory = tmp_path / name
        directory.mkdir()
        _, exercises, _ = exercise_fixture(directory)
        accept_exercises(exercises)
        deps.append(exercises)
    draft_programmes(source, catalog, layout, 1, 2, exercise_catalogs=deps)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    result = search(index, "Cloud", AuthoredEmbedder(), programme="Cloud programme", week=1, day=1)
    ref = result["passages"][0]["session"]["entries"][0]["exercise_reference"]
    assert ref["status"] == "unresolved"
    assert ref["reason"] == "ambiguous"
    assert len(ref["candidates"]) == 2
    assert "Lift slowly" not in str(result)
    r = inspect_exercises(deps[0])["records"][0]
    review_exercise(deps[0], r["id"], r["revision"], "rejected")
    with pytest.raises(ValueError, match="stale_programmes"):
        report(index)
    with pytest.raises(ValueError, match="stale_programmes"):
        search(index, "Cloud", AuthoredEmbedder())


def test_nonprogramme_index_never_answers_session_filter_with_prose_or_exercise(tmp_path):
    from test_exercises import accept as accept_exercises
    from test_exercises import build as build_exercises
    from test_exercises import fixture as exercise_fixture

    source, catalog, _ = exercise_fixture(tmp_path)
    accept_exercises(catalog)
    index = build_exercises(source, catalog, tmp_path)
    result = search(
        index, "Cloud press", AuthoredEmbedder(), programme="Cloud programme", week=1, day=1
    )
    assert result["status"] == "no_evidence"
    assert result["passages"] == []


def test_source_and_reviewed_normalization_change_cannot_keep_acceptance(tmp_path):
    from cited_retrieval import report

    source, catalog, layout, _ = schedule(tmp_path)
    draft_programmes(source, catalog, layout, 1, 2)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    old = inspect_programmes(catalog)["records"][0]["revision"]
    layout["records"][0]["weeks"] = [1]
    r = draft_programmes(source, catalog, layout, 1, 2)["records"][0]
    assert r["state"] == "pending"
    assert r["revision"] != old
    with pytest.raises(ValueError, match="review_revision_mismatch"):
        review_programme(catalog, r["id"], old, "accepted")
    with pytest.raises(ValueError, match="stale_programmes"):
        report(index)
    source.write_bytes(source.read_bytes() + b"\n% authored change")
    with pytest.raises(ValueError, match="stale_programmes"):
        inspect_programmes(catalog)


def test_one_native_line_can_supply_distinct_cells_and_numeric_quantities(tmp_path):
    source = tmp_path / "one-line.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 50), "Cloud programme")
        page.insert_text((40, 80), "Weeks 1-2 Day 1")
        page.insert_text((40, 130), "Cloud press        8        30 sec", fontname="cour")
        words = page.get_text("words")
        doc.save(source)

    def box_for(*names):
        boxes = [w[:4] for w in words if w[4] in names and w[1] > 100]
        return [
            min(b[0] for b in boxes) - 0.1,
            min(b[1] for b in boxes) - 0.1,
            max(b[2] for b in boxes) + 0.1,
            max(b[3] for b in boxes) + 0.1,
        ]

    def cell(box, quote=None):
        value = {"pdf_page": 1, "bbox": box}
        if quote is not None:
            value["quote"] = quote
        return value

    scope = [35, 65, 300, 90]
    layout = {
        "records": [
            {
                "id": "single",
                "kind": "session",
                "coverage": "complete",
                "weeks": [1, 2],
                "day": 1,
                "boundaries": [cell([30, 30, 550, 160])],
                "fields": {
                    "programme": [cell([35, 35, 300, 55])],
                    "week_range": [cell(scope, "Weeks 1-2")],
                    "day": [cell(scope, "Day 1")],
                },
                "entries": [
                    {
                        "fields": {
                            "exercise": [cell(box_for("Cloud", "press"))],
                            "reps": [cell(box_for("8"))],
                            "pause": [cell(box_for("30", "sec"))],
                        }
                    }
                ],
            }
        ]
    }
    catalog = tmp_path / "programmes.json"
    r = draft_programmes(source, catalog, layout, 1, 1)["records"][0]
    assert r["coverage"] == "complete"
    assert r["entries"][0]["exercise"] == "Cloud press"
    assert r["entries"][0]["reps"] == "8"
    assert r["entries"][0]["pause"] == "30 sec"
    assert r["day"] == "Day 1"
    assert r["citations"]["day"][0]["lines"][0]["text"] == "Weeks 1-2 Day 1"
    assert len(r["entries"][0]["citations"]["exercise"][0]["render_sha256"]) == 64


def test_pending_and_ambiguous_methods_never_leak_definitions(tmp_path):
    import copy

    source, catalog, layout, sel = schedule(tmp_path)
    method = {
        "id": "alternate",
        "kind": "method",
        "coverage": "complete",
        "aliases": ["Alternate movements"],
        "boundaries": [{"pdf_page": 2, "bbox": [30, 260, 550, 330]}],
        "fields": {"name": [sel(2, 40, 280)], "definition": [sel(2, 40, 310, 450)]},
    }
    layout["records"].append(method)
    records = draft_programmes(source, catalog, layout, 1, 2)["records"]
    review_programme(catalog, records[0]["id"], records[0]["revision"], "accepted")
    result = search(
        build(source, catalog, tmp_path),
        "Cloud",
        AuthoredEmbedder(),
        programme="Cloud programme",
        week=1,
        day=1,
    )
    ref = result["passages"][0]["session"]["method_reference"]
    assert ref == {"status": "unresolved", "reason": "not_eligible"}
    assert "Alternate each movement" not in str(result)
    accept(catalog)
    result = search(
        build(source, catalog, tmp_path),
        "Cloud",
        AuthoredEmbedder(),
        programme="Cloud programme",
        week=1,
        day=1,
    )
    assert result["passages"][0]["session"]["method_reference"]["status"] == "resolved"
    duplicate = copy.deepcopy(method)
    duplicate["id"] = "other-alternate"
    layout["records"].append(duplicate)
    draft_programmes(source, catalog, layout, 1, 2)
    accept(catalog)
    result = search(
        build(source, catalog, tmp_path),
        "Cloud",
        AuthoredEmbedder(),
        programme="Cloud programme",
        week=1,
        day=1,
    )
    assert result["passages"][0]["session"]["method_reference"]["reason"] == "ambiguous"
    assert "Alternate each movement" not in str(result)


def test_reference_requires_a_whole_reviewed_name_not_a_known_substring(tmp_path):
    from test_exercises import accept as accept_exercises
    from test_exercises import fixture as exercise_fixture

    source, catalog, layout, _ = schedule(tmp_path, "Cloud press unknown")
    directory = tmp_path / "support"
    directory.mkdir()
    _, exercises, _ = exercise_fixture(directory)
    accept_exercises(exercises)
    draft_programmes(source, catalog, layout, 1, 2, exercise_catalogs=[exercises])
    accept(catalog)
    result = search(
        build(source, catalog, tmp_path),
        "Cloud",
        AuthoredEmbedder(),
        programme="Cloud programme",
        week=1,
        day=1,
    )
    entry = result["passages"][0]["session"]["entries"][0]
    assert entry["exercise"] == "Cloud press unknown"
    assert entry["exercise_reference"]["status"] == "unresolved"


def test_resolved_method_definition_page_counts_as_imported_evidence(tmp_path):
    import json

    from cited_retrieval import report

    original, catalog, layout, sel = schedule(tmp_path)
    source = tmp_path / "three-pages.pdf"
    with pymupdf.open(original) as old, pymupdf.open() as doc:
        doc.insert_pdf(old, from_page=0, to_page=0)
        p = doc.new_page()
        p.insert_text((40, 50), "Alternate")
        p.insert_text((40, 80), "Alternate each movement in the stated order.")
        doc.insert_pdf(old, from_page=1, to_page=1)
        doc.save(source)
    layout = json.loads(json.dumps(layout).replace('"pdf_page": 2', '"pdf_page": 3'))
    layout["records"].append(
        {
            "id": "method",
            "kind": "method",
            "coverage": "complete",
            "aliases": ["Alternate movements"],
            "boundaries": [{"pdf_page": 2, "bbox": [30, 30, 550, 100]}],
            "fields": {"name": [sel(2, 40, 50)], "definition": [sel(2, 40, 80, 450)]},
        }
    )
    draft_programmes(source, catalog, layout, 1, 3)
    accept(catalog)
    index = tmp_path / "index.sqlite"
    ingest(
        source,
        index,
        1,
        3,
        AuthoredEmbedder(),
        programmes=catalog,
        approval={"sha256": inspect_source(source, 1, 3)["sha256"], "pages": [1, 2, 3]},
    )
    assert report(index)["pages"][1]["status"] == "imported"


@pytest.mark.parametrize(
    "weeks,day",
    [([], 1), ([True], 1), ([1.0], 1), ([0], 1), ([1, 1], 1), ([1], True), ([1], 1.0), ([1], 0)],
)
def test_draft_rejects_invalid_normalized_scope(tmp_path, weeks, day):
    source, catalog, layout, _ = schedule(tmp_path)
    layout["records"][0].update(weeks=weeks, day=day)
    with pytest.raises(ValueError, match="invalid_programme_layout"):
        draft_programmes(source, catalog, layout, 1, 2)


def test_support_revocation_during_embedding_is_not_a_snapshot_bypass(tmp_path):
    from catalog_changing_embedder import CatalogChangingEmbedder
    from test_exercises import accept as accept_exercises
    from test_exercises import fixture as exercise_fixture

    from cited_retrieval import inspect_exercises, review_exercise

    source, catalog, layout, _ = schedule(tmp_path)
    directory = tmp_path / "support"
    directory.mkdir()
    _, exercises, _ = exercise_fixture(directory)
    accept_exercises(exercises)
    draft_programmes(source, catalog, layout, 1, 2, exercise_catalogs=[exercises])
    accept(catalog)
    index = build(source, catalog, tmp_path)
    before = index.read_bytes()
    r = inspect_exercises(exercises)["records"][0]

    def revoke():
        review_exercise(exercises, r["id"], r["revision"], "rejected")

    with pytest.raises(ValueError, match="stale_programmes"):
        search(
            index,
            "Cloud",
            CatalogChangingEmbedder(revoke),
            programme="Cloud programme",
            week=1,
            day=1,
        )
    accept_exercises(exercises)
    draft_programmes(source, catalog, layout, 1, 2, exercise_catalogs=[exercises])
    accept(catalog)
    with pytest.raises(ValueError, match="stale_programmes"):
        build(source, catalog, tmp_path, CatalogChangingEmbedder(revoke))
    assert index.read_bytes() == before


def test_phantom_duplicate_row_cannot_be_accepted(tmp_path):
    import copy

    source, catalog, layout, _ = schedule(tmp_path)
    layout["records"][0]["entries"].append(copy.deepcopy(layout["records"][0]["entries"][0]))
    with pytest.raises(ValueError, match="invalid_programme_layout"):
        draft_programmes(source, catalog, layout, 1, 2)
