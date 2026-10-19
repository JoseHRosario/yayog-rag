from pathlib import Path

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder

from cited_retrieval import (
    draft_excerpts,
    ingest,
    inspect_excerpts,
    report,
    review_excerpt,
    search,
)


def test_reviewed_cross_page_excerpt_excludes_collateral_damage(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 70), "Rest the widget for twelve seconds, but")
        font = page.insert_font(fontname="damaged", fontbuffer=pymupdf.Font("cjk").buffer)
        page.insert_text((50, 180), "Unrelated damaged footer.", fontname="damaged")
        doc.xref_set_key(font, "ToUnicode", "null")
        page = doc.new_page()
        page.insert_text((50, 70), "stop immediately if the red warning lamp flashes.")
        page.insert_text((50, 180), "Unselected adjacent material must not enter context.")
        doc.set_toc([[1, "Widget precautions", 1]])
        doc.save(source)
    selectors = [{"pdf_page": p, "bbox": [45, 50, 400, 90]} for p in (1, 2)]
    layout = {
        "records": [
            {
                "id": "paragraph-1",
                "coverage": "complete",
                "boundaries": selectors,
                "selectors": selectors,
            }
        ]
    }
    catalog, index = tmp_path / "excerpts.json", tmp_path / "index.sqlite"
    draft = draft_excerpts(source, catalog, layout, 1, 2)
    record = draft["records"][0]
    assert record["state"] == "pending" and not record["eligible"]
    approval = {"sha256": draft["binding"]["source_sha256"], "pages": [1, 2]}
    with pytest.raises(ValueError, match="no_eligible_passages"):
        ingest(source, index, 1, 2, AuthoredEmbedder(), approval=approval, excerpts=catalog)
    review_excerpt(catalog, record["id"], record["revision"], "accepted")
    imported = ingest(source, index, 1, 2, AuthoredEmbedder(), approval=approval, excerpts=catalog)
    assert {p["status"] for p in imported["pages"]} == {"partial"}
    result = search(index, "widget warning seconds", AuthoredEmbedder())
    passage = result["passages"][0]
    assert (
        passage["text"]
        == "Rest the widget for twelve seconds, but\nstop immediately if the red warning lamp flashes."
    )
    assert passage["location"]["chapter"] == "Widget precautions"
    assert [s["location"]["pdf_page"] for s in passage["excerpt"]["segments"]] == [1, 2]
    assert not result["page_context"]
    assert report(index)["excerpts"]["records"][0]["id"] == "paragraph-1"
    review_excerpt(catalog, record["id"], record["revision"], "pending")
    with pytest.raises(ValueError, match="stale_excerpts"):
        report(index)
    assert not inspect_excerpts(catalog)["records"][0]["eligible"]


def fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 70), "Rest the widget for twelve seconds before resuming.")
        page.insert_text((50, 150), "aaaaaaaaaaaa this line is deliberately suspicious.")
        doc.save(source)
    selector = {"pdf_page": 1, "bbox": [45, 50, 400, 90]}
    layout: dict[str, object] = {
        "records": [
            {"id": "p1", "coverage": "complete", "boundaries": [selector], "selectors": [selector]}
        ]
    }
    return source, tmp_path / "excerpts.json", layout


def test_excerpt_coverage_tracks_catalog_layout_and_cross_page_record(tmp_path: Path) -> None:
    import json

    from cited_retrieval import draft_coverage, inspect_coverage

    source, catalog, layout = fixture(tmp_path)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout))
    draft = draft_excerpts(source, catalog, layout_path, 1, 1)
    record = draft["records"][0]
    review_excerpt(catalog, "p1", record["revision"], "accepted")
    index = tmp_path / "index.sqlite"
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": draft["binding"]["source_sha256"], "pages": [1]},
        excerpts=catalog,
    )
    audit = tmp_path / "audit.json"
    coverage = draft_coverage(source, audit, indexes=[index])
    assert [(r["kind"], r["pdf_pages"], r["status"]) for r in coverage["records"]] == [
        ("excerpt", [1], "trusted")
    ]
    paths = {f["path"] for f in coverage["binding"]["artifacts"]["files"]}
    assert {str(source), str(catalog), str(layout_path), str(index)} <= paths
    assert not coverage["coverage_complete"]
    layout_path.write_text(json.dumps({"records": []}))
    with pytest.raises(ValueError, match="stale_excerpts"):
        inspect_coverage(audit)


def test_bad_selected_glyphs_withhold_record_and_selector_change_resets_review(
    tmp_path: Path,
) -> None:
    source, catalog, layout = fixture(tmp_path)
    draft = draft_excerpts(source, catalog, layout, 1, 1)
    record = draft["records"][0]
    assert record["coverage"] == "complete"
    review_excerpt(catalog, "p1", record["revision"], "accepted")
    broad = {"pdf_page": 1, "bbox": [45, 50, 400, 170]}
    changed = {
        "records": [
            {"id": "p1", "coverage": "complete", "boundaries": [broad], "selectors": [broad]}
        ]
    }
    revised = draft_excerpts(source, catalog, changed, 1, 1)["records"][0]
    assert revised["state"] == "pending"
    assert revised["revision"] != record["revision"]
    assert revised["gaps"] == ["unresolved_native_text"]
    review_excerpt(catalog, "p1", revised["revision"], "accepted")
    assert not inspect_excerpts(catalog)["records"][0]["eligible"]


def test_invalid_excerpt_draft_keeps_destination_and_protected_inputs(tmp_path: Path) -> None:
    source, catalog, layout = fixture(tmp_path)
    draft_excerpts(source, catalog, layout, 1, 1)
    before = catalog.read_bytes()
    with pytest.raises(ValueError, match="invalid_excerpt_layout"):
        draft_excerpts(source, catalog, {"records": []}, 1, 1)
    assert catalog.read_bytes() == before
    with pytest.raises(ValueError, match="unsafe_review_path"):
        draft_excerpts(source, source, layout, 1, 1)
    checkout = tmp_path / "linked"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: ../git/worktrees/linked")
    with pytest.raises(ValueError, match="unsafe_review_path"):
        draft_excerpts(source, checkout / "private.json", layout, 1, 1)
    moved = checkout / "moved.json"
    moved.write_bytes(before)
    with pytest.raises(ValueError, match="unsafe_review_path"):
        inspect_excerpts(moved)


def test_excerpt_cli_draft_inspect_review(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from cited_retrieval.cli import main

    source, catalog, layout = fixture(tmp_path)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout))
    assert (
        main(
            [
                "draft-excerpts",
                str(source),
                "--catalog",
                str(catalog),
                "--layout",
                str(layout_path),
                "--start",
                "1",
                "--end",
                "1",
            ]
        )
        == 0
    )
    record = json.loads(capsys.readouterr().out)["records"][0]
    assert (
        main(
            [
                "review-excerpt",
                str(catalog),
                "--record",
                "p1",
                "--revision",
                record["revision"],
                "--state",
                "accepted",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["records"][0]["eligible"]
    assert main(["inspect-excerpts", str(catalog)]) == 0
    assert json.loads(capsys.readouterr().out)["records"][0]["eligible"]


@pytest.mark.parametrize("during_search", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_excerpt_revocation_during_embedding_revalidates_even_on_failure(
    tmp_path: Path, during_search: bool, fail: bool
) -> None:
    from input_mutating_embedder import InputMutatingEmbedder

    source, catalog, layout = fixture(tmp_path)
    draft = draft_excerpts(source, catalog, layout, 1, 1)
    record = draft["records"][0]
    review_excerpt(catalog, "p1", record["revision"], "accepted")
    approval = {"sha256": draft["binding"]["source_sha256"], "pages": [1]}
    index = tmp_path / "index.sqlite"
    ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, excerpts=catalog)
    before = index.read_bytes()

    def revoke() -> None:
        review_excerpt(catalog, "p1", record["revision"], "rejected")

    embedder = InputMutatingEmbedder(revoke, fail=fail)
    with pytest.raises(ValueError, match="stale_excerpts"):
        if during_search:
            search(index, "widget rest", embedder)
        else:
            ingest(source, index, 1, 1, embedder, approval=approval, excerpts=catalog)
    assert index.read_bytes() == before


def test_repeated_excerpt_build_keeps_complete_ids_and_original_index(tmp_path: Path) -> None:
    source, catalog, layout = fixture(tmp_path)
    draft = draft_excerpts(source, catalog, layout, 1, 1)
    record = draft["records"][0]
    review_excerpt(catalog, "p1", record["revision"], "accepted")
    approval = {"sha256": draft["binding"]["source_sha256"], "pages": [1]}
    first, second = tmp_path / "first.sqlite", tmp_path / "second.sqlite"
    one = ingest(source, first, 1, 1, AuthoredEmbedder(), approval=approval, excerpts=catalog)
    before = first.read_bytes()
    two = ingest(source, second, 1, 1, AuthoredEmbedder(), approval=approval, excerpts=catalog)
    assert one == two
    assert first.read_bytes() == before
    ids = [
        {p["passage_id"] for p in search(index, "widget rest", AuthoredEmbedder())["passages"]}
        for index in (first, second)
    ]
    assert ids[0] == ids[1] and len(ids[0]) == one["passage_count"]


def test_native_glyph_selection_with_recovery_does_not_require_whole_line(tmp_path: Path) -> None:
    from authored_ocr import AuthoredOCR

    from cited_retrieval import recover

    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 70), "Rest the widget. Unselected adjacent text.")
        doc.save(source)
    with pymupdf.open(source) as doc:
        box = list(doc[0].search_for("Rest the widget.")[0])
    selector = {"pdf_page": 1, "bbox": box}
    layout = {
        "records": [
            {"id": "p", "coverage": "complete", "boundaries": [selector], "selectors": [selector]}
        ]
    }
    bundle = tmp_path / "recovery.json"
    recover(source, bundle, 1, 1, AuthoredOCR())
    draft = draft_excerpts(source, tmp_path / "excerpts.json", layout, 1, 1, recovery=bundle)
    assert draft["records"][0]["text"] == "Rest the widget."
    assert draft["records"][0]["coverage"] == "complete"


def test_long_excerpt_keeps_final_qualification_and_full_answer_packet(tmp_path: Path) -> None:
    import json

    from cited_retrieval import prepare_answer
    from cited_retrieval.openai_completion import OpenAICompletion

    source = tmp_path / "source.pdf"
    lines = ["Rest the widget for twelve seconds before continuing."] * 35 + [
        "However, stop immediately if the red warning lamp flashes."
    ]
    with pymupdf.open() as doc:
        doc.new_page().insert_text((50, 50), "\n".join(lines), fontsize=8)
        doc.save(source)
    selector = {"pdf_page": 1, "bbox": [45, 35, 500, 500]}
    layout = {
        "records": [
            {"id": "p", "coverage": "complete", "boundaries": [selector], "selectors": [selector]}
        ]
    }
    catalog = tmp_path / "excerpts.json"
    drafted = draft_excerpts(source, catalog, layout, 1, 1)
    record = drafted["records"][0]
    review_excerpt(catalog, "p", record["revision"], "accepted")
    index = tmp_path / "index.sqlite"
    result = ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": drafted["binding"]["source_sha256"], "pages": [1]},
        excerpts=catalog,
    )
    assert result["passage_count"] > 1
    evidence = search(index, "widget rest seconds", AuthoredEmbedder(), limit=1)
    assert len(evidence["passages"]) == 1 and not evidence["page_context"]
    assert evidence["passages"][0]["text"] == "\n".join(lines)
    assert len(evidence["passages"][0]["retrieval_text"]) <= 1200
    prepared = prepare_answer(
        index,
        "widget rest seconds",
        AuthoredEmbedder(),
        limit=1,
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True
        ),
        output_dir=tmp_path / "answer",
    )
    assert prepared["status"] == "prepared"
    packet = json.loads(Path(prepared["request_file"]).read_text())
    payload = json.loads(packet["messages"][1]["content"])
    assert payload["retrieval"]["passages"][0]["text"] == "\n".join(lines)
    assert (
        payload["retrieval"]["passages"][0]["excerpt"]["segments"][0]["location"]["bbox"]
        == selector["bbox"]
    )
    assert "excerpt_coverage" in payload["disclosures"]
