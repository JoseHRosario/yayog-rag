from pathlib import Path

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder

from cited_retrieval import ingest, inspect_source, report


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source.pdf"
    doc = pymupdf.open()
    for text in [
        "Preface outside the approved section.",
        "Recovery\n\nRest allows the muscles to recover after training. Sleep supports recovery.",
        "Nutrition\n\nFood supplies energy for movement. Water supports hydration.",
        "",
    ]:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 700), text, fontsize=12)
    doc.set_toc([[1, "Recovery", 2], [1, "Nutrition", 3]])
    doc.set_page_labels([{"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1}])
    doc.save(path)
    doc.close()
    return path


def test_import_requires_artifact_bound_review_and_accounts_every_page(
    source: Path, tmp_path: Path
):
    preview = inspect_source(source, 2, 4)
    index = tmp_path / "index.sqlite"
    with pytest.raises(ValueError, match="review_required"):
        ingest(source, index, 2, 4, AuthoredEmbedder())
    result = ingest(
        source,
        index,
        2,
        4,
        AuthoredEmbedder(),
        approval={"sha256": preview["sha256"], "pages": [2, 3, 4]},
    )
    assert [(p["pdf_page"], p["status"], p["reason"]) for p in result["pages"]] == [
        (1, "excluded", "not_selected"),
        (2, "imported", None),
        (3, "imported", None),
        (4, "excluded", "no_native_text"),
    ]
    assert result["passage_count"] == 2
    assert report(index) == result
    assert inspect_source(source, 2, 4)["sha256"] == preview["sha256"]


def test_hybrid_retrieval_returns_stable_citations_and_semantic_only_matches(
    source: Path, tmp_path: Path
):
    from cited_retrieval import search

    index = tmp_path / "index.sqlite"
    preview = inspect_source(source, 2, 3)
    approval = {"sha256": preview["sha256"], "pages": [2, 3]}
    ingest(source, index, 2, 3, AuthoredEmbedder(), approval=approval)
    evidence = search(index, "tired", AuthoredEmbedder())
    assert evidence["status"] == "evidence"
    first = evidence["passages"][0]
    assert "Sleep supports recovery" in first["text"]
    assert first["location"]["pdf_page"] == 2
    assert first["location"]["printed_label"] == "ii"
    assert first["location"]["chapter"] == "Recovery"
    assert first["eligible"] is True
    assert first["review_status"] == "acknowledged"
    assert first["provenance"]["extractor"] == "pymupdf-blocks"
    assert first["lexical_rank"] is None
    assert first["semantic_score"] == pytest.approx(1.0)
    lexical = search(index, "hydration", AuthoredEmbedder())["passages"][0]
    assert lexical["location"]["chapter"] == "Nutrition"
    assert lexical["lexical_rank"] == 1
    ingest(source, index, 2, 3, AuthoredEmbedder(), approval=approval)
    assert (
        search(index, "tired", AuthoredEmbedder())["passages"][0]["passage_id"]
        == first["passage_id"]
    )
    assert report(index)["passage_count"] == 2
    assert len(report(index)["corpus_sha256"]) == 64


def test_rejects_stale_source_and_model_and_preserves_prior_index_on_failed_rebuild(
    source: Path, tmp_path: Path
):
    from cited_retrieval import search

    index = tmp_path / "index.sqlite"
    approval = {"sha256": inspect_source(source, 2, 3)["sha256"], "pages": [2, 3]}
    ingest(source, index, 2, 3, AuthoredEmbedder(), approval=approval)
    original = index.read_bytes()
    model = AuthoredEmbedder()
    model.identity = {**model.identity, "version": "2"}
    with pytest.raises(ValueError, match="incompatible_model"):
        search(index, "rest", model)
    model.identity = {**model.identity, "version": "1", "artifact_sha256": "b" * 64}
    with pytest.raises(ValueError, match="incompatible_model"):
        search(index, "rest", model)
    with pytest.raises(ValueError, match="review_required"):
        ingest(source, index, 2, 3, model, approval={"sha256": "bad", "pages": [2, 3]})
    assert index.read_bytes() == original
    source.write_bytes(source.read_bytes() + b"\n% changed artifact")
    with pytest.raises(ValueError, match="stale_source"):
        search(index, "rest", AuthoredEmbedder())
    with pytest.raises(ValueError, match="review_required"):
        ingest(source, index, 2, 3, AuthoredEmbedder(), approval=approval)
    assert index.read_bytes() == original


def test_source_cannot_be_used_as_index(source: Path):
    original_source = source.read_bytes()
    preview = inspect_source(source, 2, 3)
    with pytest.raises(ValueError, match="unsafe_index_path"):
        ingest(
            source,
            source,
            2,
            3,
            AuthoredEmbedder(),
            approval={"sha256": preview["sha256"], "pages": [2, 3]},
        )
    assert source.read_bytes() == original_source


def test_no_evidence_and_bad_inference_do_not_produce_fallback_passages(
    source: Path, tmp_path: Path
):
    from cited_retrieval import search

    index = tmp_path / "index.sqlite"
    approval = {"sha256": inspect_source(source, 2, 3)["sha256"], "pages": [2, 3]}
    ingest(source, index, 2, 3, AuthoredEmbedder(), approval=approval)

    off_topic = AuthoredEmbedder(fixed_vector=[-1.0, -1.0])
    assert search(index, "orbital mechanics", off_topic)["status"] == "no_evidence"
    assert search(index, "orbital mechanics", off_topic)["passages"] == []
    before = index.read_bytes()
    broken = AuthoredEmbedder(fixed_vector=[float("nan"), 0.0])
    with pytest.raises(ValueError, match="invalid_embeddings"):
        ingest(source, index, 2, 3, broken, approval=approval)
    assert index.read_bytes() == before


def test_paragraph_boundaries_are_preserved(source: Path, tmp_path: Path):
    from cited_retrieval import search

    doc = pymupdf.open(source)
    page = doc[1]
    page.insert_text((50, 300), "Gentle walking is another option for comfortable recovery.")
    revised = tmp_path / "paragraphs.pdf"
    doc.save(revised)
    doc.close()
    approval = {"sha256": inspect_source(revised, 2, 2)["sha256"], "pages": [2]}
    index = tmp_path / "index.sqlite"
    result = ingest(revised, index, 2, 2, AuthoredEmbedder(), approval=approval)
    assert result["passage_count"] == 2
    for passage in search(index, "recovery", AuthoredEmbedder())["passages"]:
        assert not ("Gentle walking" in passage["text"] and "Sleep supports" in passage["text"])


def test_nonempty_native_text_without_unicode_mapping_is_not_approved(tmp_path: Path):
    path = tmp_path / "unmapped.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    xref = page.insert_font(fontname="fixture", fontbuffer=pymupdf.Font("cjk").buffer)
    page.insert_text(
        (50, 100), "Rest and sleep support recovery after training.", fontname="fixture"
    )
    doc.xref_set_key(xref, "ToUnicode", "null")
    doc.save(path)
    doc.close()
    preview = inspect_source(path, 1, 1)
    assert preview["pages"][0]["status"] == "excluded"
    assert preview["pages"][0]["reason"] == "missing_unicode_map"


def test_standard_encoded_simple_font_does_not_require_redundant_unicode_map(tmp_path: Path):
    path = tmp_path / "standard-encoding.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    xref = page.insert_font(
        fontname="fixture", fontbuffer=pymupdf.Font("helv").buffer, set_simple=True
    )
    page.insert_text(
        (50, 100), "Rest and sleep support recovery after training.", fontname="fixture"
    )
    doc.xref_set_key(xref, "ToUnicode", "null")
    doc.save(path)
    doc.close()
    preview = inspect_source(path, 1, 1)
    assert preview["pages"][0]["status"] == "candidate"
    approval = {"sha256": preview["sha256"], "pages": [1]}
    index = tmp_path / "index.sqlite"
    result = ingest(path, index, 1, 1, AuthoredEmbedder(), approval=approval)
    assert result["passage_count"] == 1


def test_font_changes_inside_words_do_not_insert_characters(tmp_path: Path):
    from cited_retrieval import search

    path = tmp_path / "mixed-fonts.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 100), "R", fontsize=11)
    offset = pymupdf.get_text_length("R", fontsize=11)
    page.insert_text(
        (50 + offset, 100),
        "est supports recovery after demanding training.",
        fontsize=11,
        fontname="hebo",
    )
    doc.save(path)
    doc.close()
    preview = inspect_source(path, 1, 1)
    index = tmp_path / "index.sqlite"
    ingest(
        path, index, 1, 1, AuthoredEmbedder(), approval={"sha256": preview["sha256"], "pages": [1]}
    )
    passage = search(index, "recovery", AuthoredEmbedder())["passages"][0]
    assert passage["text"] == "Rest supports recovery after demanding training."


def test_common_question_words_do_not_make_unrelated_content_evidence(source: Path, tmp_path: Path):
    from cited_retrieval import search

    index = tmp_path / "index.sqlite"
    preview = inspect_source(source, 2, 3)
    ingest(
        source,
        index,
        2,
        3,
        AuthoredEmbedder(),
        approval={"sha256": preview["sha256"], "pages": [2, 3]},
    )
    evidence = search(
        index, "How can I repair the spacecraft?", AuthoredEmbedder(fixed_vector=[-1.0, -1.0])
    )
    assert evidence["status"] == "no_evidence"
    assert evidence["passages"] == []


def test_nested_bookmarks_preserve_chapter_and_separate_subsections(tmp_path: Path):
    from cited_retrieval import search

    path = tmp_path / "nested.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Rest supports recovery after demanding training.")
    page.insert_text((50, 200), "Gentle walking is another option for comfortable recovery.")
    doc.set_toc([[1, "Recovery", 1, 40], [2, "Sleep", 1, 80], [2, "Movement", 1, 180]])
    doc.save(path)
    doc.close()
    preview = inspect_source(path, 1, 1)
    index = tmp_path / "index.sqlite"
    ingest(
        path, index, 1, 1, AuthoredEmbedder(), approval={"sha256": preview["sha256"], "pages": [1]}
    )
    passages = search(index, "recovery", AuthoredEmbedder())["passages"]
    assert {p["location"]["chapter"] for p in passages} == {"Recovery"}
    assert {tuple(p["location"]["section_path"]) for p in passages} == {
        ("Recovery", "Sleep"),
        ("Recovery", "Movement"),
    }


def test_decorative_rules_and_page_footers_are_not_retrievable_evidence(tmp_path: Path):
    from cited_retrieval import search

    source = tmp_path / "decorations.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Rest and sleep support recovery after demanding training.")
    page.insert_text((50, 200), "--------Ornament--------")
    page.insert_text((50, 750), "Recovery 7")
    doc.save(source)
    doc.close()
    preview = inspect_source(source, 1, 1)
    index = tmp_path / "index.sqlite"
    summary = ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": preview["sha256"], "pages": [1]},
    )
    assert summary["passage_count"] == 1
    evidence = search(index, "ornament", AuthoredEmbedder(fixed_vector=[-1.0, -1.0]))
    assert evidence["status"] == "no_evidence"


def test_same_page_bookmark_boundaries_never_mix_chapters(tmp_path: Path):
    from cited_retrieval import search

    path = tmp_path / "boundaries.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (50, 100),
        "Rest supports recovery after demanding training.\nFood supplies energy for everyday physical activity.",
    )
    doc.set_toc([[1, "First", 1, 72], [1, "Second", 1, 105]])
    doc.save(path)
    doc.close()
    preview = inspect_source(path, 1, 1)
    index = tmp_path / "index.sqlite"
    ingest(
        path, index, 1, 1, AuthoredEmbedder(), approval={"sha256": preview["sha256"], "pages": [1]}
    )
    evidence = search(index, "recovery energy", AuthoredEmbedder())["passages"]
    assert len(evidence) == 2
    assert {p["location"]["chapter"] for p in evidence} == {"First", "Second"}
    assert all(
        not ("Rest supports" in p["text"] and "Food supplies" in p["text"]) for p in evidence
    )


@pytest.mark.parametrize("table", ["passages", "terms"])
def test_changed_corpus_artifact_is_rejected(source: Path, tmp_path: Path, table: str):
    import sqlite3

    from cited_retrieval import search

    index = tmp_path / "index.sqlite"
    preview = inspect_source(source, 2, 3)
    ingest(
        source,
        index,
        2,
        3,
        AuthoredEmbedder(),
        approval={"sha256": preview["sha256"], "pages": [2, 3]},
    )
    # Simulate a stale on-disk artifact, not an alternate ingestion seam.
    with sqlite3.connect(index) as db:
        if table == "passages":
            db.execute("UPDATE passages SET vector='[0.2,0.8]' WHERE rowid=1")
        else:
            db.execute("UPDATE terms SET text='orbital mechanics' WHERE rowid=1")
    with pytest.raises(ValueError, match="stale_corpus"):
        search(index, "orbital mechanics", AuthoredEmbedder())


def test_missing_source_is_explicitly_stale(source: Path, tmp_path: Path):
    from cited_retrieval import search

    index = tmp_path / "index.sqlite"
    preview = inspect_source(source, 2, 3)
    ingest(
        source,
        index,
        2,
        3,
        AuthoredEmbedder(),
        approval={"sha256": preview["sha256"], "pages": [2, 3]},
    )
    source.unlink()
    with pytest.raises(ValueError, match="stale_source"):
        search(index, "rest", AuthoredEmbedder())
