from pathlib import Path

import pymupdf
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR

from cited_retrieval import ingest, recover, review, search


def test_short_accepted_warnings_and_quantities_remain_retrievable(tmp_path: Path):
    for text in ["Stop if dizzy.", "12 seconds."]:
        case = tmp_path / ("warning" if text.startswith("Stop") else "duration")
        case.mkdir()
        source = case / "source.pdf"
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 70), "Rest and sleep support recovery after training.")
            font = page.insert_font(fontname="damaged", fontbuffer=pymupdf.Font("cjk").buffer)
            page.insert_text((50, 140), text, fontname="damaged")
            doc.xref_set_key(font, "ToUnicode", "null")
            doc.save(source)
        bundle, index = case / "bundle.json", case / "index.sqlite"
        result = recover(source, bundle, 1, 1, AuthoredOCR(text))
        region = result["regions"][0]
        review(bundle, region["id"], region["revision"], "accepted")
        ingest(
            source,
            index,
            1,
            1,
            AuthoredEmbedder(),
            approval={"sha256": result["source_sha256"], "pages": [1]},
            recovery=bundle,
        )
        evidence = search(index, text, AuthoredEmbedder())
        assert any(
            p["text"] == text and p["review_status"] == "accepted" for p in evidence["passages"]
        )


def test_filtered_native_warning_is_explicitly_reported_as_a_coverage_gap(tmp_path: Path):
    from cited_retrieval import report

    source = tmp_path / "native-warning.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 70), "Rest and sleep support recovery after training.")
        page.insert_text((50, 140), "Stop if dizzy.")
        doc.save(source)
    bundle, index = tmp_path / "bundle.json", tmp_path / "index.sqlite"
    result = recover(source, bundle, 1, 1, AuthoredOCR())
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": result["source_sha256"], "pages": [1]},
        recovery=bundle,
    )
    current = report(index)
    assert current["pages"][0]["status"] == "partial"
    assert any(
        gap["reason"] == "excluded_non_prose" and gap["text"] == "Stop if dizzy."
        for gap in current["recovery"]["gaps"]
    )


def test_mixed_paragraph_is_withheld_until_all_damaged_lines_are_accepted(tmp_path: Path):
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 70), "Rest and sleep support recovery after training.")
        page.insert_text((50, 140), "Hold this position for 12 seconds.")
        font = page.insert_font(fontname="damaged", fontbuffer=pymupdf.Font("cjk").buffer)
        page.insert_text(
            (50, 153), "Stop if pain develops during this movement.", fontname="damaged"
        )
        doc.xref_set_key(font, "ToUnicode", "null")
        doc.save(source)
    bundle, index = tmp_path / "bundle.json", tmp_path / "index.sqlite"
    result = recover(
        source, bundle, 1, 1, AuthoredOCR("Stop if pain develops during this movement.")
    )
    assert len(result["blocks"][1]["lines"]) == 2
    approval = {"sha256": result["source_sha256"], "pages": [1]}
    pending = ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    assert pending["pages"][0]["status"] == "partial"
    assert pending["recovery"]["gaps"][0]["reason"] == "unresolved_block"
    assert pending["recovery"]["gaps"][0]["pdf_page"] == 1
    assert pending["recovery"]["gaps"][0]["regions"][0]["state"] == "pending"
    assert all(
        "12 seconds" not in p["text"]
        for p in search(index, "seconds", AuthoredEmbedder())["passages"]
    )
    region = result["regions"][0]
    review(bundle, region["id"], region["revision"], "accepted")
    ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    recovered = next(
        p
        for p in search(index, "seconds pain", AuthoredEmbedder())["passages"]
        if "12 seconds" in p["text"]
    )
    assert (
        recovered["text"]
        == "Hold this position for 12 seconds.\nStop if pain develops during this movement."
    )
