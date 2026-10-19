from pathlib import Path

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR

from cited_retrieval import ingest, recover, report, review, search


def test_contained_import_keeps_whole_bundle_bound(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        for number in range(3):
            page = doc.new_page()
            page.insert_text((50, 70), f"Rest and sleep support recovery on day {number + 1}.")
            font = page.insert_font(fontname="damaged", fontbuffer=pymupdf.Font("cjk").buffer)
            page.insert_text((50, 140), "Hold for 12 seconds.", fontname="damaged")
            doc.xref_set_key(font, "ToUnicode", "null")
        doc.save(source)
    bundle, index = tmp_path / "bundle.json", tmp_path / "index.sqlite"
    result = recover(source, bundle, 1, 3, AuthoredOCR())
    selected = next(r for r in result["regions"] if r["pdf_page"] == 2)
    review(bundle, selected["id"], selected["revision"], "accepted")
    frozen = bundle.read_bytes()
    imported = ingest(
        source,
        index,
        2,
        2,
        AuthoredEmbedder(),
        approval={"sha256": result["source_sha256"], "pages": [2]},
        recovery=bundle,
    )
    assert bundle.read_bytes() == frozen
    assert imported["recovery"]["unresolved_regions"] == 0
    assert len(imported["recovery"]["outside_selection"]["regions"]) == 2
    assert {
        p["location"]["pdf_page"]
        for p in search(index, "recovery seconds", AuthoredEmbedder())["passages"]
    } == {2}
    assert report(index)["section"] == {"start": 2, "end": 2}
    outside = next(r for r in result["regions"] if r["pdf_page"] == 1)
    review(bundle, outside["id"], outside["revision"], "rejected")
    with pytest.raises(ValueError, match="stale_recovery"):
        report(index)
