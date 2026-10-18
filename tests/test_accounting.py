from pathlib import Path

import pymupdf
from authored_embedder import AuthoredEmbedder

from cited_retrieval import ingest, inspect_source


def test_heading_only_selected_page_is_excluded_from_import_accounting(tmp_path: Path):
    path = tmp_path / "headings.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    title = "An authored heading with enough words"
    page.insert_text((50, 100), title)
    page = doc.new_page()
    page.insert_text((50, 100), "Rest and sleep support recovery after training.")
    doc.set_toc([[1, title, 1]])
    doc.save(path)
    doc.close()
    preview = inspect_source(path, 1, 2)
    result = ingest(
        path,
        tmp_path / "index.sqlite",
        1,
        2,
        AuthoredEmbedder(),
        approval={"sha256": preview["sha256"], "pages": [1, 2]},
    )
    assert result["pages"][0]["status"] == "excluded"
    assert result["pages"][0]["reason"] == "no_eligible_prose"
    assert result["pages"][1]["status"] == "imported"
