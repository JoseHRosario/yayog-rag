"""CLI wiring exercises real parsing and source workflows; only model/OCR are doubled."""

import json
from pathlib import Path

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_tessdata import authored_tessdata
from test_excerpts import fixture

from cited_retrieval.cli import main


def test_excerpt_import_and_search_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import cited_retrieval.local_embedder as boundary

    monkeypatch.setattr(boundary, "LocalEmbedder", lambda _: AuthoredEmbedder())
    source, catalog, layout = fixture(tmp_path)
    selector_path = tmp_path / "layout.json"
    selector_path.write_text(json.dumps(layout))
    assert (
        main(
            [
                "draft-excerpts",
                str(source),
                "--catalog",
                str(catalog),
                "--layout",
                str(selector_path),
                "--start",
                "1",
                "--end",
                "1",
            ]
        )
        == 0
    )
    draft = json.loads(capsys.readouterr().out)
    record = draft["records"][0]
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
    capsys.readouterr()
    index = tmp_path / "index.sqlite"
    assert (
        main(
            [
                "import",
                str(source),
                "--index",
                str(index),
                "--start",
                "1",
                "--end",
                "1",
                "--approve-sha256",
                draft["binding"]["source_sha256"],
                "--approve-pages",
                "1",
                "--model-dir",
                "unused",
                "--excerpts",
                str(catalog),
            ]
        )
        == 0
    )
    imported = json.loads(capsys.readouterr().out)
    assert imported["excerpts"]["records"][0]["id"] == "p1"
    assert main(["search", "widget rest", "--index", str(index), "--model-dir", "unused"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["passages"][0]["text"]
        == "Rest the widget for twelve seconds before resuming."
    )


def test_explicit_image_recovery_cli_preserves_actual_subprocess_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(source)
    executable = tmp_path / "ocr"
    executable.write_text(
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'authored engine 1\\n'; else printf 'Rest 12 seconds.\\n'; fi\n"
    )
    executable.chmod(0o700)
    selectors = tmp_path / "rois.json"
    selectors.write_text(
        json.dumps([{"id": "annotation", "pdf_page": 1, "bbox": [20, 20, 300, 60]}])
    )
    bundle = tmp_path / "recovery.json"
    assert (
        main(
            [
                "recover",
                str(source),
                "--bundle",
                str(bundle),
                "--start",
                "1",
                "--end",
                "1",
                "--image-regions",
                str(selectors),
                "--tesseract",
                str(executable),
                "--tessdata-dir",
                str(authored_tessdata(tmp_path)),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == 2
    assert result["regions"][0]["ocr_text"] == "Rest 12 seconds.\n"
    assert result["regions"][0]["state"] == "pending"
    assert result["regions"][0]["engine"]["executable_sha256"]
