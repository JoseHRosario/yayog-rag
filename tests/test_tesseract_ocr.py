"""Authored executable exercises the public recovery boundary without system OCR."""

from pathlib import Path

import pytest
from authored_ocr import AuthoredOCR
from authored_tessdata import authored_tessdata
from test_recovery import damaged_source

from cited_retrieval import recover
from cited_retrieval.tesseract_ocr import TesseractOCR


def test_implicit_unfingerprinted_language_data_is_not_allowed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    executable = tmp_path / "ocr"
    executable.write_text("#!/bin/sh\nprintf 'authored engine 1\\n'\n")
    executable.chmod(0o700)
    with pytest.raises(ValueError, match="^ocr_unavailable$"):
        TesseractOCR(executable=executable)


def test_local_subprocess_adapter_preserves_output_and_records_settings(tmp_path: Path):
    executable = tmp_path / "ocr"
    executable.write_text(
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'authored engine 1\\n'; else printf 'Hold for 12 seconds. Stop if pain develops.\\n'; fi\n"
    )
    executable.chmod(0o700)
    engine = TesseractOCR(executable=executable, timeout=5, tessdata=authored_tessdata(tmp_path))
    result = recover(damaged_source(tmp_path), tmp_path / "bundle.json", 1, 1, engine)
    region = result["regions"][0]
    assert region["ocr_text"] == "Hold for 12 seconds. Stop if pain develops.\n"
    assert region["engine"]["version"] == "authored engine 1"
    assert region["engine"]["psm"] == 7
    assert region["engine"]["dpi"] == 400
    assert len(region["engine"]["executable_sha256"]) == 64


def test_changed_language_artifact_invalidates_acceptance_even_when_ocr_text_is_identical(
    tmp_path: Path,
):
    from cited_retrieval import review

    executable = tmp_path / "ocr"
    executable.write_text(
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'authored engine 1\\n'; else printf 'Hold for 12 seconds. Stop if pain develops.\\n'; fi\n"
    )
    executable.chmod(0o700)
    data = authored_tessdata(tmp_path)
    engine = TesseractOCR(executable=executable, tessdata=data)
    source, bundle = damaged_source(tmp_path), tmp_path / "bundle.json"
    before = recover(source, bundle, 1, 1, engine)["regions"][0]
    review(bundle, before["id"], before["revision"], "accepted")
    (data / "eng.traineddata").write_bytes(b"authored language data version two")
    after = recover(source, bundle, 1, 1, engine)["regions"][0]
    assert after["ocr_text"] == before["ocr_text"]
    assert after["state"] == "pending"
    assert after["revision"] != before["revision"]


def test_engine_failure_preserves_prior_bundle_without_exposing_provider_errors(tmp_path: Path):
    source = damaged_source(tmp_path)
    bundle = tmp_path / "bundle.json"
    recover(source, bundle, 1, 1, AuthoredOCR())
    before = bundle.read_bytes()
    executable = tmp_path / "ocr"
    executable.write_text(
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'authored engine 1\\n'; else printf 'PRIVATE STDERR' >&2; exit 1; fi\n"
    )
    executable.chmod(0o700)
    with pytest.raises(ValueError, match="^ocr_failed$"):
        recover(
            source,
            bundle,
            1,
            1,
            TesseractOCR(executable=executable, tessdata=authored_tessdata(tmp_path)),
        )
    assert bundle.read_bytes() == before
