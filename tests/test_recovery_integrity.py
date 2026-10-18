import json
import sys
from pathlib import Path

import pytest
from authored_ocr import AuthoredOCR
from authored_tessdata import authored_tessdata
from test_recovery import damaged_source

from cited_retrieval import inspect_review, recover, review
from cited_retrieval.tesseract_ocr import TesseractOCR


def test_regeneration_cannot_retain_a_modified_unreviewed_state(tmp_path: Path):
    source, bundle = damaged_source(tmp_path), tmp_path / "bundle.json"
    recover(source, bundle, 1, 1, AuthoredOCR())
    value = json.loads(bundle.read_text())
    value["regions"][0]["state"] = "accepted"
    bundle.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="stale_recovery"):
        recover(source, bundle, 1, 1, AuthoredOCR())


def test_settings_changes_reset_acceptance_and_preserve_pending_history(tmp_path: Path):
    source, bundle = damaged_source(tmp_path), tmp_path / "bundle.json"
    original = recover(source, bundle, 1, 1, AuthoredOCR())
    region = original["regions"][0]
    assert region["history"][0]["state"] == "pending"
    review(bundle, region["id"], region["revision"], "accepted")
    engine = AuthoredOCR()
    engine.identity["version"] = "2"
    changed = recover(source, bundle, 1, 1, engine)["regions"][0]
    assert changed["state"] == "pending"
    assert [h["state"] for h in changed["history"]] == ["pending", "accepted", "pending"]
    assert inspect_review(bundle)["regions"][0]["revision"] != region["revision"]


def test_ocr_deadline_is_enforced_and_failure_keeps_existing_bundle(tmp_path: Path):
    source, bundle = damaged_source(tmp_path), tmp_path / "bundle.json"
    recover(source, bundle, 1, 1, AuthoredOCR())
    before = bundle.read_bytes()
    executable = tmp_path / "ocr"
    executable.write_text(
        f"#!{sys.executable}\nimport sys,time\nif sys.argv[1] == '--version': print('authored engine 1')\nelse: time.sleep(2)\n"
    )
    executable.chmod(0o700)
    with pytest.raises(ValueError, match="^ocr_failed$"):
        recover(
            source,
            bundle,
            1,
            1,
            TesseractOCR(executable=executable, timeout=0.3, tessdata=authored_tessdata(tmp_path)),
        )
    assert bundle.read_bytes() == before
