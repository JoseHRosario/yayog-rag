"""Source work is reusable; public trust and returned mutable values are not."""

import copy
import hashlib
import json
from pathlib import Path

import pymupdf
import pytest
from test_exercises import fixture
from test_programmes import schedule

from cited_retrieval import (
    draft_exercises,
    draft_programmes,
    inspect_exercises,
    inspect_programmes,
    review_programme,
)


def test_programme_inspection_reuses_real_rendering_without_caching_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, catalog, layout, _ = schedule(tmp_path)
    calls = []
    original = pymupdf.Page.get_pixmap

    def render(page: pymupdf.Page, *args: object, **kwargs: object) -> pymupdf.Pixmap:
        calls.append(1)
        return original(page, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_pixmap", render)
    draft = draft_programmes(source, catalog, layout, 1, 2)
    assert calls
    calls.clear()
    first = inspect_programmes(catalog)
    assert first == draft
    first["records"][0]["entries"][0]["exercise"] = "not source text"
    second = inspect_programmes(catalog)
    assert second["records"][0]["entries"][0]["exercise"] == "Cloud press"
    record = second["records"][0]
    assert not record["eligible"]
    review_programme(catalog, record["id"], record["revision"], "accepted")
    assert inspect_programmes(catalog)["records"][0]["eligible"]
    review_programme(catalog, record["id"], record["revision"], "rejected")
    assert not inspect_programmes(catalog)["records"][0]["eligible"]
    assert calls == [], "unchanged source cells must not be rendered again"


def test_exercise_inspection_reuses_real_text_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, catalog, _ = fixture(tmp_path)
    calls = []
    original = pymupdf.Page.get_text

    def extract(
        page: pymupdf.Page, *args: object, **kwargs: object
    ) -> str | list[tuple[object, ...]] | dict[str, object]:
        calls.append(1)
        return original(page, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", extract)
    assert inspect_exercises(catalog) == inspect_exercises(catalog)
    assert calls == [], "unchanged source regions must not be parsed again"


def test_changed_layout_and_source_do_not_reuse_previous_derivation(tmp_path: Path) -> None:
    source, catalog, layout, sel = schedule(tmp_path)
    first = draft_programmes(source, catalog, layout, 1, 2)["records"][0]
    changed = copy.deepcopy(layout)
    changed["records"][0]["entries"][0]["fields"]["exercise"] = [sel(2, 120, 170)]
    changed["records"][0]["entries"] = changed["records"][0]["entries"][:1]
    second = draft_programmes(source, catalog, changed, 1, 2)["records"][0]
    assert second["entries"][0]["exercise"] == "Stone curl"
    assert second["revision"] != first["revision"]
    source.write_bytes(source.read_bytes() + b"\n% changed source bytes")
    with pytest.raises(ValueError, match="stale_programmes"):
        inspect_programmes(catalog)


def test_same_source_different_range_cannot_reuse_out_of_range_cells(tmp_path: Path) -> None:
    source, catalog, layout, _ = schedule(tmp_path)
    draft_programmes(source, catalog, layout, 1, 2)
    with pytest.raises(ValueError, match="invalid_programme_layout"):
        draft_programmes(source, tmp_path / "narrow.json", layout, 1, 1)


def test_cached_exercise_source_change_still_rejects(tmp_path: Path) -> None:
    source, catalog, _ = fixture(tmp_path)
    inspect_exercises(catalog)
    source.write_bytes(source.read_bytes() + b"\n% changed source bytes")
    with pytest.raises(ValueError, match="stale_exercises"):
        inspect_exercises(catalog)


def test_old_derivations_are_evicted_instead_of_retained_without_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, catalog, layout, _ = schedule(tmp_path)
    draft_programmes(source, catalog, layout, 1, 2)
    for number in range(20):
        other = copy.deepcopy(layout)
        other["records"][0]["aliases"] = [f"Authored alternate {number}"]
        draft_programmes(source, tmp_path / "other.json", other, 1, 2)
    calls = []
    original = pymupdf.Page.get_pixmap

    def render(page: pymupdf.Page, *args: object, **kwargs: object) -> pymupdf.Pixmap:
        calls.append(1)
        return original(page, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_pixmap", render)
    assert inspect_programmes(catalog)["records"][0]["programme"] == "Cloud programme"
    assert calls, "an evicted derivation must be re-rendered, not retained indefinitely"


@pytest.mark.parametrize("change", ["review", "crop"])
def test_warm_derivation_never_bypasses_recovery_artifact_checks(
    tmp_path: Path, change: str
) -> None:
    from authored_ocr import AuthoredOCR
    from test_recovery import damaged_source

    from cited_retrieval import recover, review

    source = damaged_source(tmp_path)
    bundle = tmp_path / "recovery.json"
    region = recover(source, bundle, 1, 1, AuthoredOCR())["regions"][0]
    layout = {
        "records": [
            {
                "id": "authored",
                "coverage": "complete",
                "boundaries": [{"pdf_page": 1, "bbox": [40, 40, 550, 170]}],
                "fields": {
                    "name": [{"pdf_page": 1, "bbox": [40, 40, 550, 90], "quote": "Rest"}],
                    "instructions": [{"pdf_page": 1, "bbox": [40, 100, 550, 170]}],
                },
            }
        ]
    }
    catalog = tmp_path / "exercises.json"
    pending = draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)["records"][0]
    assert pending["coverage"] == "partial"
    review(bundle, region["id"], region["revision"], "accepted")
    accepted = draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)["records"][0]
    assert accepted["coverage"] == "complete"
    inspect_exercises(catalog)
    if change == "review":
        review(bundle, region["id"], region["revision"], "rejected")
    else:
        (bundle.parent / region["crop"]).write_bytes(b"authored corruption")
    with pytest.raises(ValueError, match="stale_exercises|stale_recovery"):
        inspect_exercises(catalog)


def test_source_changes_during_real_rendering_are_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, catalog, layout, _ = schedule(tmp_path)
    original = pymupdf.Page.get_pixmap
    changed = False

    def render(page: pymupdf.Page, *args: object, **kwargs: object) -> pymupdf.Pixmap:
        nonlocal changed
        pixmap = original(page, *args, **kwargs)
        if not changed:
            changed = True
            source.write_bytes(source.read_bytes() + b"\n% concurrent authored change")
        return pixmap

    monkeypatch.setattr(pymupdf.Page, "get_pixmap", render)
    with pytest.raises(ValueError, match="invalid_programme_layout|source_changed"):
        draft_programmes(source, catalog, layout, 1, 2)
    assert not catalog.exists()


def test_new_source_bytes_at_same_path_are_actually_rederived(tmp_path: Path) -> None:
    source, catalog, layout, _ = schedule(tmp_path)
    draft_programmes(source, catalog, layout, 1, 2)
    with pymupdf.open(source) as doc:
        doc.delete_page(0)
        page = doc.new_page(pno=0)
        page.insert_text((40, 50), "Rain programme")
        page.insert_text((40, 80), "Summary weeks 1-4")
        data = doc.tobytes()
    source.write_bytes(data)
    updated = draft_programmes(source, catalog, layout, 1, 2)
    assert updated["records"][0]["programme"] == "Rain programme"
    assert updated["records"][0]["state"] == "pending"


@pytest.mark.parametrize("kind", ["programme", "exercise"])
def test_warm_derivation_still_checks_actual_sealed_record_revision(
    tmp_path: Path, kind: str
) -> None:
    if kind == "programme":
        source, catalog, layout, _ = schedule(tmp_path)
        draft_programmes(source, catalog, layout, 1, 2)
        inspect = inspect_programmes
        field = "programme"
    else:
        _, catalog, _ = fixture(tmp_path)
        inspect = inspect_exercises
        field = "name"
    inspect(catalog)
    value = json.loads(catalog.read_text())
    value.pop("seal")
    record = value["records"][0]
    record[field] = "Authored text absent from source"

    def digest(item: object) -> str:
        serialized = json.dumps(item, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    # Simulate a self-consistent edited catalog, not just a broken outer checksum.
    record["revision"] = digest(
        [
            value["binding"],
            {
                k: v
                for k, v in record.items()
                if k not in {"revision", "state", "history", "eligible"}
            },
        ]
    )
    value["seal"] = digest(value)
    catalog.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="stale_programmes|stale_exercises"):
        inspect(catalog)
