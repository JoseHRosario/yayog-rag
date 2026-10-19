"""Authored PDFs exercise blank exclusions through public catalog/recovery seams."""

from pathlib import Path
from typing import Any

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR

from cited_retrieval import (
    draft_exercises,
    ingest,
    inspect_exercises,
    inspect_review,
    recover,
    report,
    review,
    review_exercise,
    search,
)


def blank_fixture(
    tmp_path: Path, *, visible: bool = False, warning: bool = False
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    source, bundle, catalog = (tmp_path / n for n in ("source.pdf", "bundle.json", "catalog.json"))
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 60), "Cloud press")
        page.insert_text((50, 100), "Lift slowly and keep the wrist steady.")
        font = page.insert_font(fontname="damaged", fontbuffer=pymupdf.Font("cjk").buffer)
        page.insert_text(
            (50, 114), "X", fontname="damaged", color=(0, 0, 0) if visible else (1, 1, 1)
        )
        doc.xref_set_key(font, "ToUnicode", "null")
        bad = page.get_text("dict")["blocks"][-1]["lines"][-1]["bbox"]
        # The original rectangle is white; only the padded OCR crop sees this ink.
        page.draw_rect(
            pymupdf.Rect(bad[2] + 0.4, bad[1] + 2, bad[2] + 0.8, bad[3] - 2),
            color=None,
            fill=(0, 0, 0),
        )
        if warning:
            page.insert_text((50, 128), "Stop if pain develops.", fontname="damaged")
        doc.save(source)
    recovered = recover(source, bundle, 1, 1, AuthoredOCR("oe"))
    region = recovered["regions"][0]
    assert any(
        any(line["region_id"] == region["id"] for line in block["lines"])
        and any("Lift slowly" in line["text"] for line in block["lines"])
        for block in recovered["blocks"]
    )
    layout = {
        "records": [
            {
                "id": "cloud",
                "coverage": "complete",
                "boundaries": [{"pdf_page": 1, "bbox": [40, 30, 500, 160]}],
                "fields": {
                    "name": [{"pdf_page": 1, "bbox": [45, 40, 450, 65]}],
                    "instructions": [{"pdf_page": 1, "bbox": [45, 85, 450, 104]}],
                },
                "blank_source_exclusions": [
                    {"region_id": region["id"], "revision": region["revision"]}
                ],
            }
        ]
    }
    return source, bundle, catalog, layout, region


@pytest.mark.parametrize("recovery_state", ["pending", "rejected"])
def test_verified_original_white_region_unlocks_body_without_accepting_fringe(
    tmp_path: Path,
    recovery_state: str,
) -> None:
    source, bundle, catalog, layout, region = blank_fixture(tmp_path)
    if recovery_state == "rejected":
        review(bundle, region["id"], region["revision"], "rejected")
    result = draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)
    record = result["records"][0]
    assert record["coverage"] == "complete"
    assert record["eligible"] is False
    proof = record["blank_source_exclusions"][0]
    assert proof["region_id"] == region["id"]
    assert proof["revision"] == region["revision"]
    assert proof["bbox"] == region["bbox"]
    assert proof["source_sha256"] == result["binding"]["source_sha256"]
    assert proof["pdf_page"] == 1
    assert proof["renderer"]["dpi"] == 400
    assert len(proof["render_sha256"]) == 64
    accepted = review_exercise(catalog, "cloud", record["revision"], "accepted")
    assert accepted["records"][0]["eligible"] is True
    assert (
        accepted["records"][0]["instructions"][0]["text"]
        == "Lift slowly and keep the wrist steady."
    )
    assert inspect_review(bundle)["regions"][0]["state"] == recovery_state
    assert inspect_review(bundle)["regions"][0]["ocr_text"] == "oe"
    index = tmp_path / "exercises.sqlite"
    approval = {"sha256": result["binding"]["source_sha256"], "pages": [1]}
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval=approval,
        recovery=bundle,
        exercises=catalog,
    )
    passages = search(index, "Cloud press", AuthoredEmbedder())["passages"]
    assert len(passages) == 1
    assert passages[0]["exercise"]["blank_source_exclusions"] == [proof]
    assert "oe" not in passages[0]["text"]
    assert "Lift slowly and keep the wrist steady." in passages[0]["text"]
    report(index)
    # These catalog-local proofs must not make the same mixed block eligible prose.
    prose = tmp_path / "prose.sqlite"
    with pytest.raises(ValueError, match="no_eligible_passages"):
        ingest(source, prose, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    assert not prose.exists()


@pytest.mark.parametrize(
    "change",
    [
        "bogus",
        "stale",
        "malformed",
        "extra_key",
        "duplicate",
        "selected",
        "outside",
        "visible",
        "no_recovery",
    ],
)
def test_invalid_exclusions_fail_closed_without_replacing_catalog(
    tmp_path: Path, change: str
) -> None:
    source, bundle, catalog, layout, _ = blank_fixture(tmp_path, visible=change == "visible")
    parent = layout["records"][0]
    exclusions = parent.pop("blank_source_exclusions")
    draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)
    before = catalog.read_bytes()
    parent["blank_source_exclusions"] = exclusions
    if change == "bogus":
        exclusions[0]["region_id"] = "unknown"
    elif change == "stale":
        exclusions[0]["revision"] = "old"
    elif change == "malformed":
        parent["blank_source_exclusions"] = "operator says blank"
    elif change == "extra_key":
        exclusions[0]["bbox"] = [0, 0, 1, 1]
    elif change == "duplicate":
        exclusions.append(exclusions[0].copy())
    elif change == "selected":
        parent["fields"]["instructions"][0]["bbox"][3] = 140
    elif change == "outside":
        parent["boundaries"][0]["bbox"][3] = 105
    with pytest.raises(ValueError, match="invalid_blank_source_exclusion"):
        draft_exercises(
            source, catalog, layout, 1, 1, recovery=None if change == "no_recovery" else bundle
        )
    assert catalog.read_bytes() == before
    assert inspect_exercises(catalog)["records"][0]["eligible"] is False


def test_an_unlisted_unknown_warning_still_quarantines_whole_parent(tmp_path: Path) -> None:
    source, bundle, catalog, layout, _ = blank_fixture(tmp_path, warning=True)
    result = draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)
    assert len(inspect_review(bundle)["regions"]) == 2
    assert any(len(gap.get("regions", [])) == 2 for gap in inspect_review(bundle)["gaps"])
    record = result["records"][0]
    assert record["coverage"] == "partial"
    assert "unresolved_recovery" in record["gaps"]
    assert (
        review_exercise(catalog, "cloud", record["revision"], "accepted")["records"][0]["eligible"]
        is False
    )


def test_even_declared_unknown_warning_cannot_be_excluded(tmp_path: Path) -> None:
    source, bundle, catalog, layout, _ = blank_fixture(tmp_path, warning=True)
    warning = inspect_review(bundle)["regions"][1]
    layout["records"][0]["blank_source_exclusions"].append(
        {"region_id": warning["id"], "revision": warning["revision"]}
    )
    with pytest.raises(ValueError, match="invalid_blank_source_exclusion"):
        draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)


def test_generic_page_gap_remains_even_with_valid_blank_proof(tmp_path: Path) -> None:
    source, bundle, catalog, layout, _ = blank_fixture(tmp_path)
    with pymupdf.open(source) as doc:
        doc.new_page()
        doc.saveIncr()
    current = recover(source, bundle, 1, 2, AuthoredOCR("oe"))["regions"][0]
    parent = layout["records"][0]
    parent["blank_source_exclusions"] = [
        {"region_id": current["id"], "revision": current["revision"]}
    ]
    parent["boundaries"].append({"pdf_page": 2, "bbox": [40, 30, 500, 160]})
    result = draft_exercises(source, catalog, layout, 1, 2, recovery=bundle)
    assert result["records"][0]["coverage"] == "partial"
    assert result["records"][0]["gaps"] == ["unresolved_recovery"]


def test_opt_in_removal_restores_legacy_quarantine_and_exact_revision(tmp_path: Path) -> None:
    source, bundle, catalog, layout, _ = blank_fixture(tmp_path)
    parent = layout["records"][0]
    exclusions = parent.pop("blank_source_exclusions")
    old = draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)["records"][0]
    assert old["coverage"] == "partial"
    assert "blank_source_exclusions" not in old
    parent["blank_source_exclusions"] = exclusions
    new = draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)["records"][0]
    assert new["coverage"] == "complete"
    assert new["revision"] != old["revision"]
    review_exercise(catalog, "cloud", new["revision"], "accepted")
    del parent["blank_source_exclusions"]
    restored = draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)
    assert restored["binding"]["settings"]["version"] == 1
    assert restored["records"][0]["revision"] == old["revision"]
    assert restored["records"][0]["state"] == "pending"
    assert "blank_source_exclusions" not in restored["records"][0]


def test_recovery_and_source_changes_invalidate_blank_proofs(tmp_path: Path) -> None:
    source, bundle, catalog, layout, _ = blank_fixture(tmp_path)
    record = draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)["records"][0]
    review_exercise(catalog, "cloud", record["revision"], "accepted")
    recover(source, bundle, 1, 1, AuthoredOCR("different fringe"))
    with pytest.raises(ValueError, match="stale_exercises"):
        inspect_exercises(catalog)
    with pytest.raises(ValueError, match="invalid_blank_source_exclusion"):
        draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)
    source.write_bytes(source.read_bytes() + b"\n% changed source\n")
    with pytest.raises(ValueError, match="stale_recovery"):
        draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)


@pytest.mark.parametrize(
    "change", ["valid_child", "outside_child", "child_selects", "quoted_parent_selects"]
)
def test_exclusions_respect_child_ownership_and_all_selected_lines(
    tmp_path: Path, change: str
) -> None:
    source, bundle, catalog, layout, _ = blank_fixture(tmp_path)
    parent = layout["records"][0]
    child = {
        "id": "wall",
        "relation": "easier",
        "boundaries": [{"pdf_page": 1, "bbox": [40, 80, 500, 160]}],
        "fields": {"instructions": [{"pdf_page": 1, "bbox": [45, 85, 450, 104]}]},
        "blank_source_exclusions": parent.pop("blank_source_exclusions"),
    }
    parent["variations"] = [child]
    if change == "outside_child":
        child["boundaries"][0]["bbox"][3] = 105
    elif change == "child_selects":
        child["fields"]["instructions"][0]["bbox"][3] = 140
    elif change == "quoted_parent_selects":
        selector = parent["fields"]["instructions"][0]
        selector["bbox"][3] = 140
        selector["quote"] = "Lift slowly and keep the wrist steady."
    if change == "valid_child":
        record = draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)["records"][0]
        assert record["coverage"] == "complete"
        assert len(record["blank_source_exclusions"]) == 1
    else:
        with pytest.raises(ValueError, match="invalid_blank_source_exclusion"):
            draft_exercises(source, catalog, layout, 1, 1, recovery=bundle)
