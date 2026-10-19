"""Public authored-source regressions for selection-bound raster ownership."""

from pathlib import Path
from typing import Any

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR

from cited_retrieval import draft_excerpts, ingest, recover, review, review_excerpt, search
from cited_retrieval.exercises import draft_exercises, review_exercise


@pytest.mark.parametrize("kind", ["excerpt", "exercise"])
@pytest.mark.parametrize(
    "selection",
    [
        "oversized",
        "omitted",
        "partial",
        "quote",
        "partial_quote",
        "duplicate",
        "full",
        "full_quote",
        "joined",
        "pending",
        "child_full",
        "child_quote",
    ],
)
def test_only_retained_roi_evidence_owns_warning(tmp_path: Path, kind: str, selection: str) -> None:
    source, bundle, catalog = (tmp_path / n for n in ("source.pdf", "bundle.json", "catalog.json"))
    warning = "STOP IF PAIN DEVELOPS"
    ocr = "STOP IF PAIN DEVE-\nLOPS" if selection == "joined" else warning
    with pymupdf.open() as raster:
        label = raster.new_page(width=180, height=8)
        label.insert_text((0, 6), warning, fontsize=6)
        image = label.get_pixmap().tobytes("png")
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=300)
        page.insert_text((20, 70), "Rest widget")
        page.insert_text((20, 82), "for twelve seconds before continuing safely.")
        if selection == "duplicate":
            page.insert_text((20, 110), warning)
        page.insert_image(pymupdf.Rect(100, 59, 180, 65), stream=image, keep_proportion=False)
        doc.save(source)
    roi = {"pdf_page": 1, "bbox": [100, 40 if selection == "oversized" else 59, 180, 66]}
    recovered = recover(
        source, bundle, 1, 1, AuthoredOCR(ocr), image_regions=[{"id": "warning", **roi}]
    )
    region = recovered["regions"][0]
    if selection != "pending":
        review(bundle, "warning", region["revision"], "accepted")
    native = next(b for b in recovered["blocks"] if not b.get("image_region_id"))
    boundary = {"pdf_page": 1, "bbox": native["bbox"]}
    selectors: list[dict[str, Any]] = [boundary]
    boundaries = [boundary]
    first = {"pdf_page": 1, "bbox": native["lines"][0]["bbox"]}
    second = {"pdf_page": 1, "bbox": native["lines"][1]["bbox"]}
    if selection == "omitted":
        selectors = [first, second]
    elif selection == "partial":
        selectors = [{**boundary, "bbox": [20, boundary["bbox"][1], 150, boundary["bbox"][3]]}]
    elif selection == "quote":
        selectors = [{**boundary, "quote": "Rest widget"}]
    elif selection == "partial_quote":
        selectors = [{**boundary, "quote": "STOP IF PAIN"}]
    elif selection == "duplicate":
        duplicate = {
            "pdf_page": 1,
            "bbox": next(
                b["bbox"]
                for b in recovered["blocks"]
                if any(line["text"] == warning for line in b["lines"])
            ),
        }
        boundaries.append(duplicate)
        selectors = [{**boundary, "quote": "Rest widget"}, duplicate]
    elif selection in {"full", "full_quote", "joined", "pending", "child_full", "child_quote"}:
        # Exact ROI and ordered native selectors, including overlapping boundaries.
        selectors = [
            first,
            second,
            roi,
        ]
        boundaries = [boundary, *selectors]
        if selection == "joined":
            selectors[-1] = {**roi, "joins": [ocr]}
        elif selection in {"full_quote", "child_full"}:
            selectors[-1] = {**roi, "quote": warning}
        elif selection == "child_quote":
            selectors[-1] = {**roi, "quote": "STOP IF PAIN"}
    spec: dict[str, Any] = {"id": "widget", "coverage": "complete", "boundaries": boundaries}
    if kind == "excerpt":
        spec["selectors"] = selectors
        draft = draft_excerpts(source, catalog, {"records": [spec]}, 1, 1, recovery=bundle)
    else:
        spec["fields"] = {"name": [{**boundary, "quote": "Rest widget"}], "instructions": selectors}
        if selection.startswith("child_"):
            spec["fields"]["instructions"] = [first, second]
            spec["variations"] = [
                {
                    "id": "child",
                    "relation": "alternative",
                    "boundaries": [roi],
                    "fields": {"instructions": [selectors[-1]]},
                }
            ]
        draft = draft_exercises(source, catalog, {"records": [spec]}, 1, 1, recovery=bundle)
    record = draft["records"][0]
    complete = selection in {"full", "full_quote", "joined", "child_full"}
    assert record["coverage"] == ("complete" if complete else "partial")
    assert ("unresolved_recovery" in record["gaps"]) != complete
    reviewer = review_excerpt if kind == "excerpt" else review_exercise
    reviewed = reviewer(catalog, "widget", record["revision"], "accepted")
    assert reviewed["records"][0]["eligible"] == complete
    kwargs = {"excerpts" if kind == "excerpt" else "exercises": catalog}
    if complete:
        index = tmp_path / "index.sqlite"
        result = ingest(
            source,
            index,
            1,
            1,
            AuthoredEmbedder(),
            approval={"sha256": recovered["source_sha256"], "pages": [1]},
            recovery=bundle,
            **kwargs,
        )
        assert result["pages"][0]["status"] == "partial"
        assert result["recovery"]["gaps"]
        passages = search(index, "widget", AuthoredEmbedder())["passages"]
        assert passages
        assert all(warning in passage["text"] for passage in passages)
    else:
        with pytest.raises(ValueError, match="no_eligible_passages"):
            ingest(
                source,
                tmp_path / "index.sqlite",
                1,
                1,
                AuthoredEmbedder(),
                approval={"sha256": recovered["source_sha256"], "pages": [1]},
                recovery=bundle,
                **kwargs,
            )
