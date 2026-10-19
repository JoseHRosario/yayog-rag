import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR

from cited_retrieval import ingest, inspect_review, recover, report, search


def seal(bundle: Path, value: dict[str, Any]) -> None:
    value.pop("seal", None)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    value["seal"] = hashlib.sha256(encoded.encode()).hexdigest()
    bundle.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "change", ["invent", "omit", "reorder", "duplicate", "text", "raw", "bbox", "gaps", "section"]
)
def test_resealed_legacy_graph_must_match_full_source(tmp_path: Path, change: str) -> None:
    source, bundle, index = (tmp_path / n for n in ("source.pdf", "bundle.json", "index.sqlite"))
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((30, 70), "Rest the widget for exactly twelve seconds before resuming.")
        page.insert_text((30, 150), "Stop immediately if the red warning lamp begins flashing.")
        doc.new_page()
        doc.save(source)
    result = recover(source, bundle, 1, 2, AuthoredOCR())
    approval = {"sha256": result["source_sha256"], "pages": [1]}
    ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, recovery=bundle)
    assert search(index, "widget seconds", AuthoredEmbedder())["passages"]
    value = json.loads(bundle.read_bytes())
    if change == "invent":
        value["blocks"].append(
            {
                "pdf_page": 2,
                "bbox": [10, 10, 400, 60],
                "lines": [
                    {
                        "text": "Invented instruction",
                        "raw_text": "",
                        "bbox": [10, 10, 400, 60],
                        "region_id": None,
                    }
                ],
            }
        )
    elif change == "omit":
        value["blocks"].pop()
    elif change == "reorder":
        value["blocks"].reverse()
    elif change == "duplicate":
        value["blocks"].append(value["blocks"][0])
    elif change in {"text", "raw"}:
        value["blocks"][0]["lines"][0]["raw_text" if change == "raw" else "text"] = (
            "Fabricated recovery prose."
        )
    elif change == "bbox":
        value["blocks"][0]["lines"][0]["bbox"][0] += 1
    elif change == "gaps":
        value["gaps"] = []
    else:
        value["section"] = {"start": 1, "end": 1}
    seal(bundle, value)
    with pytest.raises(ValueError, match="stale_recovery"):
        inspect_review(bundle)
    with pytest.raises(ValueError, match="stale_recovery"):
        ingest(
            source,
            tmp_path / "forged.sqlite",
            1,
            1,
            AuthoredEmbedder(),
            approval=approval,
            recovery=bundle,
        )
    with pytest.raises(ValueError, match="stale_recovery"):
        report(index)
    with pytest.raises(ValueError, match="stale_recovery"):
        search(index, "widget", AuthoredEmbedder())


def test_blank_source_cannot_gain_native_prose_by_resealing(tmp_path: Path) -> None:
    source, bundle = tmp_path / "blank.pdf", tmp_path / "bundle.json"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.new_page()
        doc.save(source)
    recover(source, bundle, 1, 2, AuthoredOCR())
    value = json.loads(bundle.read_bytes())
    value["blocks"] = [
        {
            "pdf_page": 1,
            "bbox": [10, 10, 400, 60],
            "lines": [
                {
                    "text": "Rest the invented widget for exactly twelve seconds and then stop immediately.",
                    "raw_text": "",
                    "bbox": [10, 10, 400, 60],
                    "region_id": None,
                }
            ],
        }
    ]
    value["gaps"] = []
    seal(bundle, value)
    with pytest.raises(ValueError, match="stale_recovery"):
        inspect_review(bundle)


@pytest.mark.parametrize(
    "change",
    [
        "original_text",
        "raw_text",
        "crop_bbox",
        "crop_sha256",
        "source_sha256",
        "omit_region",
        "display",
        "custom_reason",
    ],
)
def test_legacy_regions_remain_bound_to_native_glyphs_and_rendered_crop(
    tmp_path: Path, change: str
) -> None:
    from test_recovery import damaged_source

    from cited_retrieval import review

    source = damaged_source(tmp_path)
    bundle = tmp_path / "bundle.json"
    recovered = recover(source, bundle, 1, 1, AuthoredOCR())
    original = recovered["regions"][0]
    review(bundle, original["id"], original["revision"], "accepted")
    value = json.loads(bundle.read_bytes())
    region = value["regions"][0]
    if change == "omit_region":
        value["regions"] = []
        for block in value["blocks"]:
            for line in block["lines"]:
                line["region_id"] = None
    elif change == "display":
        for block in value["blocks"]:
            for line in block["lines"]:
                if line["region_id"]:
                    line["text"] = region["ocr_text"]
    else:
        key = "reason" if change == "custom_reason" else change
        region[key] = [0, 0, 10, 10] if key == "crop_bbox" else "authored modification"
        revision_data = {
            k: v for k, v in region.items() if k not in {"revision", "state", "history"}
        }
        region["revision"] = hashlib.sha256(
            json.dumps(
                revision_data, sort_keys=True, ensure_ascii=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    seal(bundle, value)
    if change in {"display", "custom_reason"}:
        assert inspect_review(bundle)["regions"][0]["state"] == "accepted"
        result = ingest(
            source,
            tmp_path / "index.sqlite",
            1,
            1,
            AuthoredEmbedder(),
            approval={"sha256": recovered["source_sha256"], "pages": [1]},
            recovery=bundle,
        )
        assert result["passage_count"] == 2
    else:
        with pytest.raises(ValueError, match="stale_recovery"):
            inspect_review(bundle)
