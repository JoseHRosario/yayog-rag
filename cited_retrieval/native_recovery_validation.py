"""Reconstruct legacy native evidence from the PDF, not its editable outer seal."""

import copy
from pathlib import Path
from typing import Any

import pymupdf

from .core import _digest, _json
from .derivation_cache import memoize_derivation
from .policy import OCR_CROP_PADDING, OCR_DPI
from .text_quality import line_has_unmapped_glyphs, line_reason, raw_line_text, unmapped_fonts

_open: Any = pymupdf.open
_rect: Any = pymupdf.Rect


@memoize_derivation(lambda: (1, pymupdf.VersionBind, OCR_DPI, OCR_CROP_PADDING))
def source_graph(
    source: Path, start: int, end: int, requested: tuple[str, ...]
) -> list[dict[str, Any]]:
    blocks = []
    with _open(source) as doc:
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(doc):
            raise ValueError("stale_recovery")
        for number in range(start, end + 1):
            page = doc[number - 1]
            missing, trace = unmapped_fonts(page), page.get_texttrace()
            native = [b for b in page.get_text("dict", sort=True)["blocks"] if b["type"] == 0]
            for block in native:
                lines = []
                for ordinal, line in enumerate(block["lines"]):
                    bbox = list(line["bbox"])
                    original = "".join(s["text"] for s in line["spans"])
                    raw = raw_line_text(line, trace)
                    rid = _digest(_json([number, block["number"], ordinal, bbox]).encode())
                    reason = line_reason(
                        line, missing, raw, unmapped=line_has_unmapped_glyphs(page, line)
                    )
                    item: dict[str, Any] = {
                        "text": original,
                        "raw_text": raw,
                        "bbox": bbox,
                        "id": rid,
                        "reason": reason,
                    }
                    if reason or rid in requested:
                        clip = (
                            _rect(bbox)
                            + (
                                -OCR_CROP_PADDING,
                                -OCR_CROP_PADDING,
                                OCR_CROP_PADDING,
                                OCR_CROP_PADDING,
                            )
                        ) & page.rect
                        pixmap = page.get_pixmap(dpi=OCR_DPI, clip=clip, alpha=False)
                        item.update(
                            crop_bbox=list(clip),
                            crop_sha256=_digest(pixmap.tobytes("png")),
                            blank=bool(pixmap.samples) and min(pixmap.samples) == 255,
                        )
                    lines.append(item)
                blocks.append({"pdf_page": number, "bbox": list(block["bbox"]), "lines": lines})
            if not native:
                blocks.append({"pdf_page": number, "gap": "no_native_text_regions"})
    return blocks


def validate_native(value: dict[str, Any]) -> None:
    regions = {r["id"]: r for r in value["regions"]}
    if len(regions) != len(value["regions"]):
        raise ValueError("stale_recovery")
    section = value["section"]
    graph = source_graph(
        Path(value["source_path"]), section["start"], section["end"], tuple(sorted(regions))
    )
    blocks, gaps, used = [], [], set()
    for block in graph:
        if "gap" in block:
            gaps.append({"pdf_page": block["pdf_page"], "reason": block["gap"]})
            continue
        lines = []
        for native in block["lines"]:
            item = {k: native[k] for k in ("text", "raw_text", "bbox")}
            item["region_id"] = None
            region = regions.get(native["id"])
            if region is not None:
                expected = {
                    "pdf_page": block["pdf_page"],
                    "bbox": native["bbox"],
                    "original_text": native["text"],
                    "raw_text": native["raw_text"],
                    "crop_bbox": native["crop_bbox"],
                    "crop_sha256": native["crop_sha256"],
                    "settings": value["settings"],
                }
                if any(_json(region[k]) != _json(v) for k, v in expected.items()):
                    raise ValueError("stale_recovery")
                # Classification reasons can originate in older operator pipelines.
                # They confer no authority: geometry, glyph trace and crop must match.
                item["region_id"] = native["id"]
                used.add(native["id"])
            elif native["reason"]:
                if not native.get("blank"):
                    raise ValueError("stale_recovery")
                item.update(original_text=native["text"], text="", rendered_blank=True)
            lines.append(item)
        blocks.append({"pdf_page": block["pdf_page"], "bbox": block["bbox"], "lines": lines})
    actual = value["blocks"]
    # Some genuine legacy producers store the linked OCR display text in lines.
    # Normalize only that exact, independently revision-bound output to native text.
    normalized = copy.deepcopy(actual)
    for block in normalized:
        for line in block["lines"]:
            rid = line["region_id"]
            if rid and rid in regions and line["text"] == regions[rid]["ocr_text"]:
                line["text"] = regions[rid]["original_text"]
    if (
        used != set(regions)
        or _json(normalized) != _json(blocks)
        or _json(value["gaps"]) != _json(gaps)
    ):
        raise ValueError("stale_recovery")
