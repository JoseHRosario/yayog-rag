"""Geometric ownership of explicit ROIs, never inferred from native whitespace."""

from typing import Any

import pymupdf

_rect: Any = pymupdf.Rect


def retained_regions(lines: list[dict[str, Any]], quote: str | None) -> set[str]:
    """Witness complete source-line ranges retained by a validated selector.

    Offsets are in the actual extraction, before explicit, lossless declared joins.
    Text elsewhere (including identical native text) cannot witness an OCR line.
    Call only after validating the quote's unique occurrence and all joins.
    """
    source = "\n".join(line["text"] for line in lines)
    start = 0 if quote is None else source.index(quote)
    end = len(source) if quote is None else start + len(quote)
    retained: set[str] = set()
    offset = 0
    for line in lines:
        length = len(line["text"])
        if line.get("region_id") and length and start <= offset and offset + length <= end:
            retained.add(line["region_id"])
        offset += length + 1
    return retained


def owns_boundary(
    page: Any, number: int, box: Any, recovered: dict[str, Any], selected: set[str]
) -> bool:
    """Native blocks own text, but only explicit image ROIs can own image pixels.

    Keep the global outside-owned-blocks gap: even an image-free strip between
    separately selected crops is not implicitly selected. Image occurrences include
    inline images (xref zero); neither xrefs nor native bounding boxes prove ownership.
    """
    blocks = [b for b in recovered["blocks"] if b["pdf_page"] == number]
    if not any(_rect(b["bbox"]).contains(box) for b in blocks):
        return False
    accepted = {r["id"] for r in recovered["regions"] if r["state"] == "accepted"}
    image_blocks = [b for b in blocks if b.get("image_region_id")]
    if any(
        box.intersects(_rect(b["bbox"])) and b["image_region_id"] not in selected & accepted
        for b in image_blocks
    ):
        return False
    rois = [_rect(b["bbox"]) for b in image_blocks if b["image_region_id"] in selected & accepted]
    for image in page.get_image_info():
        overlap = box & _rect(image["bbox"])
        remaining = [] if overlap.is_empty else [overlap]
        for roi in rois:
            pieces = []
            for area in remaining:
                cut = area & roi
                if cut.is_empty:
                    pieces.append(area)
                    continue
                pieces.extend(
                    r
                    for r in (
                        _rect(area.x0, area.y0, area.x1, cut.y0),
                        _rect(area.x0, cut.y1, area.x1, area.y1),
                        _rect(area.x0, cut.y0, cut.x0, cut.y1),
                        _rect(cut.x1, cut.y0, area.x1, cut.y1),
                    )
                    if not r.is_empty
                )
            remaining = pieces
        if remaining:
            return False
    return True
