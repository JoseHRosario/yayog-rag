"""Explicit cell geometry, not automatic table/grid interpretation."""

from __future__ import annotations

from typing import Any

import pymupdf

from .core import _digest

_rect: Any = pymupdf.Rect


def native_cell(page: Any, box: Any, *, excluded: list[Any] | None = None) -> list[dict[str, Any]]:
    """Select whole glyph boxes even when a PDF line spans neighboring columns."""
    lines = []
    for block in page.get_text("rawdict", sort=True)["blocks"]:
        for line in block.get("lines", []):
            chars = [
                c
                for span in line["spans"]
                for c in span["chars"]
                if box.contains(_rect(c["bbox"]))
                and not any(region.intersects(_rect(c["bbox"])) for region in excluded or [])
            ]
            text = "".join(c["c"] for c in chars)
            if text.strip():
                lines.append(
                    {
                        "text": text,
                        "bbox": [
                            min(c["bbox"][0] for c in chars),
                            min(c["bbox"][1] for c in chars),
                            max(c["bbox"][2] for c in chars),
                            max(c["bbox"][3] for c in chars),
                        ],
                    }
                )
    return lines


def render_digest(page: Any, box: Any) -> str:
    """Bind a review to the actual rendered crop, without publishing the image."""
    return _digest(page.get_pixmap(clip=box, alpha=False).tobytes("png"))
