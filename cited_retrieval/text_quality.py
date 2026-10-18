"""Conservative raw-line quality checks shared by native import and recovery."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pymupdf

BASE14 = {
    "Helvetica",
    "Helvetica-Bold",
    "Helvetica-Oblique",
    "Helvetica-BoldOblique",
    "Times-Roman",
    "Times-Bold",
    "Times-Italic",
    "Times-BoldItalic",
    "Courier",
    "Courier-Bold",
    "Courier-Oblique",
    "Courier-BoldOblique",
}


def invalid_characters(text: str) -> bool:
    return any(
        (unicodedata.category(c) in {"Cc", "Co", "Cs", "Cn"} and c not in "\n\r\t") or c == "\ufffd"
        for c in text
    )


def unmapped_fonts(page: Any) -> set[str]:
    result = set()
    for font in page.get_fonts(full=True):
        simple = font[2] in {"Type1", "TrueType"} and font[5] in {
            "WinAnsiEncoding",
            "MacRomanEncoding",
        }
        base = (
            font[3] in BASE14
            and font[1] == "n/a"
            and font[5]
            in {
                "WinAnsiEncoding",
                "MacRomanEncoding",
                "",
            }
        )
        if page.parent.xref_get_key(font[0], "ToUnicode")[0] == "null" and not (simple or base):
            result.add(font[3].split("+")[-1].replace(" ", "").casefold())
    return result


def raw_line_text(line: dict[str, Any], trace: list[dict[str, Any]]) -> str:
    x0, y0, x1, y1 = line["bbox"]
    return "".join(
        chr(c[0])
        for span in trace
        for c in span["chars"]
        if x0 - 0.1 <= c[2][0] <= x1 + 0.1 and y0 - 0.1 <= c[2][1] <= y1 + 0.1
    )


def line_has_unmapped_glyphs(page: Any, line: dict[str, Any]) -> bool:
    # Family names are not font-resource identities: a healthy simple font and an
    # unmapped CID font can share a family. Disable CID fallback in this region.
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_USE_CID_FOR_UNKNOWN_UNICODE
    blocks = page.get_text("dict", flags=flags, clip=pymupdf.Rect(line["bbox"]))["blocks"]  # type: ignore[no-untyped-call]
    return any(
        "\ufffd" in span["text"]
        for block in blocks
        for item in block.get("lines", [])
        for span in item["spans"]
    )


def line_reason(
    line: dict[str, Any], missing: set[str], raw: str = "", *, unmapped: bool = False
) -> str | None:
    if missing and unmapped:
        return "missing_unicode_map"
    text = "".join(s["text"] for s in line["spans"])
    if invalid_characters(text) or any(
        (unicodedata.category(c) in {"Cc", "Co", "Cs", "Cn"} and c not in "\n\r\t") or c == "\ufffd"
        for c in raw
    ):
        return "invalid_unicode"
    # Conservative visible garbage signal; short headings/quantities alone are not damage.
    letters = [c for c in text if c.isalpha()]
    if len(text.strip()) >= 12 and (
        (len(text.split()) >= 5 and len(letters) / len(text) < 0.35)
        or (len(letters) >= 12 and len(set(letters)) <= 2)
        or re.search(r"([^\W\d_])\1{5,}", text) is not None
    ):
        return "suspicious_native"
    return None
