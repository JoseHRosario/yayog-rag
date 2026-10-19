"""Exact-revision, original-source pixel proofs for opt-in non-content exclusions."""

from __future__ import annotations

from typing import Any

import pymupdf

from .core import _digest

_rect: Any = pymupdf.Rect


def verify_blank_exclusions(
    doc: Any,
    spec: dict[str, Any],
    recovered: dict[str, Any] | None,
    fingerprint: str,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Prove declared original boxes white; exempt only generated whole-block gaps.

    The caller supplies a currently validated sealed recovery summary. OCR pixels and
    OCR text are deliberately not evidence of blankness. All selected lines, including
    child fields and selectors using quotes, remain subject to ordinary quarantine.
    """
    nodes = [spec, *spec.get("variations", [])]
    declarations = [(node, item) for node in nodes for item in _declarations(node)]
    if not declarations:
        return [], set()
    if recovered is None or recovered["source_sha256"] != fingerprint:
        raise ValueError("invalid_blank_source_exclusion")
    regions = {r["id"]: r for r in recovered["regions"]}
    proofs: list[dict[str, Any]] = []
    excluded: set[str] = set()
    renderer = {
        "name": "pymupdf",
        "version": pymupdf.VersionBind,
        "fitz": pymupdf.VersionFitz,
        "dpi": 400,
        "colorspace": "RGB",
        "alpha": False,
        "annots": True,
        "padding": 0,
    }
    for node, item in declarations:
        if (
            not isinstance(item, dict)
            or set(item) != {"region_id", "revision"}
            or not isinstance(item["region_id"], str)
            or not isinstance(item["revision"], str)
        ):
            raise ValueError("invalid_blank_source_exclusion")
        rid = item["region_id"]
        region = regions.get(rid)
        if (
            rid in excluded
            or region is None
            or region["revision"] != item["revision"]
            or region["source_sha256"] != fingerprint
        ):
            raise ValueError("invalid_blank_source_exclusion")
        number = region["pdf_page"]
        box = _rect(region["bbox"])
        boundaries = node.get("boundaries", spec["boundaries"])
        if (
            type(number) is not int
            or not 1 <= number <= len(doc)
            or box.is_empty
            or box.is_infinite
            or not doc[number - 1].rect.contains(box)
            or not _contained(number, box, spec["boundaries"])
            or not _contained(number, box, boundaries)
            or not recovered["section"]["start"] <= number <= recovered["section"]["end"]
        ):
            raise ValueError("invalid_blank_source_exclusion")
        matching = [
            line
            for block in recovered["blocks"]
            if block["pdf_page"] == number
            for line in block["lines"]
            if line["region_id"] == rid
        ]
        if not matching or any(line["bbox"] != region["bbox"] for line in matching):
            raise ValueError("invalid_blank_source_exclusion")
        for selected_node in nodes:
            for selectors in selected_node["fields"].values():
                for selector in selectors:
                    if selector["pdf_page"] == number and any(
                        _rect(selector["bbox"]).contains(_rect(line["bbox"])) for line in matching
                    ):
                        raise ValueError("invalid_blank_source_exclusion")
        pixmap = doc[number - 1].get_pixmap(
            dpi=400, clip=box, colorspace=pymupdf.csRGB, alpha=False, annots=True
        )
        if not pixmap.samples or min(pixmap.samples) != 255:
            raise ValueError("invalid_blank_source_exclusion")
        proofs.append(
            {
                "region_id": rid,
                "revision": region["revision"],
                "pdf_page": number,
                "bbox": list(box),
                "source_sha256": fingerprint,
                "renderer": renderer.copy(),
                "render_sha256": _digest(pixmap.tobytes("png")),
            }
        )
        excluded.add(rid)
    # _summary appends these gaps after the bundle's original generic/page gaps.
    # Reconstruct that exact suffix, never exempt a generic gap by reason alone.
    generated = []
    for block in recovered["blocks"]:
        unresolved = [
            regions[line["region_id"]]
            for line in block["lines"]
            if line["region_id"] and regions[line["region_id"]]["state"] != "accepted"
        ]
        if unresolved:
            generated.append(
                {
                    "pdf_page": block["pdf_page"],
                    "bbox": block["bbox"],
                    "reason": "unresolved_block",
                    "regions": [
                        {"id": r["id"], "revision": r["revision"], "state": r["state"]}
                        for r in unresolved
                    ],
                }
            )
    offset = len(recovered["gaps"]) - len(generated)
    if offset < 0 or recovered["gaps"][offset:] != generated:
        raise ValueError("invalid_blank_source_exclusion")
    ignored = {
        offset + i
        for i, gap in enumerate(generated)
        if all(r["id"] in excluded for r in gap["regions"])
    }
    return proofs, ignored


def _declarations(node: dict[str, Any]) -> list[Any]:
    value = node.get("blank_source_exclusions", [])
    if not isinstance(value, list):
        raise ValueError("invalid_blank_source_exclusion")  # noqa: TRY004 — public validation error
    return value


def _contained(number: int, box: Any, boundaries: list[dict[str, Any]]) -> bool:
    return any(b["pdf_page"] == number and _rect(b["bbox"]).contains(box) for b in boundaries)
