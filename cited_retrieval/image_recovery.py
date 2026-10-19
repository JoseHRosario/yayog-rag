"""Explicit image-only OCR. Never detects images or replaces native glyphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf

from .core import _digest, _hash_file, _json, _quality, inspect_source
from .ocr import OCR
from .policy import OCR_DPI
from .recovery import _revision, _summary
from .source_selection import (
    private_artifact,
    publish_crop,
    publish_json,
    recovery_artifacts,
    selection_box,
)
from .text_quality import line_has_unmapped_glyphs, line_reason, raw_line_text, unmapped_fonts

IMAGE_SETTINGS = {
    "version": 3,
    "feature": "explicit-image-only-roi-v1",
    "renderer": pymupdf.VersionBind,
    "dpi": OCR_DPI,
    "padding": 0,
    "preprocessing": "none",
}
_open: Any = pymupdf.open
_rect: Any = pymupdf.Rect


def image_structure(
    source: Path, specs: list[dict[str, Any]], start: int, end: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(specs, list) or not specs:
        raise ValueError("invalid_image_regions")
    blocks: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    ids: set[str] = set()
    boxes: list[tuple[int, Any]] = []
    with _open(source) as doc:
        for spec in specs:
            if set(spec) != {"id", "pdf_page", "bbox"}:
                raise ValueError("invalid_image_regions")
            rid = spec["id"]
            if not isinstance(rid, str) or not rid.strip() or rid in ids:
                raise ValueError("invalid_image_regions")
            ids.add(rid)
            box = selection_box(spec, doc, start, end)
            number = spec["pdf_page"]
            if any(p == number and box.intersects(other) for p, other in boxes):
                raise ValueError("invalid_image_regions")
            # Any intersecting native glyph belongs to the native path, healthy or damaged.
            if any(
                box.intersects(_rect(c["bbox"]))
                for block in doc[number - 1].get_text("rawdict")["blocks"]
                for line in block.get("lines", [])
                for span in line["spans"]
                for c in span["chars"]
            ):
                raise ValueError("image_region_native_collision")
            boxes.append((number, box))
            blocks.append(
                {
                    "pdf_page": number,
                    "bbox": list(box),
                    "image_region_id": rid,
                    "lines": [{"text": "", "raw_text": "", "bbox": list(box), "region_id": rid}],
                }
            )
        for number in range(start, end + 1):
            page = doc[number - 1]
            missing, trace = unmapped_fonts(page), page.get_texttrace()
            for block in page.get_text("dict", sort=True)["blocks"]:
                if block["type"] != 0:
                    continue
                lines = [
                    {
                        "text": "".join(s["text"] for s in line["spans"]),
                        "raw_text": raw_line_text(line, trace),
                        "bbox": list(line["bbox"]),
                        "region_id": None,
                    }
                    for line in block["lines"]
                ]
                if any(
                    line_reason(
                        line,
                        missing,
                        raw_line_text(line, trace),
                        unmapped=line_has_unmapped_glyphs(page, line),
                    )
                    for line in block["lines"]
                ):
                    gaps.append(
                        {
                            "pdf_page": number,
                            "bbox": list(block["bbox"]),
                            "reason": "unresolved_native_block",
                        }
                    )
                elif _quality(page, page.get_text()) is None:
                    blocks.append({"pdf_page": number, "bbox": list(block["bbox"]), "lines": lines})
                else:
                    gaps.append(
                        {
                            "pdf_page": number,
                            "bbox": list(block["bbox"]),
                            "reason": "native_requires_separate_review",
                        }
                    )
            gaps.append(
                {
                    "pdf_page": number,
                    "reason": "unselected_image_areas",
                    "scope": "outside_owned_blocks",
                }
            )
    return blocks, gaps


def recover_images(
    pdf: str | Path,
    bundle: str | Path,
    start: int,
    end: int,
    engine: OCR,
    specs: list[dict[str, Any]],
) -> dict[str, Any]:
    source, target = private_artifact(pdf), private_artifact(bundle)
    if target == source:
        raise ValueError("unsafe_review_path")
    preview = inspect_source(source, start, end)
    try:
        blocks, gaps = image_structure(source, specs, start, end)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid_image_regions") from exc
    previous_bytes = target.read_bytes() if target.exists() else None
    old = json.loads(previous_bytes) if previous_bytes is not None else {}
    if old:
        seal = old.pop("seal", None)
        if seal != _digest(_json(old).encode()) or any(
            _revision(r) != r["revision"] for r in old["regions"]
        ):
            raise ValueError("stale_recovery")
        if target in recovery_artifacts(target, old) - {target}:
            raise ValueError("unsafe_review_path")
    previous = {r["id"]: r for r in old.get("regions", [])}
    identity = dict(engine.identity)
    if not identity or any(not identity.get(k) for k in ("engine", "version")):
        raise ValueError("invalid_ocr_settings")
    regions = []
    prepared = []
    for spec in specs:
        # Fresh document per crop avoids image-subsampling state between renders.
        with _open(source) as doc:
            image = (
                doc[spec["pdf_page"] - 1]
                .get_pixmap(dpi=OCR_DPI, clip=_rect(spec["bbox"]), alpha=False)
                .tobytes("png")
            )
        digest = _digest(image)
        crop = f"crops/{digest}.png"
        crop_path = private_artifact(target.parent / crop)
        protected = {source, target}
        if old:
            protected.update(recovery_artifacts(target, old) - {crop_path, target})
        if crop_path in protected:
            raise ValueError("unsafe_review_path")
        if crop_path.exists() and _hash_file(crop_path) != digest:
            raise ValueError("stale_recovery")
        prepared.append((spec, image, digest, crop, crop_path))
    for spec, image, digest, crop, crop_path in prepared:
        publish_crop(crop_path, image)
        region = {
            **spec,
            "source_sha256": preview["sha256"],
            "crop_bbox": spec["bbox"],
            "original_text": "",
            "raw_text": "",
            "reason": "explicit_image_text",
            "crop": crop,
            "crop_sha256": digest,
            "ocr_text": _recognize(engine, crop_path, source, preview["sha256"], identity),
            "engine": identity,
            "settings": IMAGE_SETTINGS.copy(),
        }
        region["revision"] = _revision(region)
        prior = previous.get(spec["id"], {})
        region["state"] = (
            prior.get("state", "pending")
            if prior.get("revision") == region["revision"]
            else "pending"
        )
        region["history"] = list(prior.get("history", []))
        if prior.get("revision") != region["revision"]:
            region["history"].append(
                {
                    "state": "pending",
                    "revision": region["revision"],
                    "reason": "content_changed" if prior else "created",
                }
            )
        regions.append(region)
    if engine.identity != identity:
        raise ValueError("ocr_changed")
    if _hash_file(source) != preview["sha256"]:
        raise ValueError("source_changed")
    result = {
        "schema": 2,
        "source_path": str(source),
        "source_sha256": preview["sha256"],
        "section": preview["section"],
        "settings": IMAGE_SETTINGS.copy(),
        "image_regions": specs,
        "regions": regions,
        "blocks": blocks,
        "gaps": gaps,
    }
    validate_images(result, target)
    publish_json(target, result, previous_bytes, "stale_recovery")
    result["seal"] = _digest(_json(result).encode())
    return _summary(result)


def validate_images(value: dict[str, Any], path: Path) -> None:
    private_artifact(path)
    private_artifact(value["source_path"])
    if _json(value["settings"]) != _json(IMAGE_SETTINGS):
        raise ValueError("stale_recovery")
    section = value["section"]
    inspect_source(value["source_path"], section["start"], section["end"])
    blocks, gaps = image_structure(
        Path(value["source_path"]), value["image_regions"], section["start"], section["end"]
    )
    if _json(blocks) != _json(value["blocks"]) or _json(gaps) != _json(value["gaps"]):
        raise ValueError("stale_recovery")
    if len(value["regions"]) != len(value["image_regions"]):
        raise ValueError("stale_recovery")
    for region, spec in zip(value["regions"], value["image_regions"], strict=True):
        if (
            any(_json(region[k]) != _json(v) for k, v in spec.items())
            or _json(region["settings"]) != _json(IMAGE_SETTINGS)
            or region["crop_bbox"] != spec["bbox"]
        ):
            raise ValueError("stale_recovery")
        crop = private_artifact(path.parent / region["crop"])
        if crop in {Path(value["source_path"]).resolve(), path.resolve()}:
            raise ValueError("stale_recovery")
        with _open(value["source_path"]) as doc:
            image = (
                doc[spec["pdf_page"] - 1]
                .get_pixmap(dpi=OCR_DPI, clip=_rect(spec["bbox"]), alpha=False)
                .tobytes("png")
            )
        if _digest(image) != region["crop_sha256"] or _hash_file(crop) != region["crop_sha256"]:
            raise ValueError("stale_recovery")


def _recognize(
    engine: OCR, crop: Path, source: Path, fingerprint: str, identity: dict[str, str]
) -> str:
    before = _hash_file(crop)
    try:
        text = engine.recognize(crop)
        if not isinstance(text, str):
            raise ValueError("ocr_failed")  # noqa: TRY004 - stable public OCR error
        return text
    finally:
        if _hash_file(source) != fingerprint:
            raise ValueError("source_changed")
        if _hash_file(crop) != before:
            raise ValueError("stale_recovery")
        if engine.identity != identity:
            raise ValueError("ocr_changed")
