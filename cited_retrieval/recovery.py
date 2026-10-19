"""Private, content-bound targeted recovery bundles; never automatic approval."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf

from .core import _digest, _hash_file, _json, inspect_source
from .ocr import OCR
from .policy import OCR_CROP_PADDING, OCR_DPI
from .text_quality import line_has_unmapped_glyphs, line_reason, raw_line_text, unmapped_fonts

RECOVERY_SETTINGS = {
    "version": 2,
    "renderer": pymupdf.VersionBind,
    "dpi": OCR_DPI,
    "padding": OCR_CROP_PADDING,
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".review-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(_json(value))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _revision(region: dict[str, Any]) -> str:
    return _digest(
        _json(
            {k: v for k, v in region.items() if k not in {"revision", "state", "history"}}
        ).encode()
    )


def _summary(bundle: dict[str, Any]) -> dict[str, Any]:
    regions = {r["id"]: r for r in bundle["regions"]}
    gaps = list(bundle["gaps"])
    for block in bundle["blocks"]:
        unresolved = [
            regions[line["region_id"]]
            for line in block["lines"]
            if line["region_id"] and regions[line["region_id"]]["state"] != "accepted"
        ]
        if unresolved:
            gaps.append(
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
    return {
        **bundle,
        "gaps": gaps,
        "unresolved_regions": sum(r["state"] != "accepted" for r in bundle["regions"]),
        "warnings": [
            "Compare all OCR quantities, units and warnings against the crop; no corrections are applied."
        ],
    }


def recover(
    pdf: str | Path,
    bundle: str | Path,
    start: int,
    end: int,
    engine: OCR,
    *,
    image_regions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recover damaged native lines, or exclusively explicit image-only ROIs when supplied."""
    if image_regions is not None:
        from .image_recovery import recover_images

        return recover_images(pdf, bundle, start, end, engine, image_regions)
    source, target = Path(pdf).resolve(), Path(bundle).resolve()
    if source == target:
        raise ValueError("unsafe_review_path")
    preview = inspect_source(source, start, end)
    old = json.loads(target.read_text()) if target.exists() else {}
    if old:
        seal = old.pop("seal", None)
        if _digest(_json(old).encode()) != seal or any(
            _revision(r) != r["revision"] for r in old["regions"]
        ):
            raise ValueError("stale_recovery")
    old_regions = {r["id"]: r for r in old.get("regions", [])}
    identity = dict(engine.identity)
    regions: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    target.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(source) as doc:  # type: ignore[no-untyped-call]
        for number in range(start, end + 1):
            page = doc[number - 1]
            missing = unmapped_fonts(page)
            trace = page.get_texttrace()
            text_blocks = [b for b in page.get_text("dict", sort=True)["blocks"] if b["type"] == 0]
            if not text_blocks:
                gaps.append({"pdf_page": number, "reason": "no_native_text_regions"})
            for block in text_blocks:
                lines = []
                for ordinal, line in enumerate(block["lines"]):
                    original = "".join(s["text"] for s in line["spans"])
                    raw = raw_line_text(line, trace)
                    item: dict[str, Any] = {
                        "text": original,
                        "raw_text": raw,
                        "bbox": list(line["bbox"]),
                        "region_id": None,
                    }
                    reason = line_reason(
                        line, missing, raw, unmapped=line_has_unmapped_glyphs(page, line)
                    )
                    if reason:
                        bbox = list(line["bbox"])
                        clip = (
                            pymupdf.Rect(bbox)  # type: ignore[no-untyped-call]
                            + (
                                -OCR_CROP_PADDING,
                                -OCR_CROP_PADDING,
                                OCR_CROP_PADDING,
                                OCR_CROP_PADDING,
                            )
                        ) & page.rect
                        pixmap = page.get_pixmap(dpi=OCR_DPI, clip=clip, alpha=False)
                        if pixmap.samples and min(pixmap.samples) == 255:
                            # A rendered-empty spacing glyph has no written instructions.
                            # Preserve its extraction for audit, but never OCR blank pixels.
                            item.update(original_text=original, text="", rendered_blank=True)
                            lines.append(item)
                            continue
                        image = pixmap.tobytes("png")
                        crop_hash = _digest(image)
                        crop = f"crops/{crop_hash}.png"
                        crop_path = target.parent / crop
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        if not crop_path.exists():
                            fd = os.open(crop_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                            with os.fdopen(fd, "wb") as stream:
                                stream.write(image)
                        elif _hash_file(crop_path) != crop_hash:
                            raise ValueError("stale_recovery")
                        rid = _digest(_json([number, block["number"], ordinal, bbox]).encode())
                        region = {
                            "id": rid,
                            "source_sha256": preview["sha256"],
                            "pdf_page": number,
                            "bbox": bbox,
                            "crop_bbox": list(clip),
                            "original_text": original,
                            "raw_text": raw,
                            "reason": reason,
                            "crop": crop,
                            "crop_sha256": crop_hash,
                            "ocr_text": engine.recognize(crop_path),
                            "engine": identity,
                            "settings": RECOVERY_SETTINGS.copy(),
                        }
                        region["revision"] = _revision(region)
                        previous = old_regions.get(rid)
                        region["state"] = "pending"
                        region["history"] = list(previous["history"]) if previous else []
                        if not previous:
                            region["history"].append(
                                {
                                    "state": "pending",
                                    "revision": region["revision"],
                                    "reason": "created",
                                    "at": datetime.now(UTC).isoformat(),
                                }
                            )
                        if previous and previous["revision"] == region["revision"]:
                            region["state"] = previous["state"]
                        elif previous:
                            region["history"].append(
                                {
                                    "state": "pending",
                                    "revision": region["revision"],
                                    "reason": "content_changed",
                                }
                            )
                        regions.append(region)
                        item["region_id"] = rid
                    lines.append(item)
                blocks.append({"pdf_page": number, "bbox": list(block["bbox"]), "lines": lines})
    if engine.identity != identity:
        raise ValueError("ocr_changed")
    if _hash_file(source) != preview["sha256"]:
        raise ValueError("source_changed")
    result = {
        "schema": 1,
        "source_path": str(source),
        "source_sha256": preview["sha256"],
        "section": preview["section"],
        "settings": RECOVERY_SETTINGS.copy(),
        "regions": regions,
        "blocks": blocks,
        "gaps": gaps,
    }
    result["seal"] = _digest(_json(result).encode())
    _atomic_json(target, result)
    return _summary(result)


def load_bundle(path: Path) -> dict[str, Any]:
    """Validate every referenced artifact, including review revision, on every use."""
    try:
        value: dict[str, Any] = json.loads(path.read_text())
        seal = value.pop("seal")
        if _digest(_json(value).encode()) != seal or type(value["schema"]) is not int:
            raise ValueError("stale_recovery")
        if value["schema"] == 2:
            from .image_recovery import validate_images

            validate_images(value, path)
        elif value["schema"] != 1 or value["settings"] != RECOVERY_SETTINGS:
            raise ValueError("stale_recovery")
        if _hash_file(Path(value["source_path"])) != value["source_sha256"]:
            raise ValueError("stale_recovery")
        for region in value["regions"]:
            crop = (path.parent / region["crop"]).resolve()
            if not crop.is_relative_to(path.parent.resolve()):
                raise ValueError("stale_recovery")
            if (
                _hash_file(crop) != region["crop_sha256"]
                or _revision(region) != region["revision"]
                or region["source_sha256"] != value["source_sha256"]
                or region["state"] not in {"pending", "accepted", "rejected"}
            ):
                raise ValueError("stale_recovery")
        if value["schema"] == 1:
            from .native_recovery_validation import validate_native

            validate_native(value)
        if _hash_file(Path(value["source_path"])) != value["source_sha256"]:
            raise ValueError("stale_recovery")
        value["seal"] = seal
        return value
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError("stale_recovery") from None


def inspect_review(bundle: str | Path) -> dict[str, Any]:
    """Private inspection includes original extraction, verbatim OCR and review history."""
    return _summary(load_bundle(Path(bundle).resolve()))


def review(bundle: str | Path, region_id: str, revision: str, decision: str) -> dict[str, Any]:
    """Explicit operator decision bound to a source/crop/OCR/settings revision."""
    target = Path(bundle).resolve()
    try:
        previous_bytes = target.read_bytes()
    except OSError:
        raise ValueError("stale_recovery") from None
    value = load_bundle(target)
    if decision not in {"accepted", "rejected", "pending"}:
        raise ValueError("invalid_review_state")
    region = next((r for r in value["regions"] if r["id"] == region_id), None)
    if region is None or region["revision"] != revision:
        raise ValueError("review_revision_mismatch")
    if decision == "accepted" and not region["ocr_text"].strip():
        raise ValueError("empty_recovery")
    region["state"] = decision
    region["history"].append(
        {"state": decision, "revision": revision, "at": datetime.now(UTC).isoformat()}
    )
    value.pop("seal")
    value["seal"] = _digest(_json(value).encode())
    if value["schema"] == 2:
        from .source_selection import publish_json

        value.pop("seal")
        publish_json(target, value, previous_bytes, "stale_recovery")
        return inspect_review(target)
    _atomic_json(target, value)
    return _summary(value)
