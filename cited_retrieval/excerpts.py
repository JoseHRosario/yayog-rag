"""Individually reviewed, source-selected prose; no semantic identity or replacement text."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pymupdf

from .cells import native_cell
from .core import _digest, _hash_file, _json, inspect_source
from .policy import CHUNK_CHARS
from .recovery import inspect_review
from .recovery_selection import recovery_selection
from .source_selection import private_artifact, publish_json, recovery_artifacts, selection_box
from .text_quality import line_has_unmapped_glyphs, line_reason, raw_line_text, unmapped_fonts

VERSION = {"version": 1, "feature": "reviewed-source-excerpts", "renderer": pymupdf.VersionBind}
INDEX_VERSION = 1
_open: Any = pymupdf.open
_rect: Any = pymupdf.Rect


def protected_artifacts(binding: dict[str, Any]) -> set[Path]:
    paths = {Path(binding["source_path"]).resolve()}
    if binding.get("layout_path"):
        paths.add(Path(binding["layout_path"]).resolve())
    if binding["recovery_path"]:
        recovery = private_artifact(binding["recovery_path"])
        paths.update(recovery_artifacts(recovery, inspect_review(recovery)))
    return paths


def _revision(record: dict[str, Any], binding: dict[str, Any]) -> str:
    return _digest(
        _json(
            [
                binding,
                {
                    k: v
                    for k, v in record.items()
                    if k not in {"revision", "state", "history", "eligible"}
                },
            ]
        ).encode()
    )


def _derive(binding: dict[str, Any], layout: dict[str, Any]) -> list[dict[str, Any]]:
    start, end = binding["section"]["start"], binding["section"]["end"]
    source = Path(binding["source_path"])
    preview = inspect_source(source, start, end)
    recovered = None
    if binding["recovery_path"]:
        recovered = recovery_selection(
            inspect_review(binding["recovery_path"]), binding["source_sha256"], start, end
        )
    ids: set[str] = set()
    records = []
    if (
        set(layout) != {"records"}
        or not isinstance(layout["records"], list)
        or not layout["records"]
    ):
        raise ValueError("invalid_excerpt_layout")
    with _open(source) as doc:
        for spec in layout["records"]:
            if set(spec) - {"id", "boundaries", "selectors", "coverage", "gaps"}:
                raise ValueError("invalid_excerpt_layout")
            rid = spec["id"]
            if not isinstance(rid, str) or not rid.strip() or rid in ids:
                raise ValueError("invalid_excerpt_layout")
            ids.add(rid)
            boundaries, selectors = spec["boundaries"], spec["selectors"]
            if (
                not isinstance(boundaries, list)
                or not boundaries
                or not isinstance(selectors, list)
                or not selectors
            ):
                raise ValueError("invalid_excerpt_layout")
            gaps = list(spec.get("gaps", []))
            selected_regions: set[str] = set()
            raster_boundaries = []
            if spec.get("coverage", "partial") != "complete":
                gaps.append("unverified_completeness")
            for boundary in boundaries:
                if set(boundary) != {"pdf_page", "bbox"}:
                    raise ValueError("invalid_excerpt_layout")
                box = selection_box(boundary, doc, start, end)
                page = doc[boundary["pdf_page"] - 1]
                if recovered is None:
                    for line in native_cell(page, box):
                        item = {"bbox": line["bbox"], "spans": [{"text": line["text"]}]}
                        if line_reason(
                            item,
                            unmapped_fonts(page),
                            raw_line_text(item, page.get_texttrace()),
                            unmapped=line_has_unmapped_glyphs(page, item),
                        ):
                            gaps.append("unresolved_native_text")
                else:
                    for gap in recovered["gaps"]:
                        if gap["pdf_page"] != boundary["pdf_page"]:
                            continue
                        if gap.get("scope") == "outside_owned_blocks":
                            raster_boundaries.append((boundary, len(gaps)))
                            gaps.append("unresolved_recovery")
                            continue
                        if "bbox" not in gap or box.intersects(_rect(gap["bbox"])):
                            gaps.append("unresolved_recovery")
            segments = []
            for selector in selectors:
                if set(selector) - {"pdf_page", "bbox", "quote", "joins"}:
                    raise ValueError("invalid_excerpt_layout")
                box = selection_box(selector, doc, start, end)
                number = selector["pdf_page"]
                if not any(
                    b["pdf_page"] == number and _rect(b["bbox"]).contains(box) for b in boundaries
                ):
                    raise ValueError("invalid_excerpt_layout")
                page = doc[number - 1]
                lines = native_cell(page, box)
                if recovered is not None:
                    regions = {r["id"]: r for r in recovered["regions"]}
                    owned = [
                        line
                        for b in recovered["blocks"]
                        if b["pdf_page"] == number
                        for line in b["lines"]
                        if line["region_id"] or line.get("rendered_blank")
                    ]
                    lines = native_cell(page, box, excluded=[_rect(line["bbox"]) for line in owned])
                    lines.extend(
                        [
                            {
                                **line,
                                "text": regions[line["region_id"]]["ocr_text"]
                                if line["region_id"]
                                else line["text"],
                            }
                            for line in owned
                            if line["region_id"] and box.contains(_rect(line["bbox"]))
                        ]
                    )
                    lines.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
                text = "\n".join(line["text"] for line in lines)
                original = text
                quote = selector.get("quote")
                if quote is not None:
                    if not isinstance(quote, str) or not quote.strip() or text.count(quote) != 1:
                        raise ValueError("invalid_excerpt_layout")
                    text = quote
                for join in selector.get("joins", []):
                    if not isinstance(join, str) or "-\n" not in join or text.count(join) != 1:
                        raise ValueError("invalid_excerpt_layout")
                    text = text.replace(join, join.replace("-\n", ""), 1)
                from .raster_ownership import retained_regions

                selected_regions.update(retained_regions(lines, quote))
                if not text.strip():
                    gaps.append("empty_selector")
                ancestors: list[dict[str, Any]] = []
                for bookmark in sorted(preview["bookmarks"], key=lambda b: (b["pdf_page"], b["y"])):
                    if (
                        bookmark["pdf_page"] < number
                        or bookmark["pdf_page"] == number
                        and bookmark["y"] <= box.y1
                    ):
                        ancestors = [a for a in ancestors if a["level"] < bookmark["level"]] + [
                            bookmark
                        ]
                titles = [a["title"] for a in ancestors]
                segments.append(
                    {
                        "text": text,
                        "original_text": original,
                        "selector": selector,
                        "lines": lines,
                        "source_id": binding["source_sha256"],
                        "location": {
                            "pdf_page": number,
                            "bbox": list(box),
                            "printed_label": page.get_label() or None,
                            "chapter": titles[0] if titles else None,
                            "section_path": titles,
                        },
                    }
                )
            from .raster_ownership import owns_boundary

            for boundary, gap_index in reversed(raster_boundaries):
                assert recovered is not None
                if owns_boundary(
                    doc[boundary["pdf_page"] - 1],
                    boundary["pdf_page"],
                    _rect(boundary["bbox"]),
                    recovered,
                    selected_regions,
                ):
                    del gaps[gap_index]
            records.append(
                {
                    "id": rid,
                    "boundaries": boundaries,
                    "segments": segments,
                    "text": "\n".join(s["text"] for s in segments),
                    "coverage": "partial" if gaps else "complete",
                    "gaps": gaps,
                }
            )
    return records


def _check(binding: dict[str, Any]) -> None:
    private_artifact(binding["source_path"])
    if binding.get("layout_path"):
        private_artifact(binding["layout_path"])
    if (
        _json(binding["settings"]) != _json(VERSION)
        or _hash_file(Path(binding["source_path"])) != binding["source_sha256"]
    ):
        raise ValueError("stale_excerpts")
    if (
        binding.get("layout_path")
        and _hash_file(Path(binding["layout_path"])) != binding["layout_sha256"]
    ):
        raise ValueError("stale_excerpts")
    if binding["recovery_path"]:
        inspect_review(binding["recovery_path"])
        if _hash_file(Path(binding["recovery_path"])) != binding["recovery_sha256"]:
            raise ValueError("stale_excerpts")


def _read(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_bytes())
    seal = value.pop("seal")
    if value["binding"]["settings"].get("feature") != VERSION["feature"]:
        raise ValueError("stale_excerpts")
    if seal != _digest(_json(value).encode()) or any(
        r["revision"] != _revision(r, value["binding"])
        or r["state"] not in {"pending", "accepted", "rejected"}
        for r in value["records"]
    ):
        raise ValueError("stale_excerpts")
    return value


def draft_excerpts(
    pdf: str | Path,
    catalog: str | Path,
    layout: dict[str, Any] | str | Path,
    start: int,
    end: int,
    *,
    recovery: str | Path | None = None,
) -> dict[str, Any]:
    source, target = private_artifact(pdf), private_artifact(catalog)
    preview = inspect_source(source, start, end)
    layout_path = private_artifact(layout) if isinstance(layout, (str, Path)) else None
    layout_bytes = layout_path.read_bytes() if layout_path else None
    specs: Any = json.loads(layout_bytes) if layout_bytes is not None else layout
    binding = {
        "source_path": str(source),
        "source_sha256": preview["sha256"],
        "section": preview["section"],
        "settings": VERSION.copy(),
        "layout_path": str(layout_path) if layout_path else None,
        "layout_sha256": _digest(layout_bytes) if layout_bytes is not None else None,
        "recovery_path": str(private_artifact(recovery)) if recovery is not None else None,
        "recovery_sha256": _hash_file(Path(recovery)) if recovery is not None else None,
    }
    if target in protected_artifacts(binding):
        raise ValueError("unsafe_review_path")
    previous_bytes = target.read_bytes() if target.exists() else None
    old = _read(target) if previous_bytes is not None else {"records": []}
    previous = {r["id"]: r for r in old["records"]}
    try:
        records = _derive(binding, specs)
    except (KeyError, TypeError, ValueError, IndexError):
        raise ValueError("invalid_excerpt_layout") from None
    finally:
        _check(binding)
    for record in records:
        record["revision"] = _revision(record, binding)
        prior = previous.get(record["id"], {})
        record["state"] = (
            prior.get("state", "pending")
            if prior.get("revision") == record["revision"]
            else "pending"
        )
        record["history"] = list(prior.get("history", []))
        if prior.get("revision") != record["revision"]:
            record["history"].append(
                {
                    "revision": record["revision"],
                    "state": "pending",
                    "reason": "content_changed" if prior else "created",
                }
            )
    _check(binding)
    publish_json(target, {"binding": binding, "layout": specs, "records": records}, previous_bytes)
    return inspect_excerpts(target)


def inspect_excerpts(catalog: str | Path) -> dict[str, Any]:
    path = private_artifact(catalog)
    try:
        original = path.read_bytes()
        value = _read(path)
        binding = value["binding"]
        if path in protected_artifacts(binding):
            raise ValueError("stale_excerpts")
        _check(binding)
        derived = _derive(binding, value["layout"])
        if len(derived) != len(value["records"]) or any(
            _revision(r, binding) != old["revision"]
            for r, old in zip(derived, value["records"], strict=True)
        ):
            raise ValueError("stale_excerpts")
        _check(binding)
        if original != path.read_bytes():
            raise ValueError("stale_excerpts")
        for record in value["records"]:
            record["eligible"] = record["state"] == "accepted" and record["coverage"] == "complete"
        return value
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        raise ValueError("stale_excerpts") from None


def review_excerpt(
    catalog: str | Path, record_id: str, revision: str, state: str
) -> dict[str, Any]:
    path = private_artifact(catalog)
    previous = path.read_bytes()
    value = inspect_excerpts(path)
    if state not in {"pending", "accepted", "rejected"}:
        raise ValueError("invalid_review_state")
    record = next((r for r in value["records"] if r["id"] == record_id), None)
    if record is None or record["revision"] != revision:
        raise ValueError("review_revision_mismatch")
    record["state"] = state
    record["history"].append({"revision": revision, "state": state})
    for item in value["records"]:
        item.pop("eligible", None)
    _check(value["binding"])
    publish_json(path, value, previous)
    return inspect_excerpts(path)


def validate_excerpts(binding: dict[str, Any]) -> None:
    if type(binding.get("index_version")) is not int or binding["index_version"] != INDEX_VERSION:
        raise ValueError("stale_settings")
    catalog = inspect_excerpts(binding["catalog_path"])
    if _hash_file(Path(binding["catalog_path"])) != binding["catalog_sha256"] or _json(
        catalog["records"]
    ) != _json(binding["records"]):
        raise ValueError("stale_excerpts")


def excerpt_passages(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    passages = []
    for ordinal, record in enumerate(catalog["records"]):
        if not record["eligible"]:
            continue
        for number, window in enumerate(
            textwrap.wrap(
                record["text"],
                width=CHUNK_CHARS,
                replace_whitespace=False,
                expand_tabs=False,
                drop_whitespace=False,
                break_long_words=False,
                break_on_hyphens=False,
            )
        ):
            passages.append(
                {
                    "source_id": catalog["binding"]["source_sha256"],
                    "passage_id": _digest(_json([record["revision"], number, window]).encode()),
                    "text": record["text"],
                    "retrieval_text": window,
                    "location": {**record["segments"][0]["location"], "ordinal": ordinal},
                    "excerpt": record,
                    "eligible": True,
                    "review_status": "accepted",
                    "provenance": {
                        "excerpt_revision": record["revision"],
                        "chunk": number,
                        **VERSION,
                    },
                }
            )
    return passages
