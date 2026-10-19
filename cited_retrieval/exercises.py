"""Operator-delimited, source-derived exercise records and exact-revision review.

No field accepts authored replacement instructions. All text is selected from source lines.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import pymupdf

from .core import _digest, _hash_file, _json, inspect_source
from .derivation_cache import memoize_derivation
from .policy import CHUNK_CHARS
from .recovery import _atomic_json, inspect_review
from .text_quality import line_has_unmapped_glyphs, line_reason, raw_line_text, unmapped_fonts

FIELDS = {
    "name",
    "category",
    "muscles",
    "difficulty",
    "instructions",
    "warnings",
    "references",
    "chapter",
}
SCALARS = {"name", "category", "muscles", "difficulty", "chapter"}
# Index-only policies do not invalidate catalog extraction/reviews.
INSTRUCTION_ANCHOR_VERSION = 1
MATCHING_WINDOW_VERSION = 1
VERSION = {"version": 1, "pymupdf": pymupdf.VersionBind, "retrieval_chunk_chars": CHUNK_CHARS}
_rect_factory: Any = pymupdf.Rect
_open_document: Any = pymupdf.open


def _rect(selector: dict[str, Any], doc: Any, start: int, end: int) -> Any:
    page = selector["pdf_page"]
    if type(page) is not int or not start <= page <= end:
        raise ValueError("invalid_exercise_layout")
    box = _rect_factory(selector["bbox"])
    if box.is_empty or box.is_infinite or not doc[page - 1].rect.contains(box):
        raise ValueError("invalid_exercise_layout")
    return box


def _inside_boundaries(page: int, box: Any, boundaries: list[dict[str, Any]]) -> bool:
    return any(
        boundary["pdf_page"] == page and _rect_factory(boundary["bbox"]).contains(box)
        for boundary in boundaries
    )


@memoize_derivation(lambda: (VERSION, pymupdf.VersionFitz))
def _derive(
    source: Path,
    layout: dict[str, Any],
    start: int,
    end: int,
    recovered: dict[str, Any] | None,
    *,
    table_cells: bool = False,
) -> list[dict[str, Any]]:
    records = []
    ids: set[str] = set()
    fingerprint = _hash_file(source)
    with _open_document(source) as doc:
        for spec in layout["records"]:
            boundaries = spec["boundaries"]
            if not boundaries:
                raise ValueError("invalid_exercise_layout")
            for boundary in boundaries:
                _rect(boundary, doc, start, end)
            from .blank_source_exclusions import verify_blank_exclusions

            blank_proofs, ignored_gaps = verify_blank_exclusions(doc, spec, recovered, fingerprint)
            gaps: list[Any] = []
            selected_regions: set[str] = set()
            raster_boundaries = []
            if spec.get("coverage", "partial") != "complete":
                gaps.append("unverified_completeness")
            # Damage anywhere in the declared parent/continuation/variation region
            # withholds the entire parent, even when the damaged line has no field selector.
            for boundary in boundaries:
                page = doc[boundary["pdf_page"] - 1]
                box = _rect(boundary, doc, start, end)
                if recovered is not None:
                    for gap_index, gap in enumerate(recovered["gaps"]):
                        if gap_index in ignored_gaps:
                            continue
                        if (
                            gap.get("scope") == "outside_owned_blocks"
                            and gap["pdf_page"] == boundary["pdf_page"]
                        ):
                            raster_boundaries.append((boundary, len(gaps)))
                            gaps.append("unresolved_recovery")
                            continue
                        if gap["pdf_page"] == boundary["pdf_page"] and (
                            "bbox" not in gap or box.intersects(_rect_factory(gap["bbox"]))
                        ):
                            gaps.append("unresolved_recovery")
                else:
                    missing, trace = unmapped_fonts(page), page.get_texttrace()
                    lines = [
                        line
                        for block in page.get_text("dict")["blocks"]
                        for line in block.get("lines", [])
                    ]
                    if table_cells:
                        from .cells import native_cell

                        # A PDF line can cross several unrelated cells. Assess
                        # only glyphs actually contained by this declared region,
                        # retaining their geometry for raw/mapping checks.
                        lines = [
                            {"bbox": line["bbox"], "spans": [{"text": line["text"]}]}
                            for line in native_cell(page, box)
                        ]
                    for line in lines:
                        reason = line_reason(
                            line,
                            missing,
                            raw_line_text(line, trace),
                            unmapped=line_has_unmapped_glyphs(page, line),
                        )
                        structural = (
                            table_cells
                            and reason == "suspicious_native"
                            and not any(c.isalpha() for s in line["spans"] for c in s["text"])
                        )
                        if (
                            box.intersects(_rect_factory(line["bbox"]))
                            and reason
                            and not structural
                        ):
                            gaps.append("unresolved_native_text")

            def derive_node(
                node: dict[str, Any],
                boundaries: list[dict[str, Any]] = boundaries,
                gaps: list[Any] = gaps,
                selected_regions: set[str] = selected_regions,
                *,
                require_name: bool = False,
            ) -> dict[str, Any]:
                rid = node["id"]
                node_boundaries = node.get("boundaries", boundaries)
                if not node_boundaries:
                    raise ValueError("invalid_exercise_layout")
                for boundary in node_boundaries:
                    box = _rect(boundary, doc, start, end)
                    if not _inside_boundaries(boundary["pdf_page"], box, boundaries):
                        raise ValueError("invalid_exercise_layout")
                boundaries = node_boundaries
                if node.get("coverage", "complete") != "complete":
                    gaps.append("unverified_node_completeness")
                gaps.extend(node.get("gaps", []))
                if not isinstance(rid, str) or not rid or rid in ids:
                    raise ValueError("invalid_exercise_layout")
                ids.add(rid)
                fields = node["fields"]
                if set(fields) - FIELDS:
                    raise ValueError("invalid_exercise_layout")
                result: dict[str, Any] = {
                    "id": rid,
                    "aliases": node.get("aliases", []),
                    "boundaries": boundaries,
                }
                if not all(isinstance(a, str) and a.strip() for a in result["aliases"]):
                    raise ValueError("invalid_exercise_layout")
                citations = {}
                for field in sorted(FIELDS):
                    segments = []
                    for selector in fields.get(field, []):
                        box = _rect(selector, doc, start, end)
                        number = selector["pdf_page"]
                        if not _inside_boundaries(number, box, boundaries):
                            raise ValueError("invalid_exercise_layout")
                        page = doc[number - 1]
                        if recovered is None:
                            lines = [
                                line
                                for block in page.get_text("dict", sort=True)["blocks"]
                                for line in block.get("lines", [])
                            ]
                            selected: list[dict[str, Any]] = [
                                {
                                    "text": "".join(s["text"] for s in line["spans"]),
                                    "bbox": list(line["bbox"]),
                                }
                                for line in lines
                                if box.contains(_rect_factory(line["bbox"]))
                            ]
                        else:
                            regions = {r["id"]: r for r in recovered["regions"]}
                            selected = []
                            for block in recovered["blocks"]:
                                if block["pdf_page"] != number:
                                    continue
                                for line in block["lines"]:
                                    if box.contains(_rect_factory(line["bbox"])):
                                        region = regions.get(line["region_id"])
                                        selected.append(
                                            {
                                                **line,
                                                "text": region["ocr_text"]
                                                if region
                                                else line["text"],
                                            }
                                        )
                        if recovered is None and table_cells:
                            from .cells import native_cell

                            selected = native_cell(page, box)
                        text = "\n".join(line["text"] for line in selected)
                        quote = selector.get("quote")
                        if quote is not None:
                            if (
                                not isinstance(quote, str)
                                or not quote.strip()
                                or text.count(quote) != 1
                            ):
                                raise ValueError("invalid_exercise_layout")
                            text = quote
                        # Joining is opt-in, explicit and revision-bound, never blanket dehyphenation.
                        original = text
                        for join in selector.get("joins", []):
                            if (
                                not isinstance(join, str)
                                or "-\n" not in join
                                or text.count(join) != 1
                            ):
                                raise ValueError("invalid_exercise_layout")
                            text = text.replace(join, join.replace("-\n", ""), 1)
                        from .raster_ownership import retained_regions

                        selected_regions.update(retained_regions(selected, quote))
                        if not text.strip():
                            gaps.append("empty_field_selector")
                        segments.append(
                            {
                                "text": text,
                                "original_text": original,
                                "quote": quote,
                                "joins": selector.get("joins", []),
                                "lines": selected,
                                "location": {
                                    "pdf_page": number,
                                    "bbox": list(box),
                                    "printed_label": page.get_label() or None,
                                },
                                "source_id": fingerprint,
                            }
                        )
                    citations[field] = segments
                    result[field] = (
                        ("\n".join(s["text"] for s in segments) or None)
                        if field in SCALARS
                        else segments
                    )
                # Variations need not have a stated name; never invent one to make
                # an otherwise complete source-backed child eligible.
                if (require_name and not result["name"]) or not result["instructions"]:
                    gaps.append("missing_identity_or_instructions")
                result["citations"] = citations
                images = []
                for image in node.get("images", []):
                    number, xref = image["pdf_page"], image["xref"]
                    box = _rect(image, doc, start, end)
                    if not _inside_boundaries(number, box, boundaries):
                        raise ValueError("invalid_exercise_layout")
                    page = doc[number - 1]
                    placement = box
                    if "placement_bbox" in image:
                        placement = _rect_factory(image["placement_bbox"])
                        # Only natural page-edge clipping is opt-in. Keep the visible
                        # selector subject to ordinary page and node-boundary checks.
                        if (
                            placement.is_empty
                            or placement.is_infinite
                            or page.rect.contains(placement)
                            or list(placement & page.rect) != list(box)
                        ):
                            raise ValueError("invalid_exercise_image")
                    if xref not in {i[0] for i in page.get_images(full=True)} or not any(
                        list(rect) == list(placement) for rect in page.get_image_rects(xref)
                    ):
                        raise ValueError("invalid_exercise_image")
                    images.append(
                        {
                            "source_id": fingerprint,
                            "pdf_page": number,
                            "xref": xref,
                            "bbox": list(box),
                            **(
                                {"placement_bbox": list(placement)}
                                if "placement_bbox" in image
                                else {}
                            ),
                            "visual_understanding": False,
                        }
                    )
                result["images"] = images
                return result

            record = derive_node(spec, require_name=True)
            if blank_proofs:
                record["blank_source_exclusions"] = blank_proofs
            record["variations"] = []
            for child in spec.get("variations", []):
                if child["relation"] not in {"easier", "harder", "alternative", "unspecified"}:
                    raise ValueError("invalid_exercise_layout")
                record["variations"].append({**derive_node(child), "relation": child["relation"]})
            for node in [record, *record["variations"]]:
                chapter = node["chapter"] or record["chapter"]
                for segments in node["citations"].values():
                    for segment in segments:
                        segment["location"].update(
                            chapter=chapter, section_path=[chapter] if chapter else []
                        )
            from .raster_ownership import owns_boundary

            for boundary, gap_index in reversed(raster_boundaries):
                assert recovered is not None
                if owns_boundary(
                    doc[boundary["pdf_page"] - 1],
                    boundary["pdf_page"],
                    _rect_factory(boundary["bbox"]),
                    recovered,
                    selected_regions,
                ):
                    del gaps[gap_index]
            record.update(
                boundaries=boundaries, coverage="partial" if gaps else "complete", gaps=gaps
            )
            records.append(record)
    return records


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


def draft_exercises(
    pdf: str | Path,
    catalog: str | Path,
    layout: dict[str, Any],
    start: int,
    end: int,
    *,
    recovery: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a private pending catalog from explicit page/bbox field selectors."""
    source, target = Path(pdf).resolve(), Path(catalog).resolve()
    if target == source or (recovery is not None and target == Path(recovery).resolve()):
        raise ValueError("unsafe_review_path")
    preview = inspect_source(source, start, end)
    recovered = inspect_review(recovery) if recovery is not None else None
    if recovered:
        from .recovery_selection import recovery_selection

        recovered = recovery_selection(recovered, preview["sha256"], start, end)
    binding = {
        "source_path": str(source),
        "source_sha256": preview["sha256"],
        "section": preview["section"],
        "settings": VERSION,
        "recovery_path": str(Path(recovery).resolve()) if recovery is not None else None,
        "recovery_sha256": _hash_file(Path(recovery)) if recovery is not None else None,
    }
    old = _read(target) if target.exists() else {"records": []}
    previous = {r["id"]: r for r in old["records"]}
    records = _derive(source, layout, start, end, recovered)
    for record in records:
        record["revision"] = _revision(record, binding)
        prior = previous.get(record["id"], {})
        record["state"] = (
            prior.get("state", "pending")
            if prior.get("revision") == record["revision"]
            else "pending"
        )
        record["history"] = prior.get("history", [])
    value = {"binding": binding, "layout": layout, "records": records}
    if _hash_file(source) != binding["source_sha256"]:
        raise ValueError("source_changed")
    if recovery is not None:
        inspect_review(recovery)
        if _hash_file(Path(recovery)) != binding["recovery_sha256"]:
            raise ValueError("stale_recovery")
    _save(target, value)
    return inspect_exercises(target)


def _save(path: Path, value: dict[str, Any]) -> None:
    _atomic_json(path, {**value, "seal": _digest(_json(value).encode())})


def _read(path: Path) -> dict[str, Any]:
    try:
        value: dict[str, Any] = json.loads(path.read_text())
        seal = value.pop("seal")
        if seal != _digest(_json(value).encode()) or any(
            r["revision"] != _revision(r, value["binding"])
            or r["state"] not in {"pending", "accepted", "rejected"}
            for r in value["records"]
        ):
            raise ValueError("stale_exercises")
        return value
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("stale_exercises") from None


def inspect_exercises(catalog: str | Path) -> dict[str, Any]:
    """Validate source, recovery, selectors and record revisions; expose eligibility/gaps."""
    value = _read(Path(catalog))
    b = value["binding"]
    try:
        if _hash_file(Path(b["source_path"])) != b["source_sha256"] or b["settings"] != VERSION:
            raise ValueError("stale_exercises")
        recovered = inspect_review(b["recovery_path"]) if b["recovery_path"] else None
        if b["recovery_path"] and _hash_file(Path(b["recovery_path"])) != b["recovery_sha256"]:
            raise ValueError("stale_exercises")
        derived = _derive(
            Path(b["source_path"]),
            value["layout"],
            b["section"]["start"],
            b["section"]["end"],
            recovered,
        )
        if len(derived) != len(value["records"]) or any(
            _revision(r, b) != stored["revision"]
            for r, stored in zip(derived, value["records"], strict=True)
        ):
            raise ValueError("stale_exercises")
        if _hash_file(Path(b["source_path"])) != b["source_sha256"]:
            raise ValueError("stale_exercises")
        if b["recovery_path"] and (
            inspect_review(b["recovery_path"]) != recovered
            or _hash_file(Path(b["recovery_path"])) != b["recovery_sha256"]
        ):
            raise ValueError("stale_exercises")
        if _read(Path(catalog)) != value:
            raise ValueError("stale_exercises")
    except OSError:
        raise ValueError("stale_exercises") from None
    for record in value["records"]:
        record["eligible"] = record["state"] == "accepted" and record["coverage"] == "complete"
    return value


def review_exercise(
    catalog: str | Path, record_id: str, revision: str, state: str
) -> dict[str, Any]:
    value = inspect_exercises(catalog)
    if state not in {"pending", "accepted", "rejected"}:
        raise ValueError("invalid_review_state")
    record = next((r for r in value["records"] if r["id"] == record_id), None)
    if record is None or record["revision"] != revision:
        raise ValueError("review_revision_mismatch")
    record["state"] = state
    record["history"].append({"revision": revision, "state": state})
    for item in value["records"]:
        item.pop("eligible", None)
    _save(Path(catalog), value)
    return inspect_exercises(catalog)


def validate_exercises(binding: dict[str, Any]) -> None:
    inspect_exercises(binding["catalog_path"])
    if _hash_file(Path(binding["catalog_path"])) != binding["catalog_sha256"]:
        raise ValueError("stale_exercises")


def exercise_passages(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    passages = []
    for record in catalog["records"]:
        if not record["eligible"]:
            continue
        texts = []
        for node in [record, *record["variations"]]:
            texts.extend(node["aliases"])
            texts.extend(
                segment["text"] for segments in node["citations"].values() for segment in segments
            )
        for node in [record, *record["variations"]]:
            # Shared reference prose remains searchable, but must not dilute the
            # node's own identity/instruction vector or move its chunk boundaries.
            matching_texts = [
                "\n\n".join(
                    [*node["aliases"]]
                    + [
                        segment["text"]
                        for field, segments in node["citations"].items()
                        if field != "references"
                        for segment in segments
                    ]
                ),
                "\n\n".join(segment["text"] for segment in node["citations"]["references"]),
            ]
            windows = [
                window
                for matching_text in matching_texts
                for window in textwrap.wrap(
                    matching_text,
                    width=CHUNK_CHARS,
                    replace_whitespace=False,
                    expand_tabs=False,
                    drop_whitespace=False,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            ]
            for number, window in enumerate(windows):
                passages.append(
                    {
                        "source_id": catalog["binding"]["source_sha256"],
                        "passage_id": _digest(
                            _json([record["revision"], node["id"], number, window]).encode()
                        ),
                        # The evidence stays complete even when the matching vector
                        # represents a late instruction or a child variation.
                        "text": "\n\n".join(texts),
                        "retrieval_text": window,
                        # Identity headings may live in navigation rather than the body.
                        "location": node["citations"]["instructions"][0]["location"],
                        "exercise": record,
                        "eligible": True,
                        "review_status": "accepted",
                        "provenance": {
                            "exercise_revision": record["revision"],
                            "matched_node_id": node["id"],
                            "chunk": number,
                            **VERSION,
                        },
                    }
                )
    return passages


def coverage_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose eligibility without returning withheld drafts or operator review notes."""
    reasons = {
        "unverified_completeness",
        "unverified_node_completeness",
        "unresolved_recovery",
        "unresolved_native_text",
        "empty_field_selector",
        "missing_identity_or_instructions",
    }
    return [
        {
            **{k: record[k] for k in ("id", "revision", "state", "coverage", "eligible")},
            "gaps": [
                gap if isinstance(gap, str) and gap in reasons else "operator_reported_gap"
                for gap in record["gaps"]
            ],
        }
        for record in records
    ]


def named_candidates(question: str, passages: list[dict[str, Any]]) -> list[int]:
    """Explicit names/validated aliases select identities, never merge shared aliases."""

    def normalize(text: str) -> str:
        return " ".join(re.findall(r"\w+", text.casefold()))

    query = " " + normalize(question) + " "
    matches = [
        (match.start(), match.end(), i)
        for i, passage in enumerate(passages)
        for node in [passage["exercise"], *passage["exercise"]["variations"]]
        for name in [node["name"], *node["aliases"]]
        if name and normalize(name)
        for match in re.finditer(re.escape(" " + normalize(name) + " "), query)
    ]
    # Prefer a specific complete name over a shorter alias contained in that
    # same mention, not over unrelated names elsewhere in the question.
    maximal = {
        i
        for start, end, i in matches
        if not any(
            other_start <= start and end <= other_end and other_end - other_start > end - start
            for other_start, other_end, _ in matches
        )
    }
    return sorted(maximal)
