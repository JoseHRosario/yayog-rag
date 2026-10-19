"""Artifact-bound native-text import. All page numbers are one-based."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from .embedder import Embedder
from .policy import CHUNK_CHARS, MIN_ALPHA_FRACTION, MIN_PROSE_WORDS, ranking_settings

SETTINGS = {
    "schema": 3,
    "extractor": "pymupdf-blocks",
    "extractor_version": pymupdf.VersionBind,
    "chunk_chars": CHUNK_CHARS,
    "min_prose_words": MIN_PROSE_WORDS,
    "min_alpha_fraction": MIN_ALPHA_FRACTION,
    "quality": "raw-line-unicode-v5",
    "ranking": ranking_settings(),
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256")
    return digest.hexdigest()


def _is_prose(text: str) -> bool:
    return (
        len(text.split()) >= MIN_PROSE_WORDS
        and sum(c.isalpha() for c in text) / len(text) >= MIN_ALPHA_FRACTION
    )


def _quality(page: Any, text: str) -> str | None:
    from .text_quality import (
        invalid_characters,
        line_has_unmapped_glyphs,
        line_reason,
        raw_line_text,
        unmapped_fonts,
    )

    missing = unmapped_fonts(page)
    if missing:
        return "missing_unicode_map"
    if not text.strip():
        return "no_native_text"
    if invalid_characters(text):
        return "invalid_unicode"
    trace = page.get_texttrace()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            reason = line_reason(
                line,
                missing,
                raw_line_text(line, trace),
                unmapped=line_has_unmapped_glyphs(page, line),
            )
            if reason:
                return reason
    if not _is_prose(text):
        return "not_prose"
    return None


def inspect_source(pdf: str | Path, start: int, end: int) -> dict[str, Any]:
    """Preview quality/accounting; this is not approval. Does not expose source text."""
    path = Path(pdf)
    fingerprint = _hash_file(path)
    with pymupdf.open(path) as doc:
        if doc.is_encrypted or not 1 <= start <= end <= len(doc):
            raise ValueError("invalid_section")
        toc = doc.get_toc(simple=False)
        pages = []
        for number, page in enumerate(doc, 1):
            selected = start <= number <= end
            reason = _quality(page, page.get_text()) if selected else "not_selected"
            pages.append(
                {
                    "pdf_page": number,
                    "printed_label": page.get_label() or None,
                    "status": "candidate" if reason is None else "excluded",
                    "reason": reason,
                }
            )
        boundaries = [
            {
                "level": t[0],
                "title": t[1],
                "pdf_page": t[2],
                "y": float(t[3].get("to", pymupdf.Point(0, 0)).y),
            }
            for t in toc
            if t[2] > 0
        ]
    if _hash_file(path) != fingerprint:
        raise ValueError("source_changed")
    return {
        "sha256": fingerprint,
        "section": {"start": start, "end": end},
        "pages": pages,
        "bookmarks": boundaries,
    }


def _passages(
    pdf: Path, preview: dict[str, Any], recovery: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    boundaries = sorted(preview["bookmarks"], key=lambda b: (b["pdf_page"], b["y"]))
    with pymupdf.open(pdf) as doc:
        for info in preview["pages"]:
            if info["status"] != "candidate":
                continue
            number = info["pdf_page"]
            page = doc[number - 1]
            chunks: list[tuple[list[str], list[float], str]] = []
            native_blocks = page.get_text("blocks", sort=True)
            recovered_blocks = (
                []
                if recovery is None
                else [b for b in recovery["blocks"] if b["pdf_page"] == number]
            )
            regions = {} if recovery is None else {r["id"]: r for r in recovery["regions"]}
            eligible_blocks = []
            for b in recovered_blocks:
                if any(
                    line["region_id"] and regions[line["region_id"]]["state"] != "accepted"
                    for line in b["lines"]
                ):
                    continue
                eligible_blocks.append(b)
            selected_blocks = (
                native_blocks
                if recovery is None
                else [
                    (*b["bbox"], "\n".join(line["text"] for line in b["lines"]), i, 0)
                    for i, b in enumerate(eligible_blocks)
                ]
            )
            for block in selected_blocks:
                if block[6] != 0 or not block[4].strip():
                    continue
                if block[4].strip() in {b["title"] for b in boundaries}:
                    continue
                # Split blocks at bookmark boundaries, including boundaries inside a page.
                block_start = len(chunks)
                lines = (
                    page.get_text("dict", clip=pymupdf.Rect(block[:4]))["blocks"]
                    if recovery is None
                    else [
                        {
                            "lines": [
                                {
                                    "bbox": line["bbox"],
                                    "spans": [
                                        {
                                            "text": regions[line["region_id"]]["ocr_text"]
                                            if line["region_id"]
                                            else line["text"]
                                        }
                                    ],
                                }
                                for line in eligible_blocks[block[5]]["lines"]
                            ]
                        }
                    ]
                )
                for b in lines:
                    for line in b.get("lines", []):
                        text = "".join(s["text"] for s in line["spans"]).strip()
                        if not text:
                            continue
                        bbox = list(line["bbox"])
                        previous = [
                            x
                            for x in boundaries
                            if x["pdf_page"] < number
                            or (x["pdf_page"] == number and x["y"] <= bbox[3])
                        ]
                        ancestors: list[dict[str, Any]] = []
                        for bookmark in previous:
                            ancestors = [a for a in ancestors if a["level"] < bookmark["level"]]
                            ancestors.append(bookmark)
                        section_path = [a["title"] for a in ancestors]
                        if (
                            len(chunks) > block_start
                            and chunks[-1][0] == section_path
                            and len(chunks[-1][2]) + len(text) < CHUNK_CHARS
                        ):
                            old = chunks.pop()
                            chunks.append(
                                (
                                    section_path,
                                    [
                                        min(old[1][0], bbox[0]),
                                        min(old[1][1], bbox[1]),
                                        max(old[1][2], bbox[2]),
                                        max(old[1][3], bbox[3]),
                                    ],
                                    old[2] + "\n" + text,
                                )
                            )
                        else:
                            chunks.append((section_path, bbox, text))
            for ordinal, (section_path, bbox, text) in enumerate(chunks):
                used_regions = [
                    r
                    for r in regions.values()
                    if r["pdf_page"] == number
                    and r["state"] == "accepted"
                    and pymupdf.Rect(r["bbox"]).intersects(pymupdf.Rect(bbox))
                ]
                if not _is_prose(text) and not used_regions:
                    if recovery is not None:
                        recovery["gaps"].append(
                            {
                                "pdf_page": number,
                                "bbox": bbox,
                                "reason": "excluded_non_prose",
                                "text": text,
                            }
                        )
                    continue
                location = {
                    "pdf_page": number,
                    "printed_label": info["printed_label"],
                    "chapter": section_path[0] if section_path else None,
                    "section_path": section_path,
                    "bbox": bbox,
                    "ordinal": ordinal,
                }
                pid = _digest((preview["sha256"] + _json(location) + text).encode())
                result.append(
                    {
                        "source_id": preview["sha256"],
                        "passage_id": pid,
                        "text": text,
                        "location": location,
                        "provenance": {
                            **SETTINGS,
                            **({"recovery_regions": used_regions} if used_regions else {}),
                        },
                        "eligible": True,
                        "review_status": "accepted" if used_regions else "acknowledged",
                    }
                )
    return result


def _vectors(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    vectors = np.asarray(embedder.embed(texts), dtype=np.float64)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(texts)
        or vectors.shape[1] < 1
        or not np.isfinite(vectors).all()
        or (np.linalg.norm(vectors, axis=1) == 0).any()
    ):
        raise ValueError("invalid_embeddings")
    return vectors.tolist()  # type: ignore[no-any-return]


def ingest(
    pdf: str | Path,
    index: str | Path,
    start: int,
    end: int,
    embedder: Embedder,
    *,
    approval: dict[str, Any] | None = None,
    recovery: str | Path | None = None,
    exercises: str | Path | None = None,
    programmes: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild atomically after explicit review of this artifact's selected pages."""
    path, target = Path(pdf).resolve(), Path(index).resolve()
    if path == target:
        raise ValueError("unsafe_index_path")
    preview = inspect_source(path, start, end)
    if approval != {"sha256": preview["sha256"], "pages": list(range(start, end + 1))}:
        raise ValueError("review_required")
    recovered = None
    recovery_hash = None
    if recovery is not None:
        from .recovery import inspect_review

        recovery_path = Path(recovery).resolve()
        if recovery_path == target:
            raise ValueError("unsafe_index_path")
        recovery_hash = _hash_file(recovery_path)
        recovered = inspect_review(recovery_path)
        if (
            recovered["source_sha256"] != preview["sha256"]
            or recovered["section"] != preview["section"]
        ):
            raise ValueError("stale_recovery")
        for info in preview["pages"]:
            if start <= info["pdf_page"] <= end:
                info["status"] = "candidate"
                info["reason"] = None
    exercise_binding = None
    programme_binding = None
    if programmes is not None:
        from .programmes import inspect_programmes, programme_passages, protected_artifacts

        if exercises is not None:
            raise ValueError("invalid_programme_layout")
        catalog_path = Path(programmes).resolve()
        if catalog_path == target:
            raise ValueError("unsafe_index_path")
        catalog_hash = _hash_file(catalog_path)
        catalog = inspect_programmes(catalog_path)
        binding = catalog["binding"]
        if target in protected_artifacts(binding):
            raise ValueError("unsafe_index_path")
        if (
            binding["source_sha256"] != preview["sha256"]
            or binding["section"] != preview["section"]
            or binding["recovery_sha256"] != recovery_hash
        ):
            raise ValueError("stale_programmes")
        programme_binding = {
            "catalog_path": str(catalog_path),
            "catalog_sha256": catalog_hash,
            "records": catalog["records"],
        }
        passages = programme_passages(catalog)
    elif exercises is not None:
        from .exercises import (
            INSTRUCTION_ANCHOR_VERSION,
            MATCHING_WINDOW_VERSION,
            exercise_passages,
            inspect_exercises,
        )

        catalog_path = Path(exercises).resolve()
        if catalog_path == target:
            raise ValueError("unsafe_index_path")
        catalog_hash = _hash_file(catalog_path)
        catalog = inspect_exercises(catalog_path)
        binding = catalog["binding"]
        if (
            binding["source_sha256"] != preview["sha256"]
            or binding["section"] != preview["section"]
            or binding["recovery_sha256"] != recovery_hash
        ):
            raise ValueError("stale_exercises")
        exercise_binding = {
            "instruction_anchor_version": INSTRUCTION_ANCHOR_VERSION,
            "matching_window_version": MATCHING_WINDOW_VERSION,
            "catalog_path": str(catalog_path),
            "catalog_sha256": catalog_hash,
            "records": catalog["records"],
        }
        passages = exercise_passages(catalog)
    else:
        passages = _passages(path, preview, recovered)
    if not passages:
        raise ValueError("no_eligible_passages")
    identity = dict(embedder.identity)
    if not all(identity.get(key) for key in ("model", "version", "artifact_sha256")):
        raise ValueError("invalid_model_identity")
    vectors = _vectors(embedder, [p.get("retrieval_text", p["text"]) for p in passages])
    if identity != embedder.identity:
        raise ValueError("model_changed")
    if programme_binding is not None:
        from .programmes import validate_programmes

        validate_programmes(programme_binding)
    if exercise_binding is not None:
        from .exercises import validate_exercises

        validate_exercises(exercise_binding)
    if _hash_file(path) != preview["sha256"]:
        raise ValueError("source_changed")
    if recovery is not None:
        from .recovery import load_bundle

        load_bundle(Path(recovery).resolve())
        if _hash_file(Path(recovery)) != recovery_hash:
            raise ValueError("stale_recovery")
    imported_pages = {p["location"]["pdf_page"] for p in passages}
    if exercise_binding is not None:
        imported_pages = {
            b["pdf_page"]
            for r in exercise_binding["records"]
            if r["eligible"]
            for b in r["boundaries"]
        }
        for page in preview["pages"]:
            if start <= page["pdf_page"] <= end:
                page["status"] = "candidate"
                page["reason"] = None
    if programme_binding is not None:
        imported_pages = {
            b["pdf_page"]
            for r in programme_binding["records"]
            if r["eligible"] and r["kind"] == "session"
            for b in r["boundaries"]
        }
        imported_pages.update(
            b["pdf_page"]
            for passage in passages
            for reference in [passage["session"]["method_reference"]]
            if reference["status"] == "resolved"
            for b in reference["record"]["boundaries"]
        )
        for page in preview["pages"]:
            if start <= page["pdf_page"] <= end:
                page.update(status="candidate", reason=None)
    for page in preview["pages"]:
        if page["status"] == "candidate":
            if page["pdf_page"] in imported_pages:
                page["status"] = "imported"
            else:
                page["status"] = "excluded"
                page["reason"] = (
                    "no_eligible_sessions"
                    if programme_binding is not None
                    else "no_eligible_exercises"
                    if exercise_binding is not None
                    else "no_eligible_prose"
                )
    result = {
        **preview,
        "passage_count": len(passages),
        "model": identity,
        "dimensions": len(vectors[0]),
        "settings": SETTINGS.copy(),
        "approval": approval,
    }
    if programme_binding is not None:
        result["programmes"] = programme_binding
        for page in result["pages"]:
            related = [
                r
                for r in programme_binding["records"]
                if any(b["pdf_page"] == page["pdf_page"] for b in r["boundaries"])
            ]
            if page["status"] == "imported" and any(not r["eligible"] for r in related):
                page.update(status="partial", reason="excluded_sessions")
    records = [[_json(p), _json(v)] for p, v in zip(passages, vectors, strict=True)]
    if exercise_binding is not None:
        result["exercises"] = exercise_binding
        for page in result["pages"]:
            related = [
                r
                for r in exercise_binding["records"]
                if any(b["pdf_page"] == page["pdf_page"] for b in r["boundaries"])
            ]
            if page["status"] == "imported" and any(not r["eligible"] for r in related):
                page.update(status="partial", reason="excluded_exercises")
    if recovered is not None and recovery is not None:
        result["recovery"] = {
            "bundle_path": str(Path(recovery).resolve()),
            "bundle_sha256": recovery_hash,
            "unresolved_regions": recovered["unresolved_regions"],
            "gaps": recovered["gaps"],
            "warnings": recovered["warnings"],
        }
        for page in result["pages"]:
            gaps = [
                r
                for r in recovered["regions"]
                if r["pdf_page"] == page["pdf_page"] and r["state"] != "accepted"
            ]
            page["unresolved_regions"] = len(gaps)
            page_gaps = [g for g in recovered["gaps"] if g["pdf_page"] == page["pdf_page"]]
            if (gaps or page_gaps) and page["status"] == "imported":
                page["status"] = "partial"
                page["reason"] = "unresolved_recovery" if gaps else "excluded_content"
    result["corpus_sha256"] = _digest(_json(records).encode())
    meta = {
        "report": result,
        "source_path": str(path),
        "corpus_sha256": _digest(_json(records).encode()),
    }
    meta["manifest_sha256"] = _digest(_json(meta).encode())
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".rebuild-", suffix=".sqlite", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with sqlite3.connect(temporary) as db:
            db.execute("CREATE TABLE metadata (json TEXT NOT NULL)")
            db.execute("INSERT INTO metadata VALUES (?)", (_json(meta),))
            db.execute("CREATE TABLE passages (json TEXT NOT NULL, vector TEXT NOT NULL)")
            db.executemany("INSERT INTO passages VALUES (?, ?)", records)
            db.execute("CREATE VIRTUAL TABLE terms USING fts5(text)")
            db.executemany(
                "INSERT INTO terms(rowid,text) VALUES (?,?)",
                [(i, p.get("retrieval_text", p["text"])) for i, p in enumerate(passages, 1)],
            )
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def _validate_recovery(binding: dict[str, Any]) -> None:
    from .recovery import load_bundle

    review_path = Path(binding["bundle_path"])
    load_bundle(review_path)
    try:
        if _hash_file(review_path) != binding["bundle_sha256"]:
            raise ValueError("stale_recovery")
    except OSError:
        raise ValueError("stale_recovery") from None


def _load(index: str | Path) -> tuple[sqlite3.Connection, dict[str, Any], list[list[str]]]:
    db = sqlite3.connect(Path(index).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        meta = json.loads(db.execute("SELECT json FROM metadata").fetchone()[0])
        seal = meta.pop("manifest_sha256")
        if _digest(_json(meta).encode()) != seal:
            raise ValueError("stale_corpus")
        records = [list(r) for r in db.execute("SELECT json,vector FROM passages ORDER BY rowid")]
        if _digest(_json(records).encode()) != meta["corpus_sha256"]:
            raise ValueError("stale_corpus")
        terms = list(db.execute("SELECT rowid,text FROM terms ORDER BY rowid"))
        expected_terms = [
            (i, passage.get("retrieval_text", passage["text"]))
            for i, (payload, _) in enumerate(records, 1)
            for passage in [json.loads(payload)]
        ]
        if terms != expected_terms:
            raise ValueError("stale_corpus")
        if meta["report"]["settings"] != SETTINGS:
            raise ValueError("stale_settings")
        try:
            source_hash = _hash_file(Path(meta["source_path"]))
        except OSError:
            raise ValueError("stale_source") from None
        if source_hash != meta["report"]["sha256"]:
            raise ValueError("stale_source")
        if "recovery" in meta["report"]:
            _validate_recovery(meta["report"]["recovery"])
        if "exercises" in meta["report"]:
            from .exercises import (
                INSTRUCTION_ANCHOR_VERSION,
                MATCHING_WINDOW_VERSION,
                validate_exercises,
            )

            anchor_version = meta["report"]["exercises"].get("instruction_anchor_version")
            if type(anchor_version) is not int or anchor_version != INSTRUCTION_ANCHOR_VERSION:
                raise ValueError("stale_settings")
            matching_version = meta["report"]["exercises"].get("matching_window_version")
            if type(matching_version) is not int or matching_version != MATCHING_WINDOW_VERSION:
                raise ValueError("stale_settings")
            validate_exercises(meta["report"]["exercises"])
        if "programmes" in meta["report"]:
            from .programmes import validate_programmes

            validate_programmes(meta["report"]["programmes"])
        return db, meta, records
    except BaseException:
        db.close()
        raise


def report(index: str | Path) -> dict[str, Any]:
    db, meta, _ = _load(index)
    db.close()
    return dict(meta["report"])
