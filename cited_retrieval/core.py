"""Artifact-bound native-text import. All page numbers are one-based."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from .embedder import Embedder
from .policy import CHUNK_CHARS, MIN_ALPHA_FRACTION, MIN_PROSE_WORDS, ranking_settings

SETTINGS = {
    "schema": 2,
    "extractor": "pymupdf-blocks",
    "extractor_version": pymupdf.VersionBind,
    "chunk_chars": CHUNK_CHARS,
    "min_prose_words": MIN_PROSE_WORDS,
    "min_alpha_fraction": MIN_ALPHA_FRACTION,
    "quality": "native-unicode-v2",
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
    # PDF base-14 fonts have a specified standard mapping without a ToUnicode stream.
    base14 = {
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
    for font in page.get_fonts(full=True):
        mapping = page.parent.xref_get_key(font[0], "ToUnicode")
        standard_simple_encoding = font[2] in {"Type1", "TrueType"} and font[5] in {
            "WinAnsiEncoding",
            "MacRomanEncoding",
        }
        if mapping[0] == "null" and not (
            standard_simple_encoding
            or (
                font[3] in base14
                and font[1] == "n/a"
                and font[5] in ("WinAnsiEncoding", "MacRomanEncoding", "")
            )
        ):
            return "missing_unicode_map"
    if not text.strip():
        return "no_native_text"
    if any(
        unicodedata.category(c) in {"Cc", "Co", "Cs", "Cn"} and c not in "\n\r\t" or c == "\ufffd"
        for c in text
    ):
        return "invalid_unicode"
    # Missing character mappings must never silently become approved text.
    for span in page.get_texttrace():
        if any(c[0] in (0, 0xFFFD) for c in span["chars"]):
            return "unmapped_glyphs"
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


def _passages(pdf: Path, preview: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    boundaries = sorted(preview["bookmarks"], key=lambda b: (b["pdf_page"], b["y"]))
    with pymupdf.open(pdf) as doc:
        for info in preview["pages"]:
            if info["status"] != "candidate":
                continue
            number = info["pdf_page"]
            page = doc[number - 1]
            chunks: list[tuple[list[str], list[float], str]] = []
            for block in page.get_text("blocks", sort=True):
                if block[6] != 0 or not block[4].strip():
                    continue
                if block[4].strip() in {b["title"] for b in boundaries}:
                    continue
                # Split blocks at bookmark boundaries, including boundaries inside a page.
                block_start = len(chunks)
                lines = page.get_text("dict", clip=pymupdf.Rect(block[:4]))["blocks"]
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
                if not _is_prose(text):
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
                        "provenance": SETTINGS.copy(),
                        "eligible": True,
                        "review_status": "acknowledged",
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
) -> dict[str, Any]:
    """Rebuild atomically after explicit review of this artifact's selected pages."""
    path, target = Path(pdf).resolve(), Path(index).resolve()
    if path == target:
        raise ValueError("unsafe_index_path")
    preview = inspect_source(path, start, end)
    if approval != {"sha256": preview["sha256"], "pages": list(range(start, end + 1))}:
        raise ValueError("review_required")
    passages = _passages(path, preview)
    if not passages:
        raise ValueError("no_eligible_passages")
    identity = dict(embedder.identity)
    if not all(identity.get(key) for key in ("model", "version", "artifact_sha256")):
        raise ValueError("invalid_model_identity")
    vectors = _vectors(embedder, [p["text"] for p in passages])
    if identity != embedder.identity:
        raise ValueError("model_changed")
    if _hash_file(path) != preview["sha256"]:
        raise ValueError("source_changed")
    imported_pages = {p["location"]["pdf_page"] for p in passages}
    for page in preview["pages"]:
        if page["status"] == "candidate":
            if page["pdf_page"] in imported_pages:
                page["status"] = "imported"
            else:
                page["status"] = "excluded"
                page["reason"] = "no_eligible_prose"
    result = {
        **preview,
        "passage_count": len(passages),
        "model": identity,
        "dimensions": len(vectors[0]),
        "settings": SETTINGS.copy(),
        "approval": approval,
    }
    records = [[_json(p), _json(v)] for p, v in zip(passages, vectors, strict=True)]
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
                [(i, p["text"]) for i, p in enumerate(passages, 1)],
            )
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return result


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
        expected_terms = [(i, json.loads(r[0])["text"]) for i, r in enumerate(records, 1)]
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
        return db, meta, records
    except BaseException:
        db.close()
        raise


def report(index: str | Path) -> dict[str, Any]:
    db, meta, _ = _load(index)
    db.close()
    return dict(meta["report"])
