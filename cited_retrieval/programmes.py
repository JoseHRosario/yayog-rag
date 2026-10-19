"""Explicit schedule-cell mapping and exact-revision, per-record visual review.

The operator supplies relationships, never an automatically inferred table grid.
"""

from __future__ import annotations

import copy
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import pymupdf

from .cells import render_digest
from .core import _digest, _hash_file, _json, inspect_source
from .derivation_cache import memoize_derivation
from .exercises import VERSION as EXERCISE_VERSION
from .exercises import _derive as derive_source
from .exercises import _revision, _save, inspect_exercises
from .recovery import inspect_review

VERSION = {"programmes": 1, "pymupdf": pymupdf.VersionBind}
SESSION_FIELDS = {"programme", "week_range", "day", "category", "method", "notes"}
ENTRY_FIELDS = {"exercise", "pair", "reps", "pause", "assistance", "variations", "notes"}
_open: Any = pymupdf.open
_rect: Any = pymupdf.Rect


@memoize_derivation(lambda: (VERSION, EXERCISE_VERSION, pymupdf.VersionFitz))
def _derive(
    source: Path, layout: dict[str, Any], start: int, end: int, recovered: dict[str, Any] | None
) -> list[dict[str, Any]]:
    records = []
    ids: set[str] = set()
    for spec in layout["records"]:
        kind, rid = spec["kind"], spec["id"]
        if kind not in {"session", "method"} or not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError("invalid_programme_layout")
        ids.add(rid)
        allowed = SESSION_FIELDS if kind == "session" else {"name", "definition"}
        nodes = [(spec, allowed)]
        if kind == "session":
            weeks, day = spec["weeks"], spec["day"]
            if (
                not isinstance(weeks, list)
                or not weeks
                or any(type(w) is not int or not 1 <= w <= 104 for w in weeks)
                or len(set(weeks)) != len(weeks)
                or type(day) is not int
                or not 1 <= day <= 7
            ):
                raise ValueError("invalid_programme_layout")
            nodes.extend((e, ENTRY_FIELDS) for e in spec["entries"])
            for conflict in spec.get("conflicts", []):
                if len(conflict["claims"]) < 2:
                    raise ValueError("invalid_programme_layout")
                nodes.extend(({"fields": {"claim": c}}, {"claim"}) for c in conflict["claims"])
        selectors: list[dict[str, Any]] = []
        lengths = []
        for node, fields in nodes:
            if set(node["fields"]) - fields:
                raise ValueError("invalid_programme_layout")
            mapping = {}
            for field in sorted(fields):
                selected = node["fields"].get(field, [])
                if not isinstance(selected, list):
                    raise TypeError("invalid_programme_layout")
                mapping[field] = (len(selectors), len(selected))
                selectors.extend(selected)
            lengths.append(mapping)
        # Reuse the source trust, boundary, recovery and exact-quote machinery.
        mapped = {
            "records": [
                {
                    "id": rid,
                    "boundaries": spec["boundaries"],
                    "coverage": spec.get("coverage", "partial"),
                    "gaps": spec.get("gaps", []),
                    "fields": {"name": [], "instructions": selectors},
                }
            ]
        }
        native = derive_source(source, mapped, start, end, recovered, table_cells=True)[0]
        segments = native["citations"]["instructions"]
        with _open(source) as doc:
            for segment in segments:
                loc = segment["location"]
                segment["render_sha256"] = render_digest(
                    doc[loc["pdf_page"] - 1], _rect(loc["bbox"])
                )
        values: list[dict[str, Any]] = []
        for mapping in lengths:
            citations = {
                field: segments[offset : offset + count]
                for field, (offset, count) in mapping.items()
            }
            values.append(
                {
                    **{
                        field: "\n".join(s["text"] for s in segs) or None
                        for field, segs in citations.items()
                    },
                    "citations": citations,
                }
            )
        gaps = [g for g in native["gaps"] if g != "missing_identity_or_instructions"]
        record = {
            "id": rid,
            "kind": kind,
            "boundaries": spec["boundaries"],
            **values[0],
            "aliases": spec.get("aliases", []),
        }
        if not isinstance(record["aliases"], list) or not all(
            isinstance(a, str) and a.strip() for a in record["aliases"]
        ):
            raise ValueError("invalid_programme_layout")
        if kind == "method":
            if not record["name"] or not record["definition"]:
                gaps.append("missing_method_definition")
        else:
            record.update(weeks=weeks, day_number=day, entries=[], conflicts=[])
            exercise_cells: set[str] = set()
            for entry in spec["entries"]:
                for cell in entry["fields"].get("exercise", []):
                    key = _json([cell["pdf_page"], cell["bbox"], cell.get("quote")])
                    if key in exercise_cells:
                        raise ValueError("invalid_programme_layout")
                    exercise_cells.add(key)
            if (
                any(not record[f] for f in ("programme", "week_range", "day"))
                or not spec["entries"]
            ):
                gaps.append("missing_session_scope")
            for i, entry in enumerate(spec["entries"]):
                value = values[1 + i]
                pair = entry.get("pair")
                if pair is not None and (
                    not isinstance(pair, str) or not pair.strip() or not value["pair"]
                ):
                    raise ValueError("invalid_programme_layout")
                if not value["exercise"]:
                    gaps.append("missing_exercise_cell")
                record["entries"].append(
                    {
                        **value,
                        "order": i + 1,
                        "pair_id": pair,
                        "within_pair_order": (
                            sum(e.get("pair") == pair for e in spec["entries"][:i]) + 1
                        )
                        if pair is not None
                        else None,
                    }
                )
            offset = 1 + len(spec["entries"])
            for conflict in spec.get("conflicts", []):
                count = len(conflict["claims"])
                claims = values[offset : offset + count]
                if any(not c["claim"] for c in claims):
                    gaps.append("empty_conflict_claim")
                record["conflicts"].append(
                    {
                        "claims": [c["citations"]["claim"] for c in claims],
                        "status": "source_conflict",
                    }
                )
                offset += count
        record.update(gaps=gaps, coverage="partial" if gaps else "complete")
        records.append(record)
    return records


def _read(path: Path) -> dict[str, Any]:
    try:
        value: dict[str, Any] = json.loads(path.read_text())
        seal = value.pop("seal")
        if seal != _digest(_json(value).encode()) or any(
            r["revision"] != _revision(r, value["binding"])
            or r["state"] not in {"pending", "accepted", "rejected"}
            for r in value["records"]
        ):
            raise ValueError("stale_programmes")
        return value
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("stale_programmes") from None


def _dependencies(binding: dict[str, Any]) -> None:
    for dependency in binding["exercise_catalogs"]:
        inspect_exercises(dependency["path"])
        if _hash_file(Path(dependency["path"])) != dependency["sha256"]:
            raise ValueError("stale_programmes")


def protected_artifacts(binding: dict[str, Any]) -> set[Path]:
    """Enumerate validated source/review/crop inputs that outputs must never replace."""
    paths: set[Path] = set()

    def collect(current: dict[str, Any]) -> None:
        paths.add(Path(current["source_path"]).resolve())
        if current.get("recovery_path"):
            recovery_path = Path(current["recovery_path"]).resolve()
            paths.add(recovery_path)
            recovered = inspect_review(recovery_path)
            paths.add(Path(recovered["source_path"]).resolve())
            paths.update((recovery_path.parent / r["crop"]).resolve() for r in recovered["regions"])
        for dependency in current.get("exercise_catalogs", []):
            path = Path(dependency["path"]).resolve()
            if path not in paths:
                paths.add(path)
                collect(inspect_exercises(path)["binding"])

    collect(binding)
    return paths


def draft_programmes(
    pdf: str | Path,
    catalog: str | Path,
    layout: dict[str, Any],
    start: int,
    end: int,
    *,
    recovery: str | Path | None = None,
    exercise_catalogs: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Draft source-selected sessions/methods; acceptance means visual layout verification."""
    source, target = Path(pdf).resolve(), Path(catalog).resolve()
    deps = [Path(p).resolve() for p in exercise_catalogs or []]
    if target == source or target in deps or (recovery and target == Path(recovery).resolve()):
        raise ValueError("unsafe_review_path")
    preview = inspect_source(source, start, end)
    recovered = inspect_review(recovery) if recovery else None
    if recovered and (
        recovered["source_sha256"] != preview["sha256"]
        or recovered["section"] != preview["section"]
    ):
        raise ValueError("stale_recovery")
    binding = {
        "source_path": str(source),
        "source_sha256": preview["sha256"],
        "section": preview["section"],
        "settings": VERSION,
        "recovery_path": str(Path(recovery).resolve()) if recovery else None,
        "recovery_sha256": _hash_file(Path(recovery)) if recovery else None,
        "exercise_catalogs": [{"path": str(p), "sha256": _hash_file(p)} for p in deps],
    }
    _dependencies(binding)
    if target in protected_artifacts(binding):
        raise ValueError("unsafe_review_path")
    previous = {r["id"]: r for r in _read(target)["records"]} if target.exists() else {}
    try:
        records = _derive(source, layout, start, end, recovered)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid_programme_layout") from exc
    for record in records:
        record["revision"] = _revision(record, binding)
        prior = previous.get(record["id"], {})
        record["state"] = (
            prior.get("state", "pending")
            if prior.get("revision") == record["revision"]
            else "pending"
        )
        record["history"] = prior.get("history", [])
    if _hash_file(source) != binding["source_sha256"]:
        raise ValueError("source_changed")
    _dependencies(binding)
    if recovery and _hash_file(Path(recovery)) != binding["recovery_sha256"]:
        raise ValueError("stale_recovery")
    if recovery and inspect_review(recovery) != recovered:
        raise ValueError("stale_recovery")
    _save(target, {"binding": binding, "layout": layout, "records": records})
    return inspect_programmes(target)


def inspect_programmes(catalog: str | Path) -> dict[str, Any]:
    """Re-derive all source evidence, dependencies and review revisions."""
    value = _read(Path(catalog))
    b = value["binding"]
    try:
        if _hash_file(Path(b["source_path"])) != b["source_sha256"] or b["settings"] != VERSION:
            raise ValueError("stale_programmes")
        _dependencies(b)
        recovered = inspect_review(b["recovery_path"]) if b["recovery_path"] else None
        if b["recovery_path"] and _hash_file(Path(b["recovery_path"])) != b["recovery_sha256"]:
            raise ValueError("stale_programmes")
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
            raise ValueError("stale_programmes")
        if _hash_file(Path(b["source_path"])) != b["source_sha256"]:
            raise ValueError("stale_programmes")
        _dependencies(b)
        if b["recovery_path"] and (
            inspect_review(b["recovery_path"]) != recovered
            or _hash_file(Path(b["recovery_path"])) != b["recovery_sha256"]
        ):
            raise ValueError("stale_programmes")
        if _read(Path(catalog)) != value:
            raise ValueError("stale_programmes")
    except (OSError, KeyError, TypeError):
        raise ValueError("stale_programmes") from None
    for r in value["records"]:
        r["eligible"] = r["state"] == "accepted" and r["coverage"] == "complete"
    return value


def review_programme(
    catalog: str | Path, record_id: str, revision: str, state: str
) -> dict[str, Any]:
    value = inspect_programmes(catalog)
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
    return inspect_programmes(catalog)


def validate_programmes(binding: dict[str, Any]) -> None:
    inspect_programmes(binding["catalog_path"])
    if _hash_file(Path(binding["catalog_path"])) != binding["catalog_sha256"]:
        raise ValueError("stale_programmes")


def _reference(name: str | None, records: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    if not name:
        return {"status": "unresolved", "reason": "not_stated"}

    def normalize(value: str) -> str:
        return " ".join(re.findall(r"\w+", value.casefold()))

    normalized = normalize(name)
    candidates = [
        i
        for i, record in enumerate(records)
        if normalized
        and any(
            normalized == normalize(label)
            for node in [record, *record.get("variations", [])]
            for label in [node["name"], *node["aliases"]]
            if label
        )
    ]
    if len(candidates) != 1:
        return {
            "status": "unresolved",
            "reason": "ambiguous" if candidates else "not_found",
            "candidates": [
                {"id": records[i]["id"], "source_id": records[i]["source_id"]} for i in candidates
            ],
        }
    record = records[candidates[0]]
    if not record["eligible"]:
        return {"status": "unresolved", "reason": "not_eligible"}
    return {"status": "resolved", "kind": kind, "record": record}


def programme_passages(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    from .policy import CHUNK_CHARS

    exercises: list[dict[str, Any]] = []
    for dependency in catalog["binding"]["exercise_catalogs"]:
        support = inspect_exercises(dependency["path"])
        exercises.extend(
            {**r, "source_id": support["binding"]["source_sha256"]} for r in support["records"]
        )
    methods = [
        {**r, "source_id": catalog["binding"]["source_sha256"]}
        for r in catalog["records"]
        if r["kind"] == "method"
    ]
    passages = []
    for original in catalog["records"]:
        if original["kind"] != "session" or not original["eligible"]:
            continue
        r = copy.deepcopy(original)
        r["method_reference"] = _reference(r["method"], methods, "method")
        for entry in r["entries"]:
            entry["exercise_reference"] = _reference(entry["exercise"], exercises, "exercise")
        texts = [
            segment["text"]
            for node in [r, *r["entries"]]
            for segments in node["citations"].values()
            for segment in segments
        ]
        texts.extend(
            segment["text"]
            for conflict in r["conflicts"]
            for claim in conflict["claims"]
            for segment in claim
        )
        for ref in [r["method_reference"], *[e["exercise_reference"] for e in r["entries"]]]:
            if ref["status"] == "resolved":
                support = ref["record"]
                for node in [support, *support.get("variations", [])]:
                    texts.extend(
                        s["text"] for segments in node["citations"].values() for s in segments
                    )
        text = "\n\n".join(texts)
        for i, window in enumerate(
            textwrap.wrap(
                text,
                width=CHUNK_CHARS,
                replace_whitespace=False,
                break_long_words=False,
                break_on_hyphens=False,
            )
        ):
            passages.append(
                {
                    "source_id": catalog["binding"]["source_sha256"],
                    "passage_id": _digest(_json([r["revision"], i]).encode()),
                    "text": text,
                    "retrieval_text": window,
                    "session": r,
                    "location": r["citations"]["programme"][0]["location"],
                    "eligible": True,
                    "review_status": "accepted",
                    "provenance": {"programme_revision": r["revision"], "chunk": i, **VERSION},
                }
            )
    return passages
