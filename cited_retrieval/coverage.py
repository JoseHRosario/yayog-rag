"""Private, source-bound whole-page accounting; never grants retrieval eligibility."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pymupdf

from .core import _hash_file, _json, inspect_source
from .coverage_artifacts import snapshot
from .coverage_evaluation import evaluate
from .inference_client import private_path

_open: Any = pymupdf.open


STATUSES = ("trusted", "reviewed_exclusion", "unresolved")


def _private_path(path: str | Path) -> Path:
    try:
        return private_path(path)
    except ValueError:
        raise ValueError("unsafe_coverage_path") from None


def _decision(value: dict[str, Any]) -> None:
    if (
        value["status"] not in STATUSES
        or not isinstance(value.get("reason"), str)
        or not value["reason"].strip()
    ):
        raise ValueError("invalid_coverage_manifest")


def _counts(values: list[dict[str, Any]]) -> dict[str, int]:
    return {state: sum(v["status"] == state for v in values) for state in STATUSES}


def _bound_regions(regions: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
    observed = {r["key"]: r for r in expected}
    seen: set[str] = set()
    for region in regions:
        _decision(region)
        key = region["key"]
        if key in seen or key not in observed:
            raise ValueError("invalid_coverage_manifest")
        seen.add(key)
        original = observed[key]
        if any(region.get(k) != v for k, v in original.items() if k not in {"status", "reason"}):
            raise ValueError("invalid_coverage_manifest")
        if region["status"] == "trusted" and original["status"] != "trusted":
            raise ValueError("invalid_coverage_manifest")
    if seen != observed.keys():
        raise ValueError("invalid_coverage_manifest")


def _inventory(path: Path, source_sha256: str) -> dict[str, Any]:
    path = _private_path(path)
    try:
        data = path.read_bytes()
        value = json.loads(data)
        if value["source_sha256"] != source_sha256 or not isinstance(value["records"], list):
            raise ValueError("invalid_coverage_manifest")
        records = value["records"]
        if any(r.get("source_sha256", source_sha256) != source_sha256 for r in records):
            raise ValueError("invalid_coverage_manifest")
    except (KeyError, TypeError, AttributeError, UnicodeError, json.JSONDecodeError):
        raise ValueError("invalid_coverage_manifest") from None
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "records": records}


def draft_coverage(
    pdf: str | Path,
    audit: str | Path,
    *,
    indexes: list[str | Path] | None = None,
    exercise_catalogs: list[str | Path] | None = None,
    programme_catalogs: list[str | Path] | None = None,
    recovery_bundles: list[str | Path] | None = None,
    excerpt_catalogs: list[str | Path] | None = None,
    inventory: str | Path | None = None,
) -> dict[str, Any]:
    """Create an unresolved full-source operator manifest, outside the repository."""
    source, target = Path(pdf).resolve(), _private_path(audit)
    if target == source or target.exists():
        raise ValueError("unsafe_coverage_path")
    with _open(source) as doc:
        count = len(doc)
    preview = inspect_source(source, 1, count)
    specs = [
        {"kind": kind, "path": str(Path(p).resolve())}
        for kind, paths in (
            ("index", indexes),
            ("exercise", exercise_catalogs),
            ("programme", programme_catalogs),
            ("excerpt", excerpt_catalogs),
            ("recovery", recovery_bundles),
        )
        for p in paths or []
    ]
    specs = [
        {"kind": kind, "path": path}
        for kind, path in sorted({(s["kind"], s["path"]) for s in specs})
    ]
    artifacts = snapshot(specs, preview["sha256"])
    inventory_path = _private_path(inventory) if inventory is not None else None
    protected = {source, *(Path(f["path"]).resolve() for f in artifacts["files"])}
    if inventory_path is not None and inventory_path in protected | {target}:
        raise ValueError("unsafe_coverage_path")
    registered = _inventory(inventory_path, preview["sha256"]) if inventory_path else None
    if target in protected:
        raise ValueError("unsafe_coverage_path")
    manifest = {
        "schema": 1,
        "binding": {
            "source_path": str(source),
            "source_sha256": preview["sha256"],
            "specs": specs,
            "artifacts": artifacts,
            **({"inventory": registered} if registered else {}),
        },
        "pages": [
            {**p, "native_reason": p["reason"], "status": "unresolved", "reason": "unreviewed"}
            for p in preview["pages"]
        ],
        "records": artifacts["records"] + (registered["records"] if registered else []),
        "regions": artifacts["regions"],
        "evaluations": [],
        "gates": {g: {"status": "not_run", "evidence": ""} for g in ("offline", "private", "live")},
    }
    try:
        result = _validate_manifest(deepcopy(manifest), target)
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError):
        raise ValueError("invalid_coverage_manifest") from None
    except OSError:
        raise ValueError("stale_coverage") from None
    _publish_manifest(target, manifest)
    return result


def _publish_manifest(target: Path, manifest: dict[str, Any]) -> None:
    """Publish complete validated bytes atomically without replacing another writer."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".coverage-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(_json(manifest))
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise ValueError("unsafe_coverage_path") from None
    finally:
        temporary.unlink(missing_ok=True)


def inspect_coverage(audit: str | Path) -> dict[str, Any]:
    """Revalidate current inputs and report conservative coverage, not content approval."""
    try:
        target = _private_path(audit)
        data = target.read_bytes()
        result = _validate_manifest(json.loads(data), target)
        if _hash_file(target) != hashlib.sha256(data).hexdigest():
            raise ValueError("stale_coverage")
        return result
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError):
        raise ValueError("invalid_coverage_manifest") from None
    except OSError:
        raise ValueError("stale_coverage") from None


def _validate_manifest(manifest: dict[str, Any], audit: Path) -> dict[str, Any]:
    binding = manifest["binding"]
    if _hash_file(Path(binding["source_path"])) != binding["source_sha256"]:
        raise ValueError("stale_coverage")
    with _open(binding["source_path"]) as doc:
        preview = inspect_source(binding["source_path"], 1, len(doc))
    pages = manifest["pages"]
    if manifest["schema"] != 1 or len(pages) != len(preview["pages"]):
        raise ValueError("invalid_coverage_manifest")
    for actual, expected in zip(pages, preview["pages"], strict=True):
        if (
            type(actual["pdf_page"]) is not int
            or actual["pdf_page"] != expected["pdf_page"]
            or actual["printed_label"] != expected["printed_label"]
            or actual["native_reason"] != expected["reason"]
        ):
            raise ValueError("invalid_coverage_manifest")
        _decision(actual)
        if actual["status"] != "unresolved" and (
            actual.get("whole_page_reviewed") is not True
            or actual.get("inventory_complete") is not True
        ):
            raise ValueError("invalid_coverage_manifest")
    current = snapshot(binding["specs"], binding["source_sha256"])
    if current != binding["artifacts"]:
        raise ValueError("stale_coverage")
    records = manifest["records"]
    declared = {r["key"]: r["status"] for r in records}
    observed = {r["key"]: r for r in current["records"]}
    registered = binding.get("inventory")
    manual = {}
    if registered is not None:
        inventory_path = _private_path(registered["path"])
        protected = {
            Path(binding["source_path"]).resolve(),
            *(Path(f["path"]).resolve() for f in current["files"]),
        }
        if inventory_path in protected | {_private_path(audit)}:
            raise ValueError("unsafe_coverage_path")
        if _hash_file(inventory_path) != registered["sha256"]:
            raise ValueError("stale_coverage")
        if _inventory(inventory_path, binding["source_sha256"]) != registered:
            raise ValueError("stale_coverage")
        manual = {r["key"]: r for r in registered["records"]}
        if len(manual) != len(registered["records"]) or manual.keys() & observed.keys():
            raise ValueError("invalid_coverage_manifest")
    seen: set[str] = set()
    for record in records:
        _decision(record)
        if not isinstance(record["kind"], str) or not record["kind"].strip():
            raise ValueError("invalid_coverage_manifest")
        for field in ("dependencies", "gaps"):
            codes = record.get(field, [])
            if not isinstance(codes, list) or any(
                not isinstance(c, str) or not c.strip() for c in codes
            ):
                raise ValueError("invalid_coverage_manifest")
        if not isinstance(record["pdf_pages"], list) or len(set(record["pdf_pages"])) != len(
            record["pdf_pages"]
        ):
            raise ValueError("invalid_coverage_manifest")
        key = record["key"]
        if (
            key not in observed
            and record.get("source_sha256", binding["source_sha256"]) != binding["source_sha256"]
        ):
            raise ValueError("invalid_coverage_manifest")
        if not isinstance(key, str) or not key.strip() or key in seen:
            raise ValueError("invalid_coverage_manifest")
        seen.add(key)
        if key not in observed and key not in manual:
            raise ValueError("invalid_coverage_manifest")
        expected = observed[key] if key in observed else manual[key]
        if {k: v for k, v in record.items() if k not in {"status", "reason"}} != {
            k: v for k, v in expected.items() if k not in {"status", "reason"}
        }:
            raise ValueError("invalid_coverage_manifest")
        if key in observed:
            if record["status"] == "trusted" and expected["status"] != "trusted":
                raise ValueError("invalid_coverage_manifest")
        elif record["status"] == "trusted":
            evidence = record.get("evidence_keys", [])
            if (
                record.get("whole_record_reviewed") is not True
                or not isinstance(evidence, list)
                or not evidence
                or record.get("dependencies")
                or record.get("gaps")
                or any(
                    e not in observed
                    or observed[e]["status"] != "trusted"
                    or declared.get(e) != "trusted"
                    for e in evidence
                )
                or any(observed[e]["source_sha256"] != binding["source_sha256"] for e in evidence)
                or set(record["pdf_pages"])
                != {p for e in evidence for p in observed[e]["pdf_pages"]}
            ):
                raise ValueError("invalid_coverage_manifest")
        if (
            not record["pdf_pages"]
            or any(type(n) is not int or not 1 <= n <= len(pages) for n in record["pdf_pages"])
        ) and record.get("source_sha256", binding["source_sha256"]) == binding["source_sha256"]:
            raise ValueError("invalid_coverage_manifest")
    if (observed.keys() | manual.keys()) != seen:
        raise ValueError("invalid_coverage_manifest")
    regions = manifest["regions"]
    _bound_regions(regions, current["regions"])
    for page in pages:
        number = page["pdf_page"]
        blockers = [
            r["key"]
            for r in records
            if r["status"] == "unresolved"
            and r.get("source_sha256", binding["source_sha256"]) == binding["source_sha256"]
            and number in r["pdf_pages"]
        ]
        blockers.extend(
            r["key"]
            for r in regions
            if r["status"] == "unresolved"
            and r["source_sha256"] == binding["source_sha256"]
            and r["pdf_page"] == number
        )
        if page.get("gaps") or blockers:
            page["declared_status"] = page["status"]
            page["status"] = "unresolved"
            page["blockers"] = blockers

    complete = (
        all(r["status"] != "unresolved" for r in regions)
        and all(p["status"] != "unresolved" for p in pages)
        and all(r["status"] != "unresolved" for r in records)
    )
    evaluated, gates = evaluate(manifest)
    _revalidate_identities(manifest)
    return {
        "binding": binding,
        "source_sha256": binding["source_sha256"],
        "pages": manifest["pages"],
        "records": manifest["records"],
        "regions": regions,
        "coverage_complete": complete,
        "evaluation_complete": evaluated,
        "evaluations": manifest["evaluations"],
        "delivery_complete": complete and evaluated and all(s == "passed" for s in gates.values()),
        "counts": {
            "pages": _counts(pages),
            "records": _counts(records),
            "regions": _counts(regions),
        },
        "gates": gates,
    }


def _revalidate_identities(manifest: dict[str, Any]) -> None:
    """Reject input changes during evaluation; this never substitutes for trust validation."""
    binding = manifest["binding"]
    for artifact in binding["artifacts"]["files"]:
        if _hash_file(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("stale_coverage")
    registered = binding.get("inventory")
    if (
        registered is not None
        and _hash_file(_private_path(registered["path"])) != registered["sha256"]
    ):
        raise ValueError("stale_coverage")
    for value in [*manifest["gates"].values(), *manifest["evaluations"]]:
        if value["status"] != "not_run":
            evidence = value["evidence"]
            try:
                if _hash_file(Path(evidence["path"])) != evidence["sha256"]:
                    raise ValueError("stale_coverage_evidence")
            except OSError:
                raise ValueError("stale_coverage_evidence") from None
    if _hash_file(Path(binding["source_path"])) != binding["source_sha256"]:
        raise ValueError("stale_coverage")
