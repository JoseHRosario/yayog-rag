"""Validated artifact graph and record inventory for private coverage manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import _hash_file, _load, report
from .excerpts import inspect_excerpts
from .excerpts import protected_artifacts as excerpt_artifacts
from .exercises import inspect_exercises
from .programmes import inspect_programmes, programme_passages, protected_artifacts
from .recovery import inspect_review


def snapshot(specs: list[dict[str, str]], source_sha256: str) -> dict[str, Any]:
    """Pin every input, including transitive recovery crops and support catalogs."""
    files: set[Path] = set()
    catalogs: dict[Path, str] = {}
    indexes = []
    prose_records = []
    recoveries: set[Path] = set()
    for spec in specs:
        path = Path(spec["path"]).resolve()
        files.add(path)
        kind = spec["kind"]
        if kind == "index":
            result = report(path)
            if result["sha256"] != source_sha256:
                raise ValueError("invalid_coverage_manifest")
            indexes.append(
                {
                    "path": str(path),
                    "corpus_sha256": result["corpus_sha256"],
                    "model": result["model"],
                    "settings": result["settings"],
                }
            )
            for key, catalog_kind in (
                ("exercises", "exercise"),
                ("programmes", "programme"),
                ("excerpts", "excerpt"),
            ):
                if key in result:
                    catalogs[Path(result[key]["catalog_path"]).resolve()] = catalog_kind
            if (
                "exercises" not in result
                and "programmes" not in result
                and "excerpts" not in result
            ):
                db, _, rows = _load(path)
                db.close()
                for payload, _ in rows:
                    passage = json.loads(payload)
                    prose_records.append(
                        {
                            "key": str(path) + "#" + passage["passage_id"],
                            "kind": "passage",
                            "source_sha256": passage["source_id"],
                            "pdf_pages": [passage["location"]["pdf_page"]],
                            "location": passage["location"],
                            "revision": passage["passage_id"],
                            "review_state": passage["review_status"],
                            "eligible": passage["eligible"],
                            "gaps": [],
                            "dependencies": [],
                            "status": "trusted",
                            "reason": "eligible_index_passage_not_whole_page",
                        }
                    )
            if "recovery" in result:
                recovery = Path(result["recovery"]["bundle_path"]).resolve()
                recovered = inspect_review(recovery)
                recoveries.add(recovery)
                files.add(recovery)
                files.add(Path(recovered["source_path"]).resolve())
                files.update((recovery.parent / r["crop"]).resolve() for r in recovered["regions"])
        elif kind == "recovery":
            recoveries.add(path)
        elif kind in {"exercise", "programme", "excerpt"}:
            catalogs[path] = kind
        else:
            raise ValueError("invalid_coverage_manifest")
    records = prose_records
    visited: set[Path] = set()
    while catalogs:
        path = min(catalogs)
        kind = catalogs.pop(path)
        if path in visited:
            continue
        visited.add(path)
        catalog = (
            inspect_excerpts(path)
            if kind == "excerpt"
            else inspect_exercises(path)
            if kind == "exercise"
            else inspect_programmes(path)
        )
        binding = catalog["binding"]
        files.add(path)
        files.update(
            excerpt_artifacts(binding) if kind == "excerpt" else protected_artifacts(binding)
        )
        if binding.get("recovery_path"):
            recoveries.add(Path(binding["recovery_path"]).resolve())
        for dependency in binding.get("exercise_catalogs", []):
            catalogs[Path(dependency["path"]).resolve()] = "exercise"
        sessions = (
            {p["session"]["id"]: p["session"] for p in programme_passages(catalog)}
            if kind == "programme"
            else {}
        )
        for record in catalog["records"]:
            dependencies = _unresolved(sessions.get(record["id"], record))
            records.append(
                {
                    "key": str(path) + "#" + record["id"],
                    "kind": record.get("kind", "excerpt" if kind == "excerpt" else "exercise"),
                    "source_sha256": binding["source_sha256"],
                    "pdf_pages": sorted({b["pdf_page"] for b in record["boundaries"]}),
                    "revision": record["revision"],
                    "review_state": record["state"],
                    "eligible": record["eligible"],
                    "gaps": record["gaps"],
                    "dependencies": dependencies,
                    "status": "trusted"
                    if record["eligible"] and not dependencies
                    else "unresolved",
                    "reason": "validated_catalog"
                    if record["eligible"]
                    else "ineligible_catalog_record",
                }
            )
    regions = []
    for path in sorted(recoveries):
        recovered = inspect_review(path)
        files.add(path)
        files.add(Path(recovered["source_path"]).resolve())
        for region in recovered["regions"]:
            files.add((path.parent / region["crop"]).resolve())
            regions.append(
                {
                    "key": str(path) + "#" + region["id"],
                    "region_id": region["id"],
                    "source_sha256": recovered["source_sha256"],
                    "pdf_page": region["pdf_page"],
                    "bbox": region["bbox"],
                    "revision": region["revision"],
                    "review_state": region["state"],
                    "status": "trusted" if region["state"] == "accepted" else "unresolved",
                    "reason": "recovery_review"
                    if region["state"] == "accepted"
                    else "unreviewed_content",
                }
            )
    return {
        "regions": regions,
        "files": [{"path": str(p), "sha256": _hash_file(p)} for p in sorted(files)],
        "indexes": sorted(indexes, key=lambda r: r["path"]),
        "records": sorted(records, key=lambda r: r["key"]),
    }


def _unresolved(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"exercise_reference", "method_reference"} and child["status"] != "resolved":
                result.append(key + ":" + child.get("reason", "unresolved"))
            else:
                result.extend(_unresolved(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_unresolved(child))
    return sorted(set(result))
