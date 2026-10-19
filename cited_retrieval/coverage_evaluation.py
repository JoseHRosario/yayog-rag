"""Owner-recorded evaluations are distinct from automated/model acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf

from .core import _json

CATEGORIES = {"exact_name", "paraphrase", "variation", "session_filter", "unsupported", "conflict"}
_open: Any = pymupdf.open


def _evidence(value: dict[str, Any]) -> bytes:
    evidence = value["evidence"]
    try:
        data = Path(evidence["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != evidence["sha256"]:
            raise ValueError("stale_coverage_evidence")
    except OSError:
        raise ValueError("stale_coverage_evidence") from None
    return data


def _case_evidence(case: dict[str, Any], source_sha256: str) -> None:
    try:
        envelope = json.loads(_evidence(case))
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("coverage_case_binding_mismatch") from None
    expected = {
        **{k: v for k, v in case.items() if k not in {"status", "claim_support", "evidence"}},
        "source_sha256": source_sha256,
    }
    if (
        not isinstance(envelope, dict)
        or _json(envelope.get("case")) != _json(expected)
        or "result" not in envelope
    ):
        raise ValueError("coverage_case_binding_mismatch")


def evaluate(manifest: dict[str, Any]) -> tuple[bool, dict[str, str]]:
    """Validate evidence identity and exact native source locations, not entailment."""
    gates = manifest["gates"]
    if set(gates) != {"offline", "private", "live"}:
        raise ValueError("invalid_coverage_manifest")
    for gate in gates.values():
        if gate["status"] not in {"passed", "failed", "not_run"}:
            raise ValueError("invalid_coverage_manifest")
        if gate["status"] != "not_run":
            _evidence(gate)
    passed = set()
    ids: set[str] = set()
    with _open(manifest["binding"]["source_path"]) as doc:
        for case in manifest["evaluations"]:
            category = case["category"]
            if (
                category not in CATEGORIES
                or not isinstance(case["id"], str)
                or not case["id"]
                or case["id"] in ids
                or not isinstance(case["question"], str)
                or not case["question"].strip()
                or case["status"] not in {"passed", "failed", "not_run"}
            ):
                raise ValueError("invalid_coverage_manifest")
            ids.add(case["id"])
            expected = (
                "abstain"
                if category == "unsupported"
                else "disclose_conflict"
                if category == "conflict"
                else "answer"
            )
            locations = case["locations"]
            minimum = 0 if category == "unsupported" else 2 if category == "conflict" else 1
            if case["expected_behavior"] != expected or len(locations) < minimum:
                raise ValueError("invalid_coverage_manifest")
            for location in locations:
                number, quote = location["pdf_page"], location["quote"]
                if type(number) is not int or not 1 <= number <= len(doc):
                    raise ValueError("invalid_evaluation_location")
                page = doc[number - 1]
                if (
                    not isinstance(quote, str)
                    or not quote.strip()
                    or quote not in page.get_text()
                    or location["printed_label"] != (page.get_label() or None)
                ):
                    raise ValueError("invalid_evaluation_location")
            if category == "conflict" and locations[0] == locations[1]:
                raise ValueError("invalid_evaluation_location")
            if case["status"] != "not_run":
                _case_evidence(case, manifest["binding"]["source_sha256"])
                if not isinstance(case["claim_support"], str) or not case["claim_support"].strip():
                    raise ValueError("invalid_coverage_manifest")
            if case["status"] == "passed":
                passed.add(category)
    complete = passed == CATEGORIES and all(
        c["status"] == "passed" for c in manifest["evaluations"]
    )
    return complete, {k: v["status"] for k, v in gates.items()}
