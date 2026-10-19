"""Model-only copies of structured records; originals remain bound and retained."""

from typing import Any


def _segment(segment: Any) -> Any:
    # Only this known extraction trace is redundant with approved field text/location.
    # Unknown fields (including qualifications/guidance) are deliberately preserved.
    if isinstance(segment, dict) and {"text", "location", "source_id"} <= segment.keys():
        return {key: value for key, value in segment.items() if key != "lines"}
    return segment


def exercise_prompt(record: dict[str, Any]) -> dict[str, Any]:
    """Keep every field and scalar citation; remove exact duplicate array citations."""
    projected = dict(record)
    citations = dict(record.get("citations", {}))
    for field in ("instructions", "warnings", "references"):
        if field in record:
            projected[field] = [_segment(segment) for segment in record[field]]
            if citations.get(field) == record[field]:
                del citations[field]
    if "citations" in record:
        projected["citations"] = {
            field: (
                [_segment(segment) for segment in segments]
                if isinstance(segments, list)
                else segments
            )
            for field, segments in citations.items()
        }
    if "variations" in record:
        projected["variations"] = [exercise_prompt(child) for child in record["variations"]]
    return projected


def passage_prompt(passage: dict[str, Any]) -> dict[str, Any]:
    """Project exercises directly and inside resolved programme entries only."""
    projected = dict(passage)
    if "exercise" in passage:
        projected["exercise"] = exercise_prompt(passage["exercise"])
    if "session" in passage:
        session = dict(passage["session"])
        if "entries" in session:
            entries = []
            for original in session["entries"]:
                entry = dict(original)
                reference = entry.get("exercise_reference")
                if isinstance(reference, dict) and isinstance(reference.get("record"), dict):
                    entry["exercise_reference"] = {
                        **reference,
                        "record": exercise_prompt(reference["record"]),
                    }
                entries.append(entry)
            session["entries"] = entries
        projected["session"] = session
    return projected


def coverage_prompt(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count routine accepted inventory; retain every unavailable/unknown disclosure."""
    routine_keys = {"id", "revision", "state", "eligible", "coverage", "gaps"}
    routine = [
        record
        for record in records
        if record.keys() <= routine_keys
        and record.get("state") == "accepted"
        and record.get("eligible") is True
        and record.get("coverage") == "complete"
        and record.get("gaps") == []
    ]
    return {
        "record_count": len(records),
        "accepted_complete_count": len(routine),
        "unavailable_or_nonroutine": [record for record in records if record not in routine],
    }
