"""Selection views never replace or weaken validation of a complete recovery bundle."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def recovery_selection(
    bundle: dict[str, Any], fingerprint: str, start: int, end: int
) -> dict[str, Any]:
    section = bundle["section"]
    if (
        bundle["source_sha256"] != fingerprint
        or type(section["start"]) is not int
        or type(section["end"]) is not int
        or not section["start"] <= start <= end <= section["end"]
    ):
        raise ValueError("stale_recovery")
    if section == {"start": start, "end": end}:
        return bundle
    selected = deepcopy(bundle)
    for key in ("regions", "blocks", "gaps"):
        selected[key] = [r for r in bundle[key] if start <= r["pdf_page"] <= end]
    selected["unresolved_regions"] = sum(r["state"] != "accepted" for r in selected["regions"])
    selected["outside_selection"] = {
        "regions": [r for r in bundle["regions"] if not start <= r["pdf_page"] <= end],
        "gaps": [r for r in bundle["gaps"] if not start <= r["pdf_page"] <= end],
        "warning": "Outside-selection content is not imported or approved by this selection.",
    }
    selected["warnings"] = [*bundle["warnings"], selected["outside_selection"]["warning"]]
    return selected
