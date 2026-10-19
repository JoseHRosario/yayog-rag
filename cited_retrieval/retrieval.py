"""Hybrid evidence retrieval, with no generated answers or fallback passages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .core import _load, _validate_recovery, _vectors
from .embedder import Embedder
from .policy import (
    CANDIDATE_LIMIT,
    LEXICAL_SEMANTIC_FLOOR,
    LEXICAL_STOPWORDS,
    RRF_CONSTANT,
    SEMANTIC_THRESHOLD,
)

_UNSET = object()


def search(
    index: str | Path,
    question: str,
    embedder: Embedder,
    *,
    limit: int = 5,
    source: str | Path | None = None,
    programme: object = _UNSET,
    week: object = _UNSET,
    day: object = _UNSET,
) -> dict[str, Any]:
    filtered = any(v is not _UNSET for v in (programme, week, day))
    if filtered and (
        not isinstance(programme, str)
        or not programme.strip()
        or type(week) is not int
        or not 1 <= week <= 104
        or type(day) is not int
        or not 1 <= day <= 7
    ):
        raise ValueError("invalid_session_filter")
    if not 1 <= limit <= 50 or (not question.strip() and not filtered) or len(question) > 4000:
        raise ValueError("invalid_query")
    db, meta, records = _load(index)
    try:
        if embedder.identity != meta["report"]["model"]:
            raise ValueError("incompatible_model")
        if source is not None:
            from .core import _hash_file

            if _hash_file(Path(source)) != meta["report"]["sha256"]:
                raise ValueError("stale_source")
        try:
            query = np.asarray(
                _vectors(embedder, [question if question.strip() else str(programme)])[0]
            )
        except BaseException:
            checked, checked_meta, _ = _load(index)
            checked.close()
            if checked_meta != meta:
                raise ValueError("stale_corpus")
            if embedder.identity != meta["report"]["model"]:
                raise ValueError("incompatible_model")
            raise
        if embedder.identity != meta["report"]["model"]:
            raise ValueError("incompatible_model")
        if len(query) != meta["report"]["dimensions"]:
            raise ValueError("incompatible_dimensions")
        documents = np.asarray([json.loads(r[1]) for r in records])
        scores = documents @ query / (np.linalg.norm(documents, axis=1) * np.linalg.norm(query))
        tokens = [
            token
            for token in re.findall(r"\w+", question.casefold(), flags=re.UNICODE)
            if token not in LEXICAL_STOPWORDS
        ]
        match = " OR ".join('"' + t + '"' for t in tokens)
        lexical = (
            [
                r[0] - 1
                for r in db.execute(
                    "SELECT rowid FROM terms WHERE terms MATCH ? ORDER BY bm25(terms), rowid LIMIT ?",
                    (match, CANDIDATE_LIMIT),
                )
                if scores[r[0] - 1] >= LEXICAL_SEMANTIC_FLOOR
            ]
            if match
            else []
        )
        lex_ranks = {idx: rank for rank, idx in enumerate(lexical, 1)}
        semantic = sorted(
            (i for i in range(len(records)) if scores[i] >= SEMANTIC_THRESHOLD),
            key=lambda i: (-scores[i], i),
        )[:CANDIDATE_LIMIT]
        sem_ranks = {idx: rank for rank, idx in enumerate(semantic, 1)}
        candidates = set(lex_ranks) | set(sem_ranks)
        fused = {
            i: (1 / (RRF_CONSTANT + lex_ranks[i]) if i in lex_ranks else 0.0)
            + (1 / (RRF_CONSTANT + sem_ranks[i]) if i in sem_ranks else 0.0)
            for i in candidates
        }
        ranked = sorted(candidates, key=lambda i: (-fused[i], i))
        ambiguous = []
        exercise_coverage = []
        if "exercises" in meta["report"]:
            from .exercises import coverage_summary, named_candidates, validate_exercises

            catalog_records = meta["report"]["exercises"]["records"]
            # Resolve identity before eligibility. Withheld aliases can block
            # unsafe routing, but never activate their instructions as evidence.
            known = named_candidates(question, [{"exercise": r} for r in catalog_records])
            if len(known) > 1:
                ambiguous = [
                    {
                        "id": catalog_records[i]["id"],
                        "name": catalog_records[i]["name"],
                        "eligible": catalog_records[i]["eligible"],
                        "state": catalog_records[i]["state"],
                        "location": (
                            catalog_records[i]["citations"]["name"][0]["location"]
                            if catalog_records[i]["citations"]["name"]
                            else None
                        ),
                    }
                    for i in known
                ]
                ranked = []
            elif known:
                parent = catalog_records[known[0]]
                nodes = [parent, *parent["variations"]]
                matched_nodes = named_candidates(
                    question, [{"exercise": {**node, "variations": []}} for node in nodes]
                )
                node_ids = {nodes[i]["id"] for i in matched_nodes}
                matching = [
                    i
                    for i, record in enumerate(records)
                    if json.loads(record[0])["exercise"]["id"] == parent["id"]
                    and json.loads(record[0])["provenance"]["matched_node_id"] in node_ids
                ]
                # Retain relevance within the explicitly named node. Append
                # matching windows as a fallback when semantic/lexical cutoffs
                # exclude an exact accepted identity, never the parent's first
                # unrelated window merely because it precedes the variation.
                ranked = [i for i in ranked if i in matching] + [
                    i for i in matching if i not in ranked
                ]
            validate_exercises(meta["report"]["exercises"])
            exercise_coverage = coverage_summary(meta["report"]["exercises"]["records"])
            seen: set[str] = set()
            parents = []
            for i in ranked:
                parent_id = json.loads(records[i][0])["exercise"]["id"]
                if parent_id not in seen:
                    parents.append(i)
                    seen.add(parent_id)
            ranked = parents
        if "programmes" in meta["report"]:
            from .programmes import validate_programmes

            validate_programmes(meta["report"]["programmes"])
            if filtered:
                matching_sessions = [
                    session
                    for session in meta["report"]["programmes"]["records"]
                    if session["kind"] == "session"
                    and str(programme).strip().casefold()
                    in {
                        n.strip().casefold()
                        for n in [session["programme"], *session["aliases"]]
                        if n is not None
                    }
                    and week in session["weeks"]
                    and day == session["day_number"]
                ]
                identities = {session["id"] for session in matching_sessions}
                if len(identities) > 1:
                    ambiguous = [{"id": rid} for rid in sorted(identities)]
                    ranked = []
                else:
                    ranked = [
                        i
                        for i, record in enumerate(records)
                        if json.loads(record[0])["session"]["id"] in identities
                    ]
            seen_sessions: set[str] = set()
            unique_sessions = []
            for i in ranked:
                rid = json.loads(records[i][0])["session"]["id"]
                if rid not in seen_sessions:
                    unique_sessions.append(i)
                    seen_sessions.add(rid)
            ranked = unique_sessions
        elif filtered:
            ranked = []
            ambiguous = []
        if "excerpts" in meta["report"]:
            seen_excerpts: set[str] = set()
            unique_excerpts = []
            for i in ranked:
                rid = json.loads(records[i][0])["excerpt"]["id"]
                if rid not in seen_excerpts:
                    unique_excerpts.append(i)
                    seen_excerpts.add(rid)
            ranked = unique_excerpts
        ranked = ranked[:limit]
        passages = [
            {
                **json.loads(records[i][0]),
                "score": fused.get(i, 0.0),
                "semantic_score": float(scores[i]),
                "lexical_rank": lex_ranks.get(i),
                "semantic_rank": sem_ranks.get(i),
            }
            for i in ranked
        ]
        # Expand only from the already validated, eligible prose corpus. Never
        # extract fresh page text or cross into structured catalog siblings.
        page_context: list[dict[str, Any]] = []
        prose = "exercises" not in meta["report"] and "programmes" not in meta["report"]
        if prose:
            pages = {p["location"]["pdf_page"] for p in passages}
            matched_ids = {p["passage_id"] for p in passages}
            page_context = sorted(
                (
                    p
                    for record in records
                    for p in [json.loads(record[0])]
                    if p["eligible"]
                    and p["location"]["pdf_page"] in pages
                    and p["passage_id"] not in matched_ids
                ),
                key=lambda p: (p["location"]["pdf_page"], p["location"]["ordinal"]),
            )
            if "excerpts" in meta["report"]:
                matched_records = {p["excerpt"]["id"] for p in passages}
                unique_context: dict[str, dict[str, Any]] = {}
                for passage in page_context:
                    rid = passage["excerpt"]["id"]
                    if rid not in matched_records and rid not in unique_context:
                        unique_context[rid] = passage
                page_context = list(unique_context.values())
        if "recovery" in meta["report"]:
            _validate_recovery(meta["report"]["recovery"])
        # Revalidate the complete dependency graph after inference, including source changes.
        checked, checked_meta, _ = _load(index)
        checked.close()
        if checked_meta != meta:
            raise ValueError("stale_corpus")
        return {
            "status": "ambiguous" if ambiguous else "evidence" if passages else "no_evidence",
            **({"candidates": ambiguous} if ambiguous else {}),
            **({"exercise_coverage": exercise_coverage} if "exercises" in meta["report"] else {}),
            "passages": passages,
            **(
                {"page_context": page_context, "context_scope": "eligible_indexed_same_page_prose"}
                if prose
                else {}
            ),
            "source_id": meta["report"]["sha256"],
            "model": meta["report"]["model"],
            **(
                {
                    "excerpt_coverage": {
                        "records": [
                            {
                                k: r[k]
                                for k in ("id", "revision", "state", "eligible", "coverage", "gaps")
                            }
                            for r in meta["report"]["excerpts"]["records"]
                        ],
                        "warning": "Operator-selected excerpts are not whole-page or semantic completeness proof.",
                    }
                }
                if "excerpts" in meta["report"]
                else {}
            ),
            **(
                {
                    "coverage": {
                        "pages": meta["report"]["pages"],
                        "unresolved_regions": meta["report"]["recovery"]["unresolved_regions"],
                        "gaps": [
                            {
                                k: v
                                for k, v in gap.items()
                                if k in {"pdf_page", "bbox", "reason", "regions"}
                            }
                            for gap in meta["report"]["recovery"]["gaps"]
                        ]
                        if "exercises" in meta["report"] or "programmes" in meta["report"]
                        else meta["report"]["recovery"]["gaps"],
                        "warnings": meta["report"]["recovery"]["warnings"],
                    }
                }
                if "recovery" in meta["report"]
                else {}
            ),
        }
    finally:
        db.close()
