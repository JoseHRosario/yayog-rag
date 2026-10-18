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


def search(
    index: str | Path,
    question: str,
    embedder: Embedder,
    *,
    limit: int = 5,
    source: str | Path | None = None,
) -> dict[str, Any]:
    if not 1 <= limit <= 50 or not question.strip() or len(question) > 4000:
        raise ValueError("invalid_query")
    db, meta, records = _load(index)
    try:
        if embedder.identity != meta["report"]["model"]:
            raise ValueError("incompatible_model")
        if source is not None:
            from .core import _hash_file

            if _hash_file(Path(source)) != meta["report"]["sha256"]:
                raise ValueError("stale_source")
        query = np.asarray(_vectors(embedder, [question])[0])
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
        ranked = sorted(candidates, key=lambda i: (-fused[i], i))[:limit]
        passages = [
            {
                **json.loads(records[i][0]),
                "score": fused[i],
                "semantic_score": float(scores[i]),
                "lexical_rank": lex_ranks.get(i),
                "semantic_rank": sem_ranks.get(i),
            }
            for i in ranked
        ]
        if "recovery" in meta["report"]:
            _validate_recovery(meta["report"]["recovery"])
        return {
            "status": "evidence" if passages else "no_evidence",
            "passages": passages,
            "source_id": meta["report"]["sha256"],
            "model": meta["report"]["model"],
            **(
                {
                    "coverage": {
                        "pages": meta["report"]["pages"],
                        "unresolved_regions": meta["report"]["recovery"]["unresolved_regions"],
                        "gaps": meta["report"]["recovery"]["gaps"],
                        "warnings": meta["report"]["recovery"]["warnings"],
                    }
                }
                if "recovery" in meta["report"]
                else {}
            ),
        }
    finally:
        db.close()
