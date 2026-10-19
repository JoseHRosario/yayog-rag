# Private programme sessions

Use this workflow to map and visually review selected schedule cells, including
continuation pages and workout-method definitions. It is explicit operator layout,
not automatic table detection, whole-book coverage or answer generation. Keep PDFs,
layouts, catalogs, rendered pages and successful command output outside the repository.
All names and coordinates below are invented.

## Review workflow

1. Render the selected pages locally. Identify every programme heading, scope header,
   session column, ordered exercise row, pairing, quantity, pause, assistance, variation
   and note. Compare native extraction with the visible cells. A PDF text line can span
   columns: neither text order nor an automatic table extractor establishes a grid.
2. Write the layout below. Select every relevant cell and continuation explicitly.
   Include earlier programme headings and both source claims of any conflict. Select
   separate method records from their actual definition pages. Missing definitions may
   remain unresolved; missing schedule instructions must not be replaced by definitions.
3. Draft and inspect the catalog. Compare **every field, relationship and normalized
   value** with the rendered pages. Set complete coverage only for complete sessions or
   complete method definitions. Review each exact record revision individually.
4. Import and query an exact tuple; compare complete returned context with the source.
   An accepted revision records the operator's visual verification, not an automated
   semantic proof. Hashes and coordinates establish artifact identity, not correctness
   of a human's cell-to-session mapping.

```sh
umask 077
uv run cited draft-programmes "$PRIVATE_DATA/source.pdf" --start 1 --end 4 \
  --layout "$PRIVATE_DATA/layout.json" --catalog "$PRIVATE_DATA/programmes.json" \
  --exercise-catalog "$PRIVATE_DATA/exercises-a.json" \
  --exercise-catalog "$PRIVATE_DATA/exercises-b.json"
uv run cited inspect-programmes "$PRIVATE_DATA/programmes.json"
uv run cited review-programme "$PRIVATE_DATA/programmes.json" \
  --record cloud-1 --revision "$EXACT_REVIEWED_REVISION" --state accepted
# Repeat for each visually reviewed session and method. Also supports pending/rejected.
uv run cited import "$PRIVATE_DATA/source.pdf" --start 1 --end 4 \
  --index "$PRIVATE_DATA/sessions.sqlite" --model-dir "$MODEL_DIR" \
  --approve-sha256 "$APPROVED_SHA256" --approve-pages 1,2,3,4 \
  --programmes "$PRIVATE_DATA/programmes.json"
uv run cited search --index "$PRIVATE_DATA/sessions.sqlite" --model-dir "$MODEL_DIR" \
  --programme 'Cloud programme' --week 2 --day 1
uv run cited report --index "$PRIVATE_DATA/sessions.sqlite"
```

The optional question remains supported before the flags. Empty questions work with
all three filters. Supplying **any** filter requires all three: `None`, empty programme,
boolean/float week or day, week outside 1–104, or day outside 1–7 fails with
`invalid_session_filter`. Programme matching is full-string case-insensitive matching
with outer whitespace removed, or an explicitly reviewed session alias; no fuzzy name
fallback. Week/day use exact reviewed metadata. Valid unmatched tuples return
`no_evidence`; filtered queries against non-programme indexes cannot return prose or
exercises. An exact tuple selects independently of cosine score. Multiple distinct
matching catalog identities return `ambiguous` without choosing one, including collisions
with pending, partial or rejected records. A sole withheld identity returns `no_evidence`;
withheld instructions are never included in ambiguity metadata.

## Layout schema

The JSON root contains `records`. Each record has a unique nonempty `id`, `kind`
(`session` or `method`), `boundaries`, `fields`, optional `coverage` (defaults to
`partial`), `gaps` and `aliases` (reviewed strings). Boundaries are ordered page regions,
not inferred table rows. A source PDF and the inclusive draft/import section must
include every selected page, including earlier headings and definitions.

```json
{
  "records": [
    {
      "id": "cloud-1",
      "kind": "session",
      "coverage": "complete",
      "weeks": [1, 2],
      "day": 1,
      "boundaries": [
        {"pdf_page": 1, "bbox": [30, 30, 550, 110]},
        {"pdf_page": 3, "bbox": [30, 30, 550, 260]}
      ],
      "fields": {
        "programme": [{"pdf_page": 1, "bbox": [40, 35, 300, 60]}],
        "week_range": [{"pdf_page": 3, "bbox": [40, 35, 180, 60]}],
        "day": [{"pdf_page": 3, "bbox": [210, 35, 300, 60]}],
        "category": [{"pdf_page": 3, "bbox": [40, 65, 180, 90]}],
        "method": [{"pdf_page": 3, "bbox": [210, 65, 350, 90]}]
      },
      "entries": [
        {
          "pair": "A",
          "fields": {
            "pair": [{"pdf_page": 3, "bbox": [40, 110, 100, 140]}],
            "exercise": [{"pdf_page": 3, "bbox": [110, 110, 270, 140]}],
            "reps": [{"pdf_page": 3, "bbox": [280, 110, 350, 140]}],
            "pause": [{"pdf_page": 3, "bbox": [360, 110, 480, 140]}],
            "assistance": [{"pdf_page": 3, "bbox": [40, 190, 200, 215]}],
            "variations": [{"pdf_page": 3, "bbox": [210, 190, 480, 215]}]
          }
        },
        {
          "pair": "A",
          "fields": {
            "pair": [{"pdf_page": 3, "bbox": [40, 150, 100, 180]}],
            "exercise": [{"pdf_page": 3, "bbox": [110, 150, 270, 180]}],
            "reps": [{"pdf_page": 3, "bbox": [280, 150, 350, 180]}],
            "pause": [{"pdf_page": 3, "bbox": [360, 150, 480, 180]}]
          }
        }
      ],
      "conflicts": [{
        "claims": [
          [{"pdf_page": 1, "bbox": [40, 70, 400, 100]}],
          [{"pdf_page": 3, "bbox": [40, 35, 180, 60]}]
        ]
      }]
    },
    {
      "id": "alternating",
      "kind": "method",
      "coverage": "complete",
      "aliases": ["Alternate movements"],
      "boundaries": [{"pdf_page": 2, "bbox": [30, 30, 550, 250]}],
      "fields": {
        "name": [{"pdf_page": 2, "bbox": [40, 35, 500, 65]}],
        "definition": [{"pdf_page": 2, "bbox": [40, 75, 500, 230]}]
      }
    }
  ]
}
```

### Selectors and scope

- Every field is an **ordered array** of `{pdf_page, bbox}` selectors. Page numbers are
  one-based physical PDF pages; boxes are PyMuPDF page coordinates in points. Every box
  must be nonempty, on its page and inside a record boundary. Multiple selectors preserve
  wrapped cells and cross-page continuation in array order. Native cells are clipped by
  glyph geometry even when the underlying PDF line extends into neighboring columns;
  full-line containment is not required. The optional `cell: true` is documentary: all
  programme selectors use cell geometry.
- Optional `quote` selects an exact, unique, nonempty substring of selected native text;
  it cannot introduce new text. Original selected lines, pre-join text, exact quote,
  coordinates, source SHA-256 and rendered-crop SHA-256 remain in each citation. Explicit
  `joins: ["care-\nfully"]` work like exercise selectors: only the reviewed split is joined.
  A crop can be recreated privately with `page.get_pixmap(clip=pymupdf.Rect(bbox),
  alpha=False)`; `render_sha256` binds its PNG bytes. The renderer version is bound too.
- `weeks` is a nonempty list of distinct integers in 1–104; it lists exact session weeks,
  not an inferred range from a summary. `day` is an integer in 1–7. Both are explicitly
  reviewed normalization, bound to the revision alongside selected `week_range` and `day`
  evidence. Booleans/floats are rejected. Raw week/day text remains unchanged even when
  conflicting with another source statement.
- Session fields: `programme`, `week_range`, `day`, `category`, `method`, `notes`.
  The first three and at least one exercise entry are required for completeness.
  Entry fields: `exercise`, `pair`, `reps`, `pause`, `assistance`, `variations`, `notes`.
  Entry-array order is source exercise order; `pair` outside `fields` is the reviewed
  membership ID and requires selected pair evidence. Returned entries contain one-based
  `order`, `pair_id` and `within_pair_order`. An absent pairing remains `null`, not inferred
  from alternating positions. Duplicate exercise-cell selections within a session are
  rejected; empty phantom cells withhold the whole session even after acceptance.
- Missing optional fields remain `null`; no repetition counts, units, timings, assistance
  or variations are synthesized. A present but empty selected field creates a gap.
  Include stated instructions/notes even when their supporting catalog link is unavailable.
- Method fields: `name`, `definition`, both required for complete support. Select every
  paragraph of the definition, including continuations and qualifications. Reviewed
  `aliases` handle different source labels (including spacing differences) explicitly;
  whitespace merging or semantic synonym invention is not performed.
- Each conflict has `claims`, an array of at least two selector arrays. Results return
  both exact source statements and their individual citations under `source_conflict`;
  neither is rewritten or treated as an OCR correction. Decide session scope from the
  reviewed schedule, retaining the contradictory summary separately.

### Trust, dependencies and output

Whole-page prose eligibility is not required. Quality checks cover the declared record
regions: unrelated footers outside them cannot veto a clean table. Raw control/replacement
characters and unmapped glyphs remain blockers; structurally numeric/punctuation-only
cells are not prose. Existing prose/exercise policies are unchanged. If recovery is
needed, draft with `--recovery` and import with the same reviewed bundle. Recovery
selectors currently require complete recovered-line containment; use cell-sized recovery
regions. Unresolved recovery anywhere in a declared boundary withholds the whole record.

Only accepted, complete, gap-free sessions enter programme indexes, never adjacent raw
snippets or withheld drafts. Optional supporting exercise catalogs are independently
validated, can cover other sections or PDFs, and are pinned by exact hash at draft time.
Only accepted, complete exercise records and method records resolve. Normalized names
and reviewed aliases must match the whole normalized source name, never a known substring;
competing identities remain unresolved,
not first-match wins. A missing/ambiguous/unaccepted reference stays an explicit
`exercise_reference` or `method_reference` with `status: "unresolved"`, while preserving
its original schedule cell. Resolved references include the complete reviewed supporting
record, revision, source ID and field-level citations.

`passages[].session` contains the entire session, ordered entries, raw citations, scope,
links and conflicts. `text` includes the full selected session and resolved support;
bounded `retrieval_text` windows are only matching aids. The outer `location` anchors the
programme identity, possibly on an earlier page: individual field citations identify the
actual schedule/definition page. Printed labels come from PDF metadata or remain `null`.
Results are deduplicated per session before applying `limit`.

Inspect/report are private review surfaces containing draft content. Re-drafting changed
layout, metadata, aliases, source, renderer, recovery or exercise-catalog dependencies
resets affected revisions to pending while retaining decision history. Identical
re-drafts retain acceptance. Report/search revalidate source, recovery/crops, catalog and
support dependencies; query/import also validate after embedding. Review revocation,
missing/changed support or changed artifacts invalidate existing indexes until rebuild
(`stale_programmes`, or a dependency-specific stale error). Failed imports preserve the
previous index. Draft/import output paths cannot replace any declared source PDF, supporting
catalog, recovery bundle or crop in the complete dependency graph. Supply exercise
dependencies at draft time, not `--exercises` alongside
`--programmes` at import.

Repeated inspection in one Python process reuses only deterministic source derivation
(including cell extraction and rendered-crop digests). The process-private LRU retains
at most 16 results and 32 MiB of serialized result payloads; oversized results are not
retained. Keys bind actual source bytes, exact layout/range, complete validated recovery
state and processing/renderer versions. Inputs and returned results are isolated by deep
copy. Every inspection still reads and verifies the sealed catalog and exact record
revisions, hashes sources and dependencies, and validates recovery bundles and crop bytes;
these checks also run after derivation. Review decisions, eligibility and reports are never
memoized. No cache is written to disk, no index/settings identity changes, and separate CLI
processes still pay the initial extraction cost.

Python exports:

```python
draft_programmes(pdf, catalog, layout_dict, start, end,
                 recovery=None, exercise_catalogs=[catalog_a, catalog_b])
inspect_programmes(catalog)
review_programme(catalog, record_id, exact_revision, "accepted")
ingest(pdf, index, start, end, embedder, approval=approval, programmes=catalog)
search(index, "", embedder, programme="Cloud programme", week=2, day=1)
```
