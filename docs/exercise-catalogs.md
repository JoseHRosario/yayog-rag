# Private exercise catalogs

This is explicit, operator-reviewed extraction, not heuristic chapter parsing. No model
server, layout inference, generated replacement instructions or image interpretation is
used. All artifacts and successful command output are **private**: keep them outside the
repository and CI logs. Examples below contain only invented names and coordinates.

## Layout schema

Write a private JSON object with a `records` array. Each parent has a unique `id`,
`boundaries` (all original-page regions containing its identity, metadata, instructions,
continuations, warnings, references, and variations), `fields`, and optional `aliases`,
`variations`, `images`, `coverage` and `gaps`.

```json
{
  "records": [{
    "id": "cloud",
    "coverage": "complete",
    "aliases": ["sky push"],
    "boundaries": [
      {"pdf_page": 1, "bbox": [40, 30, 540, 300]},
      {"pdf_page": 2, "bbox": [40, 30, 540, 250]}
    ],
    "fields": {
      "name": [{"pdf_page": 1, "bbox": [45, 40, 500, 65]}],
      "muscles": [{"pdf_page": 1, "bbox": [45, 70, 500, 95], "quote": "Arms"}],
      "difficulty": [{"pdf_page": 1, "bbox": [45, 70, 500, 95], "quote": "Easy"}],
      "instructions": [
        {"pdf_page": 1, "bbox": [45, 100, 500, 180]},
        {"pdf_page": 2, "bbox": [45, 40, 500, 80]}
      ],
      "warnings": [{"pdf_page": 2, "bbox": [45, 85, 500, 110]}]
    },
    "variations": [{
      "id": "wall",
      "relation": "easier",
      "fields": {
        "name": [{"pdf_page": 2, "bbox": [45, 120, 500, 145]}],
        "instructions": [{"pdf_page": 2, "bbox": [45, 150, 500, 190]}]
      }
    }]
  }]
}
```

- Page numbers are one-based physical PDF pages; boxes use PDF points in PyMuPDF page
  coordinates. Every box must be nonempty and contained by the page. Every field/image
  selector must be inside a declared parent boundary on that page. Boundaries must stop
  before an unrelated next movement; select every continuation explicitly. Variations
  belong only to their declared parent, with separately selected fields. A child's optional
  `boundaries` further constrain its selectors and must themselves lie inside parent regions;
  absent child boundaries inherit the parent regions. Explicit child `coverage: "partial"`
  or child `gaps` withhold the whole parent; otherwise parent completeness covers its children.
- Fields are `name`, `category`, `muscles`, `difficulty`, `chapter`, `instructions`,
  `warnings`, `references`. Every field takes an **ordered array** of selectors. Text is
  derived from complete native/recovered lines contained in each selector, in page reading
  order; array order preserves cross-page continuation. Include entire line bounding boxes,
  not just visible glyph pixels. Do not use overlapping instruction selectors.
- Optional `quote` selects an **exact, nonempty, unique substring** of the selected source
  lines (including literal newlines). It cannot introduce or paraphrase source text. Use it
  to separate muscles/difficulty sharing one line. Full selected lines remain in provenance.
  Absent scalar attributes become `null`; absent warning/reference arrays remain empty,
  never inferred. A missing parent identity, missing instructions or empty selectors create
  gaps. An unnamed variation remains eligible when its source-backed instructions and
  parent are complete; its name stays `null` rather than inventing a title.
- `chapter` is also source-selected. Its exact value is attached to citation locations;
  children inherit the parent's chapter unless they have their own chapter selector.
  Missing chapter and printed-label metadata stay `null`. No chapter offset is guessed.
- Line-end hyphens/newlines are preserved. To join one visually reviewed split, add e.g.
  `"joins": ["care-\nfully"]` to that selector. Each string must occur exactly once and
  contain `-\n`; only that hyphen/newline is removed. Original text, selected source lines,
  explicit joins and revised output remain inspectable. Joins change the review revision.
- Aliases are operator-validated strings, not auto-generated synonyms. Alias edits invalidate
  acceptance. Search normalizes case/word punctuation; exact parent/variation names and
  aliases select the parent. A longer complete name suppresses a shorter alias contained
  in that same mention; separate mentions are not collapsed. Shared names/aliases across
  distinct parents return `status: "ambiguous"`, empty passages and separate candidate
  IDs/names/locations and eligibility states. A known withheld source name or alias returns
  `no_evidence` rather than an answer about another movement. Withheld identities still
  participate in ambiguity detection, but never supply trusted instructions.
  Other paraphrases use the existing lexical/semantic retrieval policy, not guaranteed
  entailment.
- Variation `relation` is `easier`, `harder`, `alternative` or `unspecified`, explicitly
  reviewed rather than inferred. Children accept the same fields, aliases and image format.
- Images are **explicit per-node** `images` entries:
  `{"pdf_page": 1, "xref": 12, "bbox": [50,180,100,230]}`. Obtain actual image occurrences
  privately from `page.get_images(full=True)` and `page.get_image_rects(xref)`. The exact
  page/xref/occurrence box must exist in the source. Never attach all images of a shared page.
  Returned refs include source SHA-256, page, xref, bbox and `visual_understanding: false`;
  they associate an original photograph location, not its visual meaning. Vector drawings
  are not fabricated as image xrefs; original page/region citations remain available.
- Coverage defaults to `partial`. Set `coverage: "complete"` **only after** comparison with
  original pages verifies all relevant instructions, warnings, cross-references and
  variations. Optional `gaps` is an array of operator gap codes; any gap withholds the whole
  parent. Record acceptance cannot override missing fields, partial coverage or unresolved
  OCR anywhere inside a declared boundary. Selecting a readable fragment cannot bypass a
  damaged warning elsewhere in that boundary. Operators must still identify boundaries
  correctly: coordinates and hashes are not independent semantic review.

## Draft, inspect, review, import

```sh
umask 077
uv run cited draft-exercises "$PRIVATE_DATA/source.pdf" --start 1 --end 2 \
  --layout "$PRIVATE_DATA/layout.json" --catalog "$PRIVATE_DATA/exercises.json"
# For damaged native text, first recover/review the existing OCR bundle, then add:
# --recovery "$PRIVATE_DATA/review/bundle.json"
uv run cited inspect-exercises "$PRIVATE_DATA/exercises.json"
uv run cited review-exercise "$PRIVATE_DATA/exercises.json" \
  --record cloud --revision "$EXACT_REVIEWED_REVISION" --state accepted
# Repeat for each reviewed parent; pending/rejected are also valid decisions.
uv run cited import "$PRIVATE_DATA/source.pdf" --start 1 --end 2 \
  --index "$PRIVATE_DATA/exercises.sqlite" --model-dir "$MODEL_DIR" \
  --approve-sha256 "$APPROVED_SHA256" --approve-pages 1,2 \
  --exercises "$PRIVATE_DATA/exercises.json"
# If catalog uses recovery, pass the same --recovery bundle to import as well.
uv run cited search 'How do I do the sky push?' \
  --index "$PRIVATE_DATA/exercises.sqlite" --model-dir "$MODEL_DIR"
uv run cited report --index "$PRIVATE_DATA/exercises.sqlite"
```

`draft_exercises(pdf, catalog, layout_dict, start, end, recovery=None)`,
`inspect_exercises(catalog)` and `review_exercise(catalog, record_id, revision, state)`
are exported Python functions. `ingest(..., exercises=catalog, recovery=bundle)` extends
existing import. Inspect output contains `binding`, original `layout` and `records` with
exact `revision`, `state`, `history`, `coverage`, `gaps`, `eligible`, source-derived fields,
child variations and citations. Review the **whole parent and every child** before accepting
that exact revision. Re-draft after layout/source/recovery edits; changed revisions reset
pending and retain historical decisions without transferring trust. Do not hand-edit a
sealed catalog. Identical re-drafts retain current acceptance.

Exercise mode never indexes ordinary raw-page snippets alongside records. Only accepted,
complete, gap-free parent records enter the index; none eligible fails explicitly and
preserves the previous index. Each parent's own fields and each variation receive separate
bounded lexical/semantic matching windows, so a late variation is not lost to the embedding
model's context limit. `passage_count` counts these windows, not exercise records. Results
are deduplicated by parent before applying the result limit. A match returns the whole parent
in `passages[].exercise`, including all instruction segments, warnings, references, children
and image references. `text` retains the complete source-backed parent/variation context
plus reviewed aliases; `retrieval_text` and `provenance.matched_node_id` identify the matching
window/node. The outer location is that node's source anchor, not a claim that every
cross-page instruction occurs there; individual citation segments retain their exact pages,
regions and original lines. Search coverage contains only record IDs/revisions,
states and gap summaries, not withheld sibling text. Reports/inspect are private review
surfaces and include withheld draft content; do not treat that as eligible evidence.

Source, recovery/crops, settings, exact catalog hash and derived revisions are revalidated
on report/search and before/after query embedding. Any changed catalog/review decision
invalidates the existing index (`stale_exercises`) until rebuilt. Recovery changes require
re-drafting and reviewing affected records before rebuilding. Imports and catalog writes
are atomic and mode 0600; protect parent directories and serialize operator updates.
Hashes detect staleness, not hostile rewriting of all trust artifacts.

Private acceptance should separately compare representative multi-page exercises, adjacent
same-page movements, harder/easier children, shared-line metadata, hyphens and explicitly
associated image occurrences to rendered original pages. Record exclusions and unresolved
gaps. Offline authored tests do not establish real-source or live-model correctness.
