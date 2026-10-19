# Reviewed source excerpts and explicit image text

All paths and successful command output in this workflow are private source-derived data.
Keep catalogs, selector JSON, crops, indexes and evaluations outside **every** Git checkout
(including linked worktrees). These APIs never write the original PDF and never interpret
photographs, repair instructions, invent headings, or establish semantic completeness.

## Individually reviewed paragraphs

Use excerpts for prose, legal qualifications, definitions, front/back matter and complete
cross-page paragraphs. Do not invent an exercise or method identity to import prose.
`draft_excerpts(pdf, catalog, layout, start, end, recovery=None)` takes a layout mapping
or a private JSON file path. The file form additionally pins the input file's exact hash.
The layout contains only `records`; each record has an operator ID, ordered source selectors,
full paragraph boundaries, optional explicit gaps, and an operator completeness declaration:

```json
{
  "records": [{
    "id": "paragraph-01",
    "coverage": "complete",
    "boundaries": [
      {"pdf_page": 2, "bbox": [40, 500, 550, 700]},
      {"pdf_page": 3, "bbox": [40, 40, 550, 100]}
    ],
    "selectors": [
      {"pdf_page": 2, "bbox": [40, 500, 550, 700]},
      {"pdf_page": 3, "bbox": [40, 40, 550, 100]}
    ]
  }]
}
```

Coordinates are finite PDF page points, in the page's coordinate system. Pages are strict
one-based integers (not booleans). Rectangles must have positive area and stay inside the
source page and declared boundaries. Select whole glyphs. Selector order is evidence order;
include continuations, exceptions and terminal qualifications. An optional exact, nonempty,
unique `quote` selects a substring of actual extraction. Optional `joins` each name an exact
unique source substring containing `-\n`; only those explicitly recorded line-end hyphens
are joined. No selector accepts replacement text. Original text and glyph/line locations
remain available in `segments`. Chapters/section paths and printed labels come from actual
PDF bookmarks/page-label metadata, not operator-authored semantic names.

Quality checks examine selected native glyphs and raw characters inside boundaries, not
unrelated damaged text elsewhere on the page. Damage inside a boundary, empty selectors,
operator gaps or omitted completeness declarations make the entire record partial and
ineligible. The `coverage: complete` declaration is an operator assertion, **not** an
algorithmic proof of complete paragraphs or semantic support. Inspect the original source.

```sh
uv run cited draft-excerpts "$PRIVATE_DATA/source.pdf" \
  --catalog "$PRIVATE_DATA/excerpts.json" --layout "$PRIVATE_DATA/excerpt-layout.json" \
  --start 2 --end 3
uv run cited inspect-excerpts "$PRIVATE_DATA/excerpts.json"
uv run cited review-excerpt "$PRIVATE_DATA/excerpts.json" \
  --record paragraph-01 --revision '<exact inspected revision>' --state accepted
uv run cited import "$PRIVATE_DATA/source.pdf" --start 2 --end 3 \
  --excerpts "$PRIVATE_DATA/excerpts.json" --index "$PRIVATE_DATA/excerpts.sqlite" \
  --model-dir "$MODEL_DIR" --approve-sha256 "$APPROVED_SHA256" --approve-pages 2,3
```

Public Python exports are `draft_excerpts`, `inspect_excerpts`, `review_excerpt`, and
`ingest(..., excerpts=catalog)`. `excerpts`, `exercises` and `programmes` are mutually
exclusive index modes. Drafts default to pending. Exact revision acceptance plus complete
selected coverage is required. Regenerating changed source/selectors/settings/recovery
resets acceptance while retaining history. Revoking review or changing a bound input makes
an existing index stale; rebuild explicitly. Reports retain every selected record's state,
coverage and gaps. Imported excerpt pages are always partial, not approved whole pages.

Matching windows stay bounded by the existing chunk size, while every returned passage
retains the **entire approved paragraph**, all segments and their citations. Same-page
context expands only from other eligible indexed excerpts; unselected source material is
never read into context. Answer packets retain the complete excerpt evidence and bind the
normal report snapshot; no additional model protocol is needed.

`draft_coverage(..., excerpt_catalogs=[catalog])` and CLI `draft-coverage --excerpt-catalog`
also accept catalogs without indexes. Index-based discovery automatically inventories
excerpt records using **all** boundary pages and pins catalog, source, external layout,
recovery and every transitive crop. This is conservative accounting, not a trust bypass.

## Contained imports from a full recovery bundle

`ingest(..., recovery=full_bundle)` permits an index section contained within the validated
bundle's inclusive section. The bundle is never rewritten to fit the selection. Its whole
file hash and complete source/crop/review graph remain pinned, before and after embedding
and on report/search. Changes or revocation **outside** selected pages still invalidate the
index. Eligibility and gaps are scoped to selected pages; `recovery.outside_selection`
discloses omitted regions and gaps without granting their acceptance. Exact-range legacy
imports keep their original shape. Exercise/excerpt selectors can also use contained ranges.

## Explicit image-only text ROIs

Default `recover` remains damaged-native-line OCR. For printed text inside raster panels
or image-only pages, supply an explicit JSON **array**:

```json
[{"id": "annotation-01", "pdf_page": 2, "bbox": [50, 200, 450, 250]}]
```

```sh
uv run cited recover "$PRIVATE_DATA/source.pdf" --start 2 --end 3 \
  --bundle "$PRIVATE_DATA/image-review.json" --image-regions "$PRIVATE_DATA/image-rois.json" \
  --tesseract "$TESSERACT_BIN" --tessdata-dir "$TESSDATA_DIR"
uv run cited inspect-review "$PRIVATE_DATA/image-review.json"
# Compare each crop against the original, then use the existing review command.
uv run cited review "$PRIVATE_DATA/image-review.json" --region annotation-01 \
  --revision '<exact inspected revision>' --state accepted
```

Python: `recover(..., image_regions=[{"id": ..., "pdf_page": ..., "bbox": [...]}])`.
Supplying ROIs switches to **exclusive explicit mode**; it does not additionally OCR native
damage on hundreds of pages. The full configuration is validated before crop publication.
Reject empty lists, duplicate IDs, overlapping ROIs, nonfinite/boolean coordinates,
out-of-range pages and any overlap with native glyphs (healthy **or** damaged). Use the
native workflow for those glyphs. Native material is not silently approved by image review:
it must satisfy ordinary native policy or be selected/reviewed separately.

Crops use the exact requested source rectangle, 400 DPI, no padding and no preprocessing;
the existing OCR adapter/PSM is unchanged. Raw external OCR output is retained verbatim.
Source hash, geometry, rendered crop hash, executable/tessdata identity, renderer/settings
and output all bind the revision. Changed ROI/tool/output/pixels creates a pending revision.
Each ROI becomes real block/line provenance, so an accepted region with **zero native
glyphs** can enter prose and exercise/excerpt selectors using its exact ROI coordinates.

Unselected image areas remain explicit unresolved gaps outside owned blocks. Acceptance is
not whole-page approval. A broad paragraph boundary spanning several crops and unselected
space stays partial; use precise ordered boundaries/selectors and review every selected
region. A paragraph including any pending crop remains ineligible even if its record itself
was accepted. Empty OCR cannot be accepted; irrelevant photograph noise is never accepted
automatically. Source/crop mutation invalidates dependent catalogs/indexes.

## Compatibility and limits

Legacy recovery schema 1 / settings version 2 and existing source/exercise/programme
reviews are unchanged. Explicit image bundles alone opt into schema 2 / settings version 3
(`explicit-image-only-roi-v1`). Opted-in indexes carry strict image/excerpt policy markers
in their settings so old readers reject them rather than reinterpret them. Readers reject
missing/invalid current markers. Rebuild only the affected opted-in index; no global
extraction-settings bump is needed and legacy bundles need no resealing.

New catalogs validate source reconstruction and review on every inspection; these are not
trusted-cache results. Dependencies are revalidated after embedding, including external
failures. New output creation is atomic/no-clobber; regeneration compares the owned prior
bytes before replacement. Protected dependency paths cannot be used as index/catalog
outputs. Existing input files are not deleted to recover from an invalid draft.

Index publication on the POSIX retrieval host snapshots destination identity and bytes.
New destinations use atomic no-clobber creation. Rebuilds reject observed destination
changes and serialize cooperating publishers with a directory advisory lock. This is
not kernel compare-and-swap against arbitrary noncooperating filesystem writers in the
final check/replace interval; protect the private data directory from such writers.
The standalone Windows inference client is unaffected by these host-side locks.

Automated tests use authored synthetic sources and external inference/OCR doubles. They do
not prove private-source transcription quality, coverage completeness or real-model ranking.
Local OCR tools/language data must already exist; the workflow installs or downloads nothing.
