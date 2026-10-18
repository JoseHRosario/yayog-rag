# Source-readiness rules

This guide preserves extraction lessons, not private source content or corpus-specific progress.
Read it before changing extraction, chunking, eligibility or source-validation behaviour.

## Extraction

- Compare representative prose, illustrated instructions and schedule tables with rendered pages.
  Nonempty selectable text and absence of replacement characters do not establish correctness.
- Inspect raw spans and font mappings: sorted output can hide control characters. Unmapped
  Identity-H fonts differ from simple fonts with explicit standard encodings; a missing
  ToUnicode stream alone does not prove the latter are unreadable.
- Preserve mixed-font words when joining spans. Keep meaningful paragraphs, nested section
  paths and chapter boundaries separate from cosmetic layout and page footers.
- Where approved work requires OCR, recover damaged regions while retaining sound native
  text. Full-page OCR can read photographs as text. Blind image-box masking can erase
  adjacent instructions. Compare recovery against the visible source.
- Keep original extraction, recovered text, coordinates, tool versions and review history
  in private artifacts. Invalidate or explicitly reconcile approval after content changes.

## Structure and trust

- Read embedded printed-page labels separately from physical PDF page numbers. Do not guess
  a document-wide offset. Preserve stable source/record IDs and the source fingerprint.
- Keep unresolved passages ineligible. Review must be bound to the actual source artifact;
  heuristic checks are not independent content verification.
- Bind OCR decisions to the source, crop, verbatim OCR and processing revision. Preserve
  prior decisions as history, not transferable approval. Revalidate review bundles and crops
  on every query/report; changing a review decision invalidates an existing index until rebuild.
- Withhold a complete mixed paragraph block while any damaged line is unresolved, rather
  than returning an apparently complete instruction with its warning omitted. Distinguish
  separate eligible native blocks from partial page coverage in retrieval output.
- Preserve exercise continuations and variation relationships when that feature is in scope.
  Retaining an image reference does not mean a text-only model understands the image.
- Flattened tables lose programme/day/exercise relationships. Verify grids, ordered pairs,
  quantities, units and timings against the source before trusting structured sessions.
- Disclose contradictions with references to both claims; never silently rewrite them as OCR fixes.

## Retrieval and acceptance

- Validate source, corpus, model identity and compatibility settings before returning evidence.
  Changes require a compatible rebuild, not silent reuse or mixing of vectors.
- Reject irrelevant candidates explicitly; common question words are not lexical evidence.
  Similarity thresholds and citation syntax are not proof that an answer is supported.
- Use authored synthetic material for automated tests. Keep real-source/model checks separate
  and local. Report actual coverage, exclusions and unresolved limits; a clean sample is not
  proof of full-corpus readiness.
