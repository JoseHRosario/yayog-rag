# Cited local retrieval

Import a **manually reviewed prose section** from a local PDF and retrieve supporting
passages. Damaged text lines can be recovered with local targeted OCR and explicitly
accepted against rendered crops. Reviewed exercise catalogs can preserve complete movement
context using operator-supplied boundaries and source-backed field selectors. Reviewed
programme sessions preserve schedule cells, exact week/day scope, pair order, timings,
supporting references and source conflicts. Search returns evidence; optional grounded
answers use an explicitly authorized existing OpenAI-compatible inference endpoint.
Full-source operator accounting is available through
[Coverage and acceptance](docs/full-source-coverage.md); this does not establish
whole-source semantic correctness or real-model acceptance. Hosted databases and UI
are not included.
See [Exercise catalogs](docs/exercise-catalogs.md) and
[Programme sessions](docs/programme-sessions.md) for private layout schemas and commands.
See [Reviewed source excerpts](docs/reviewed-source-excerpts.md) for individually reviewed
native paragraphs, contained imports from full recovery bundles, and opt-in image-text ROIs.
See [Grounded answers](docs/grounded-answers.md) for `answer`, `prepare-answer`,
`finish-answer`, and the portable owner-run Windows client (standard library only).

## Install and private storage

Python 3.11+ and `uv` are required:

```sh
uv sync --extra dev --locked
uv run python -m cited_retrieval --help
```

Keep **all sources, model files, extracted text/images, indexes, review records, and live
evaluation output outside the repository**, for example under `$HOME/yayog/private`.
Do not copy source content into issues, tests, documentation, terminals recorded in CI,
or commits. Existing exclusions cover PDFs, `/data`, `/artifacts`, `/models`, `.local`,
SQLite databases, model arrays, environments, credentials, and caches. These are defense
in depth, not permission to commit other derived formats. Only authored synthetic
fixtures belong in tests; tests create their PDFs in the temporary directory.

The CLI emits safe JSON errors with exit status 2, without raw exceptions, source paths,
or provider logs. Successful `inspect`/`report` output includes private metadata such as
bookmarks. **Successful `search`, `recover`, `inspect-review`, and `review` stdout contains
private source or recovered text.** Answer/prepare/finish results (including error
disclosures) and all answer bundles/response files are private source-derived data too.
Use a private shell/output destination; never capture live retrieval output in CI logs.
The Python API propagates exceptions for callers to handle and is not a logging boundary.

## Provision the real local embedding model

Production uses CPU ONNX inference through FastEmbed with
`sentence-transformers/all-MiniLM-L6-v2`, symmetrically embedding documents and queries.
There is no hash, bag-of-words, hosted inference, or other semantic fallback.

Provision the model separately and explicitly; imports/searches never download it.
Choose a reviewed Hugging Face commit revision and set `MODEL_REVISION` to that commit.
This optional provisioning step requires network access; ordinary tests do not:

```sh
export PRIVATE_DATA="$HOME/yayog/private"
export MODEL_DIR="$PRIVATE_DATA/models/minilm"
# Set MODEL_REVISION to a reviewed commit from qdrant/all-MiniLM-L6-v2-onnx.
umask 077
mkdir -p "$PRIVATE_DATA"
uv run python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="qdrant/all-MiniLM-L6-v2-onnx", revision=os.environ["MODEL_REVISION"], local_dir=os.environ["MODEL_DIR"], allow_patterns=["model.onnx", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])'
```

The adapter requires those files locally and fingerprints the complete model directory.
It records model name, actual artifact SHA-256, FastEmbed/ONNX Runtime/tokenizer/NumPy
versions and embedding convention. Changing files, runtime versions, dimensions, source,
corpus, or extraction/ranking settings requires rebuilding. Keep the provisioned model
directory immutable. Model licensing and source rights remain the operator's responsibility.

## Review, import, report, query

PDF page numbers are **one-based physical PDF pages**, not printed labels. A section is
an inclusive start/end interval. This generic example reviews pages 2–3; choose actual
native-text prose pages from your own source:

```sh
uv run cited inspect "$PRIVATE_DATA/source.pdf" --start 2 --end 3
```

Inspect the original PDF visually and compare its native extraction privately. Verify
reading order, paragraph boundaries, characters, chapter transitions, and printed labels;
nonempty native text is **not** verification. `inspect` reports candidates and exclusions,
not approved pages. Fonts requiring a missing ToUnicode map, replacement/control/private-use
characters, unmapped glyphs, short/non-prose pages and blank/scanned pages are excluded.
Simple Type1/TrueType fonts with explicit WinAnsi/MacRoman encoding and standard
unembedded base-14 fonts can map text without a redundant ToUnicode stream; they still
require the same glyph checks and manual review. These checks are conservative heuristics,
not a substitute for human review. `inspect` and native-only import do not run OCR.

After reviewing the exact artifact and every selected page, copy the reported SHA-256
into `APPROVED_SHA256` and explicitly acknowledge all selected pages:

```sh
export APPROVED_SHA256='<sha256 from inspect after review>'
uv run cited import "$PRIVATE_DATA/source.pdf" --start 2 --end 3 \
  --index "$PRIVATE_DATA/prose.sqlite" --model-dir "$MODEL_DIR" \
  --approve-sha256 "$APPROVED_SHA256" --approve-pages 2,3
uv run cited report --index "$PRIVATE_DATA/prose.sqlite"
uv run cited search 'How does rest support recovery?' \
  --index "$PRIVATE_DATA/prose.sqlite" --model-dir "$MODEL_DIR" --limit 5
```

Acknowledgement is bound to the source hash and exact selected pages; a changed source or
missing acknowledgement cannot be imported. Bad selected pages remain excluded even when
acknowledged. The report explicitly accounts for **every PDF page**: imported pages,
selected exclusions with reasons, and unselected pages with `not_selected`. It includes
section bounds, bookmark boundaries, PDF/printed labels, model identity, corpus hash,
extraction version and approval. Printed labels come from PDF page-label metadata and are
`null` if absent; labels visible only in the page image/text are not guessed.

Re-run the same import command to rebuild, not append. Source PDFs are never written.
The new SQLite database is built in a private temporary file beside the target, then
atomically replaces it only after successful extraction/inference/validation. Failures
preserve the prior index. The index file is mode 0600; protect its parent directory too.

## Targeted recovery and explicit review

Provision Tesseract and its language data **locally and separately**. Supply the language
directory explicitly or through `TESSDATA_PREFIX`; implicit installation defaults are
rejected because their processing artifacts cannot be fingerprinted reliably. The application
installs nothing and never downloads OCR data. The production adapter runs the selected
executable without a shell, inherits the process environment (including loader-library
settings), and applies a per-subprocess deadline. Use explicit executable and tessdata
paths for reproducible processing; selected traineddata files and the executable are hashed.

```sh
# TESSERACT_BIN and TESSDATA_DIR refer to existing external local tools/data.
umask 077
uv run cited recover "$PRIVATE_DATA/source.pdf" --start 2 --end 3 \
  --bundle "$PRIVATE_DATA/review/bundle.json" \
  --tesseract "$TESSERACT_BIN" --tessdata-dir "$TESSDATA_DIR" \
  --language eng --timeout 60
uv run cited inspect-review "$PRIVATE_DATA/review/bundle.json"
```

Only damaged text-line rectangles are rendered, with a small margin at **400 DPI**;
Tesseract uses **PSM 7**. No page-wide OCR, photograph masking, LLM reconstruction,
spelling correction, quantity correction, or warning removal occurs. Detection checks
font mappings, raw text-trace characters, replacements and conservative garbage signals,
not merely nonempty sorted extraction. Font-family names do not identify individual font
resources: region checks disable CID fallback to avoid OCRing healthy same-family text.
A completely white rendered spacing crop is retained as audited blank layout, not sent to
OCR; its original/raw extraction remains in the bundle. Detection remains heuristic:
plausible wrong words can escape it. Blank/scanned pages without native text regions remain reported gaps;
the default command does not attempt full-page OCR or exercise-layout reconstruction.
Explicit image-only text rectangles can instead opt into `--image-regions`; see
[the ROI workflow](docs/reviewed-source-excerpts.md#explicit-image-only-text-rois).

The private JSON bundle contains readable native lines, original damaged extraction and
raw text, one-based physical PDF pages, coordinates, crop references/SHA-256, verbatim OCR,
source SHA-256, renderer/OCR versions and settings, revision IDs, and review-state history.
Open **each referenced PNG** beside the original page and compare the complete text,
especially quantities, units and warnings. Choose the region ID and exact revision from
`inspect-review`; acknowledgement of a page is not acceptance of its OCR:

```sh
export REGION_ID='<region id from inspect-review>'
export REGION_REVISION='<exact revision reviewed against its crop>'
uv run cited review "$PRIVATE_DATA/review/bundle.json" \
  --region "$REGION_ID" --revision "$REGION_REVISION" --state accepted
# Other decisions: --state rejected or --state pending.
# Repeat explicitly for every region you have actually inspected.
uv run cited import "$PRIVATE_DATA/source.pdf" --start 2 --end 3 \
  --index "$PRIVATE_DATA/prose.sqlite" --model-dir "$MODEL_DIR" \
  --approve-sha256 "$APPROVED_SHA256" --approve-pages 2,3 \
  --recovery "$PRIVATE_DATA/review/bundle.json"
uv run cited report --index "$PRIVATE_DATA/prose.sqlite"
uv run cited search 'How does rest support recovery?' \
  --index "$PRIVATE_DATA/prose.sqlite" --model-dir "$MODEL_DIR"
```

Pending/rejected OCR is never eligible. **An entire native paragraph block containing an
unresolved line is withheld**, including its readable lines, so a truncated instruction
cannot lose a warning. Separate sound native blocks remain eligible after page review;
reports mark partial pages and identify unresolved blocks, regions and states. Once every
damaged line in a block is accepted, native and recovered lines enter together without
rewriting the OCR. Explicitly accepted recovered warnings/quantities remain eligible even
when shorter than the native prose filter. Other filtered native chunks are preserved in
the private coverage gaps with their text and coordinates; the page is marked partial,
not silently complete. `unresolved_regions` counts unaccepted OCR regions, not every exclusion.
If nothing is eligible, import fails explicitly rather than building empty evidence.

Re-running `recover` retains acceptance only for identical source/crop/extraction/OCR/
engine/settings revisions. Changed content resets pending and retains prior review history.
A changed or missing source/crop or a modified bundle fails validation. Regenerating after
a source change creates new pending revisions. Every search/report revalidates the recorded
bundle and **all referenced crops**, as well as the source and corpus. **Any review-state
change makes an existing recovery-backed index fail with `stale_recovery` until rebuilt**;
it cannot keep serving formerly accepted content. Older index schemas fail with
`stale_settings` and require rebuilding. Writes replace bundles/indexes atomically;
failed OCR or inference preserves the previous artifact. Protect the private directory,
retain crops with the bundle, and serialize operator writes to a bundle.

Python recovery API (in addition to the unchanged native API):

```python
from cited_retrieval import recover, inspect_review, review, ingest
from cited_retrieval.tesseract_ocr import TesseractOCR

engine = TesseractOCR(executable=tesseract_bin, tessdata=tessdata_dir, timeout=60)
preview = recover(pdf, bundle, start, end, engine)
current = inspect_review(bundle)
# Only after visual comparison of this region/revision:
review(bundle, region_id, exact_revision, "accepted")
result = ingest(pdf, index, start, end, model, approval=approval, recovery=bundle)
```

`OCR` is the external inference protocol (`identity -> dict[str, Any]`,
`recognize(image: Path) -> str`). Authored outputs replace only this boundary in offline
tests; production uses the real local Tesseract adapter. `inspect-review` is also available
as `inspectreview`. Successful recovery/review output is private, not a safe logging format.

## Public Python API and output contract

```python
from cited_retrieval import ingest, inspect_source, report, search
from cited_retrieval.local_embedder import LocalEmbedder

# All values here refer to explicitly chosen private paths and reviewed pages.
preview = inspect_source(pdf, start, end)
model = LocalEmbedder(model_dir)
import_report = ingest(
    pdf, index, start, end, model,
    approval={"sha256": reviewed_sha256, "pages": reviewed_pdf_pages},
)
evidence = search(index, question, model, limit=5)
current_report = report(index)
```

`Embedder` is the public inference protocol: `identity -> dict[str, str]` with `model`,
`version`, `artifact_sha256`; and `embed(list[str]) -> list[list[float]]`. Only the ML
inference boundary is replaceable in offline tests. Test vectors are authored examples,
not a production semantic algorithm.

`search` returns `status: "evidence"` or **`status: "no_evidence"`**, a `passages` list,
source ID and model identity. Empty evidence has no invented passages. Each ranked passage
has stable `source_id`/`passage_id`, `text`, `location` (PDF page, printed label, chapter,
hierarchical `section_path`,
bounding box and page-local ordinal), extraction `provenance`, `eligible: true`,
`review_status: "acknowledged"` (native) or `"accepted"` (containing reviewed recovery),
fused `score`, `semantic_score`, and lexical/semantic ranks. Recovery-backed responses also
include `coverage` with page status, unresolved counts, block gaps and warnings; accepted
passages include exact region revisions and OCR processing provenance.
`acknowledged` means explicit operator acknowledgement, not independent verification.

SQLite FTS5/BM25 lexical candidates and cosine semantic candidates are combined using
reciprocal-rank fusion (constant 60). Semantic candidates need cosine >= 0.45. Lexical
matches require a non-stopword English query term and nonnegative cosine similarity;
common question/function words alone cannot supply lexical evidence. These are fixed
retrieval heuristics, not calibrated truth/confidence scores or entailment checks.
Policy values are shared with the serialized compatibility settings. Paragraph-like PDF text blocks are split
at bookmark transitions and about 1,200 characters, never merged across pages/chapters.
Heading-only blocks matching bookmarks remain metadata rather than standalone evidence.
Native short/non-prose chunks (fewer than five words or less than half alphabetic characters)
are excluded, which removes common page labels and decorative rules. Accepted recovered
blocks bypass this length filter; recovery-backed reports explicitly list other exclusions.
This conservative native-prose policy can also exclude brief legitimate text; review the chosen section.
Complex layouts can require exclusion during human review in native prose mode. Exercise
mode instead requires an explicit reviewed catalog as described above.

Every query/report rehashes the recorded source and verifies manifest, corpus, lexical
content and settings. Queries also verify the exact model identity/dimensions. Removed or
changed sources and incompatible/stale artifacts fail explicitly instead of silently using
old evidence. Hashes detect accidental staleness; they are not signatures against an
attacker who can rewrite the entire index. Optional API `source=` additionally checks a
supplied source fingerprint; it does not bypass validation of the original recorded source.

## Offline verification and separate live acceptance

```sh
uv run pytest -q
uv run mypy cited_retrieval
uv run ruff check cited_retrieval tests
uv run ruff format --check cited_retrieval tests
```

Tests require no private source, model download, credentials, server, or network. They use
public import/report/search boundaries with authored PDFs and deterministic vectors at the
external inference seam. CLI tests check error privacy and runnable commands.

A separate operator acceptance run must provision a real local model, review a small clean
section of a private PDF, import it, and compare retrieved passage text, PDF pages, printed
labels and chapter metadata directly with the original. Repeat the import and query to
check stable IDs/counts, and try an unrelated question. Keep all evidence and outputs
private and external; do not add excerpts, screenshots, index files or source identifiers
to this repository. Ordinary tests alone do not claim real-source citation correctness or
real-model retrieval quality.
