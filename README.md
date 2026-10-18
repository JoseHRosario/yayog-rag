# Cited local retrieval

Import a **manually reviewed native-text prose section** from a local PDF and retrieve
supporting passages. This slice returns evidence, not generated advice. No OCR, exercise
parsing, schedules, answer generation, hosted database, or UI is included.

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
bookmarks. **Successful `search` stdout intentionally includes private source passages.**
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
not a substitute for human review. No OCR is run.

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
`review_status: "acknowledged"`, fused `score`, `semantic_score`, and lexical/semantic ranks.
`acknowledged` means explicit operator acknowledgement, not independent verification.

SQLite FTS5/BM25 lexical candidates and cosine semantic candidates are combined using
reciprocal-rank fusion (constant 60). Semantic candidates need cosine >= 0.45. Lexical
matches require a non-stopword English query term and nonnegative cosine similarity;
common question/function words alone cannot supply lexical evidence. These are fixed
retrieval heuristics, not calibrated truth/confidence scores or entailment checks.
Policy values are shared with the serialized compatibility settings. Paragraph-like PDF text blocks are split
at bookmark transitions and about 1,200 characters, never merged across pages/chapters.
Heading-only blocks matching bookmarks remain metadata rather than standalone evidence.
Short/non-prose chunks (fewer than five words or less than half alphabetic characters)
are excluded, which removes common page labels and decorative rules. This conservative
prose-only policy can also exclude brief legitimate text; review the chosen section.
Complex layouts can require exclusion during human review; this is a prose-only slice.

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
