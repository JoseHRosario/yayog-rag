# Full-source coverage and owner acceptance

This is **private operator accounting**, not automatic extraction, semantic verification,
medical validation, personalised coaching or visual form assessment. A selected passage,
a clean extraction candidate, an accepted record, or a successful sample query never
establishes whole-page or whole-source review. Existing import/search/answer trust policy
is unchanged. Coverage exclusions cannot authorize rejected OCR or incomplete records.

Keep source PDFs, manifests, catalogs, indexes, evidence files and successful command
output outside the checkout and publication artifacts. Never publish source identifiers,
names, quotations, derived content or secrets. The examples use generic placeholders.

## Create and inspect

```sh
umask 077
uv run cited draft-coverage "$PRIVATE_DATA/source.pdf" \
  --audit "$PRIVATE_DATA/coverage-v1.json" \
  --index "$PRIVATE_DATA/prose.sqlite" \
  --index "$PRIVATE_DATA/exercises.sqlite" \
  --exercise-catalog "$PRIVATE_DATA/exercises.json" \
  --programme-catalog "$PRIVATE_DATA/programmes.json" \
  --inventory "$PRIVATE_DATA/inventory-v1.json" \
  --recovery "$PRIVATE_DATA/review/bundle.json"
# Edit operator decisions/evaluations/gates, not registered inventory or binding.
uv run cited inspect-coverage "$PRIVATE_DATA/coverage-v1.json"
```

Catalog, index and recovery options are repeatable and optional; `--inventory` is a
single optional private JSON file (create it as described below). Omit that option if
there are no manual records yet. Supply **all** applicable artifacts;
indices automatically add their bound catalogs/recovery and transitive support inputs.
A catalog is useful even when no record is eligible for import. Recovery bundles can be
supplied without an index. These commands do not load an embedding model or contact an
inference server. Do not redirect stdout onto an input artifact.

Python exports:

```python
draft_coverage(pdf, audit, *, indexes=None, exercise_catalogs=None,
               programme_catalogs=None, recovery_bundles=None, inventory=None)
inspect_coverage(audit)
```

`draft_coverage` requires a **new** output path; it refuses any existing target, including
input aliases. Both draft and inspection reject an audit path inside **any Git ancestor**,
including another checkout, linked worktree, or a manifest moved there later. The same
private-path rule applies to registered inventory files. Symlinks are resolved before
checking boundaries and input/output collisions; the inventory cannot alias the source,
any supplied/discovered artifact, or the audit target. It writes atomic mode-0600 JSON.
Protect parent directories and serialize operator updates. The original PDF, catalogs,
review bundles, crops and indexes are never
modified. Create a new draft after an artifact changes; reconcile earlier decisions
explicitly rather than copying approval onto new revisions.

## Manifest schema 1

The manifest is intentionally editable operator evidence, unlike sealed extraction
catalogs. Retain its generated `schema` and `binding` unchanged. `binding` pins the source
SHA-256 and absolute path, artifact specs, validated corpus/model/settings identity,
catalog records/revisions, and exact hashes of all discovered dependencies and crops.
When supplied, `binding.inventory` pins the external inventory's absolute path, exact
file SHA-256, and complete manual-record baseline. Keep both the binding and external
inventory unchanged for the lifetime of that audit. Inspection checks the external file
and every registered record against that baseline; it never learns a new baseline from
an edited manifest or writes anything during inspection.
Hashing detects accidental staleness, not hostile rewriting of every trust artifact.
Inspection reruns existing source/catalog/recovery/index validation and compares the
snapshot. Source changes, catalog review revocation, crop changes, embedding/settings
incompatibility and changed/missing artifacts require re-draft/rebuild, not silent reuse.
Identical inputs produce identical drafts; no timestamps or generated IDs are introduced.

### Pages

Every physical page appears exactly once, in physical order. Retain `pdf_page`, actual
`printed_label` (possibly null), and `native_reason`. No printed-page offset is guessed.
Initial status is always `unresolved`, including textless pages and front matter.
Native reasons are diagnostics, **not** decisions. Set:

- `status`: `trusted`, `reviewed_exclusion`, or `unresolved`;
- `reason`: nonempty operator explanation of scope, exclusions or unresolved limits;
- `whole_page_reviewed: true` and `inventory_complete: true` for either resolved status;
- optional `gaps`: nonempty gap-code strings, e.g. `missing_body_continuation`.

Review the rendered **entire page**, not merely selected bounding boxes. Inventory all
relevant principles, exercises, variations, continuations, warnings, methods and sessions.
A deliberate textless/front-matter exclusion requires the same whole-page/inventory
assertions and a reason. Record-level selection never supplies these assertions.
Unresolved records/regions on a page, or explicit page gaps, yield effective
`status: unresolved` while preserving `declared_status` and reporting `blockers`.

### Records and missing material

Generated `records` enumerate every supplied catalog parent/session/method and ordinary
index passage, not just eligible records. Their stable audit `key` is artifact path plus
record/passage ID. Preserve all generated identity, source, page, revision, eligibility,
review-state, gap and dependency fields. Only `status` and `reason` are editable.
An eligible complete record defaults to trusted; a session's missing/ambiguous method or
exercise links remain explicit dependency blockers, even if retrieval can still return
its faithfully preserved schedule cells. Eligibility is not silently revoked or expanded.

A region/record can be marked `reviewed_exclusion` with an explicit operator reason.
This means accounted-for exclusion, **not trusted instructions**. Do not exclude critical
missing instructions merely to obtain a clean report. Unsupported material must continue
to limit answers. An unresolved excluded-looking footer is not automatically harmless.

Register missing relevant records in a **separate private inventory file before drafting**,
instead of appending them to the editable audit. For example, `inventory-v1.json`:

```json
{"source_sha256":"<actual source SHA-256>","records":[
 {"key":"missing-principle-1","kind":"principle","pdf_pages":[2],
  "status":"unresolved","reason":"Continuation not yet mapped",
  "dependencies":["missing_continuation"]}
]}
```

For a manually inventoried principle supported by existing eligible indexed prose, use
`status: trusted`, `whole_record_reviewed: true`, and nonempty `evidence_keys` referencing
trusted generated records on the same source. `pdf_pages` must equal the union of evidence
pages; `gaps` and `dependencies` must be empty. The operator must compare the **complete
principle including qualifications** with the original. This declaration cannot create
retrieval eligibility or prove the selected passages are semantically complete. Other
uncatalogued items stay unresolved or explicitly excluded. Register `evidence_keys`,
`whole_record_reviewed`, page union, gaps and dependencies in the external inventory;
they are frozen, not fields to add later in the manifest. The referenced generated records
must be trusted **in the current operator manifest as well as in the validated artifact**.
An audit-excluded or unresolved record cannot support a trusted manual group, regardless
of its unchanged retrieval eligibility. References to unknown/manual keys are not evidence.

Every registered key must remain present exactly once; only its audit `status` and
`reason` may change. Deleting a registered unresolved item cannot promote completeness;
adding an unregistered item, or modifying registered content, invalidates inspection.
Do not delete generated items either. A draft without an inventory permits **no manual
records**. Existing schema-1 audits with only generated records remain valid; old audits
with ad-hoc manual additions require explicit registration and a fresh draft.

### Registration and reconciliation after discovery

1. Draft with all applicable artifacts. Use their generated record keys to identify
   candidate evidence; this initial unresolved draft is not whole-source acceptance.
2. Review the original rendered pages, then author a versioned private inventory with
   the exact source SHA-256 and all manually discovered items, including unresolved ones.
3. Create a **new** audit using `--inventory` and the same complete artifact set. Retain
   the earlier audit and its inputs; review and reconcile decisions explicitly.
4. Edit only record status/reason, page review decisions, region status/reason, evaluations
   and gates. Run `inspect-coverage`; fix unresolved content rather than deleting it.
5. For any inventory change (even formatting), retain the old inventory and audit, create
   a new versioned inventory and a fresh audit path, and reconcile again. Editing the
   original external inventory makes its old audit stale; inspection never overwrites it.

Hashes and baselines detect accidental omission/staleness, not malicious rewriting of
both the binding and all external evidence. Protect private files and keep review history.

### Recovery regions

`regions` pin actual `region_id`, source SHA-256, bundle-qualified `key`, exact `revision`,
physical page, bbox and recovery `review_state`. Preserve identity fields. Accepted OCR
defaults to trusted; pending/rejected OCR defaults to unresolved. A separately visually
reviewed running footer can be given audit `status: reviewed_exclusion` and a precise
reason while recovery state remains rejected. This does **not** make that region available
for retrieval, override a damaged instruction in a catalog boundary, or reinterpret other
regions as exclusions. Pending body text continues to block coverage. Changing a bound
region's geometry/revision, deleting it, or changing its review bundle invalidates audit.

### Private evaluation and separate gates

`evaluations` is an owner-checkable array. Each case has unique nonempty `id`, `question`,
`category`, `status` (`not_run`, `passed`, `failed`), `expected_behavior`, `locations`,
`claim_support` and `evidence`. Required categories are `exact_name`, `paraphrase`,
`variation`, `session_filter`, `unsupported`, `conflict`. Record actual session filter
inputs and expected identities in the private question/evidence; do not claim an unexecuted
filter passed. For answer cases, expected behavior is `answer`; unsupported requires
`abstain`; conflict requires `disclose_conflict` and at least two distinct source locations.

Locations are `{pdf_page, printed_label, quote}`. Quotes must be nonempty exact substrings
of the original page's native extraction and labels must match metadata. Unsupported cases
may use an empty list. Recovered-only text cannot be verified by this native-quote check:
retain rendered/recovery comparisons in private evidence and anchor a native location
where possible; do not manufacture a matching quote. `claim_support` explains the actual
claim-to-source comparison, not merely citation presence. A passed/failed case must point
to an existing evidence file with exact SHA-256:

```json
{"path":"/private/operator-results.json","sha256":"<actual SHA-256>"}
```

A passed/failed case's evidence file must be a JSON envelope with `case` and
`result`. The `case` object repeats **every case input** (including optional structured
`filters`/expected identities), excluding only the editable `status`, `claim_support`
and `evidence` fields, and adds the audit's exact `source_sha256`. `result` retains the
actual output or its owner-checkable evidence references. For example:

```json
{
  "case": {
    "source_sha256": "<audit source SHA-256>",
    "id": "session-01",
    "category": "session_filter",
    "question": "<actual evaluated question>",
    "filters": {"programme": "<actual programme>", "week": 1, "day": 1},
    "expected_behavior": "answer",
    "locations": [{"pdf_page": 2, "printed_label": "ii", "quote": "<exact native quote>"}]
  },
  "result": {"path": "/private/retained-result.json", "sha256": "<actual result SHA-256>"}
}
```

The envelope's exact file SHA-256 is the case's `evidence` reference. Altering a question,
identity, filter, expected behavior or location while retaining old evidence raises
`coverage_case_binding_mismatch`; changed envelope bytes raise `stale_coverage_evidence`.
Record a new evidence revision after re-evaluation, retaining the earlier failed attempt.
This association prevents accidental case/result reuse; nested result references and
claim entailment still require owner inspection and are not independently authenticated.

`gates` has exactly `offline`, `private`, `live`, each with `status` and `evidence`.
Initial statuses are `not_run`; passed/failed gates require an existing file and exact
SHA-256, but do not require a case envelope.
Keep synthetic offline results separate from real-source checks and actual model calls.
Changed/missing evidence files invalidate recorded results. These are **operator-recorded
results**, not tool-run attestations: the auditor checks identity and locations, never
executes evaluations, proves entailment, or establishes that a live request really ran.
Use existing public search/answer flows and inspect actual supported claims, abstention
and conflict disclosure before recording private/live acceptance.

## Large rebuilds

Local CPU embeddings use bounded batches of 16 documents rather than the upstream
large-batch default. This limits inference memory without dropping passages or
changing the model/artifact identity. It is not a total process-memory guarantee;
serialize memory-heavy rebuild/acceptance jobs on constrained machines and retain
failed runs. Exercise review-revocation and changed-source tests on isolated copies,
not the active corpus; a killed test must not leave the production index revoked.

## Reading the report

- `binding` exposes the exact source/artifact/dependency snapshot for owner checking.
- `counts` separately summarizes pages, records and regions by the three dispositions.
- `coverage_complete` means every page/inventory and supplied record/region is accounted
  for, including reviewed exclusions. It does **not** mean all content is trusted/indexed.
- `evaluation_complete` requires all six categories passed and no pending/failed case.
- `delivery_complete` additionally requires coverage and all three gates passed. This is
  completeness of the **operator's declared audit**, not independent proof of whole-source
  semantic completeness or model correctness. A small curated manifest cannot establish
  representative acceptance on its own.

Report unresolved limits, reviewed exclusions, synthetic tests, real-corpus validation and
live-model acceptance separately. Keep original PDFs unchanged and inspect publication
artifacts/history for sensitive source material and secrets before publishing repository
changes. Full-source accounting does not replace the source-bound exercise/programme
review procedures or medical expertise.
