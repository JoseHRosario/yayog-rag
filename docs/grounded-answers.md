# Grounded answers and the Windows-side inference handoff

These commands reuse the existing reviewed native/recovery/exercise/programme index.
They do not create a new UI, hosted service, MCP server or model installation.
**All requests, responses, bundles and answer stdout are private source-derived data.**
Use private storage outside this checkout, never issue/PR comments or CI logs.

## Connectivity rule

Windows `127.0.0.1` is not the retrieval machine's `127.0.0.1`. The safe default
workflow is preparation on the retrieval machine, an owner-run client on Windows,
then validation back on the retrieval machine. Do not scan the LAN, expose/change
server bindings/firewalls, start tunnels, or guess hostnames/model IDs. A different
route requires explicit owner authorization. `--connection-authorized` attests to
that authorization; it does not establish connectivity or grant permission itself.
Preparation never connects to the inference endpoint.

The client checks `GET BASE_URL/models` for the **exact** configured `--model` before
`POST BASE_URL/chat/completions`. There is no model-name normalization/fallback,
redirect following, environment proxy use or automatic retry. Optional bearer auth
comes only from the named environment variable **on the machine running inference**;
never put the secret in a URL, command argument or handoff packet. Userinfo, query
strings and fragments in base URLs are rejected. Plain HTTP is suitable only for
owner-approved loopback/protected routes; it supplies no transport confidentiality.

## 1. Prepare on the retrieval machine

Set `MODEL_ID` to the exact ID supplied by the endpoint owner, `MODEL_DIR` to the
already provisioned embedding directory, and the other paths to existing private
locations. `ANSWER_DIR` must be a **new** directory under an existing private parent.
Do not create that final directory first.
Resolved artifact paths inside any Git checkout (including linked worktrees) are
rejected during preparation, standalone-client writes and finish diagnostics.

```sh
umask 077
uv run cited prepare-answer 'How does rest support recovery?' \
  --index "$PRIVATE_DATA/prose.sqlite" --model-dir "$MODEL_DIR" \
  --base-url http://127.0.0.1:8899/v1 --model "$MODEL_ID" \
  --connection-authorized --timeout 60 --max-tokens 1024 \
  --max-prompt-bytes 120000 --max-response-bytes 2000000 \
  --output-dir "$ANSWER_DIR"
```

Optional `--auth-env INFERENCE_API_KEY` names an owner-managed environment variable;
preparation does not need/read its secret. `--limit` defaults to 5. For an exact
session, add **all three** of `--programme NAME --week N --day N`.

A `prepared` result names `inference-request.json`. A deterministic `abstain`
(`no_evidence` or `ambiguous`) creates no exported inference request and must not
contact the model. Retrieval failures have `status:error, phase:retrieval`.

Keep `request.json` on the retrieval machine: it contains the complete local
snapshot, actual question/prompt, source/passage IDs, PDF/printed locations,
provenance, complete retrieved exercise/session context, disclosures, and endpoint
configuration (not credentials). Only transfer these two files privately to Windows:

- `$ANSWER_DIR/inference-request.json` (reviewed evidence and prompt, **not** the
  index, embedding model, local snapshot or Linux index paths);
- `cited_retrieval/inference_client.py` (generic standalone code from this checkout).

No private payload belongs in Git. No archive or installation is necessary.

## 2. Owner runs the standalone client on Windows

Python 3.11+ is required, **standard library only**. No embedding dependencies,
Linux paths or local index are needed. Use an owner-only NTFS directory/ACL for
these files; POSIX creation modes do not establish Windows ACL privacy. Review
`provider.base_url`, `provider.model` and the prompt privately before authorizing.
The endpoint below is reached from Windows, not from the retrieval machine.

```powershell
py -3 .\inference_client.py .\inference-request.json `
  --response .\inference-response.json --connection-authorized
```

If authentication was configured, provision the named variable locally using the
owner's secret-management workflow; do not send the key back. Return
`inference-response.json` privately, including when it records a safe error code.
The destination must not already exist: the client reserves it before network I/O
and never overwrites input or user files. Use a fresh response filename for retries.
`response_saved` means only that a real HTTP response was saved, **not** that its
answer/citations were accepted. No inference output is synthesized on failure.

Wire packet schema 1 is:

```text
request:  {schema:1, request_id:SHA256, provider:{base_url,model,auth_env,timeout,
           max_response_bytes,max_tokens,max_prompt_bytes}, messages:[system,user]}
response: {schema:1, request_id:the_same_SHA256, response:actual_completion_object}
       or {schema:1, request_id:the_same_SHA256, error:safe_error_code}
```

`request_id` is SHA-256 of canonical JSON (`ensure_ascii=True, sort_keys=True`,
Python default JSON separators) containing only `provider` and `messages`.
The shared client generates this; do not edit/reconstruct packets manually.
Binding catches packet mixups/edits, **not malicious forgery or server identity**.
The bound user message also carries `snapshot_sha256`, a canonical digest of the
entire retained retrieval snapshot, including its corpus revision. Replacing only
the local snapshot cannot rebind an old response to a rebuilt index. Re-prepare
bundles created before this field was introduced rather than bypassing validation.

The prompt retains all supplied approved passages and their complete structured
contexts. Repetitive coverage diagnostics are summarized: selected pages, an
unselected-page count, and gap counts/reasons. Full page/gap details remain in the
local bundle. This is metadata compaction, not instruction/warning truncation.

## Ordinary prose: matching windows and same-page context

`search(..., limit=N)` still returns at most N threshold-qualified matching windows
in `passages`, with their original text, locations, ranks and scores unchanged.
For ordinary prose indexes only, it additionally returns:

- `context_scope: "eligible_indexed_same_page_prose"`;
- `page_context`: every other eligible indexed passage on a physical PDF page of
  those matches, deduplicated against the matches and ordered by page then ordinal.
  These passages retain their exact source/passage IDs, text, section paths,
  PDF/printed locations, bounding boxes, review status and provenance. They are
  context, **not independently threshold-qualified matches**, and receive no scores.

No raw PDF re-extraction, adjacent-page expansion or eligibility override occurs.
No match means an empty context list. Selected-page/recovery exclusions remain
excluded; whole mixed blocks with unresolved or rejected recovery remain withheld.
Exercise/variation and programme indexes do not expose or use `page_context`:
existing reviewed structured-record scope remains unchanged, including withheld
same-page siblings.

`prepare_answer` and `answer` send both lists in full, assign citation IDs to
matches first and context second, and bind both lists into the existing request
packet and snapshot checks. `finish_answer` accepts citations into either list and
rejects changed retained context. All original evidence stays in the private
`request.json`; source/review/corpus revalidation still occurs before and after
inference. A myth heading and its eligible same-page refutation can therefore
reach the model together even with `limit=1`.

This means **complete eligible indexed same-page context**, not complete source
pages, lists, chapters or semantic topics. Non-indexed headings/short blocks,
unselected pages, unresolved recovery and cross-page qualifications cannot be
reconstructed or inferred. Existing page/gap disclosures must still be considered;
index-time exclusions are not repaired by retrieval. Page context may include
other sections on that page, but each passage keeps its separate section/location.
Larger prompts fail the existing byte limit rather than silently dropping text.
There is no index-format, embedding-window, matching-policy or settings change:
existing compatible indexes remain validated and usable without rebuilding.

## 3. Finish on the retrieval machine

```sh
uv run cited finish-answer --index "$PRIVATE_DATA/prose.sqlite" \
  --request-dir "$ANSWER_DIR" --response "$PRIVATE_DATA/inference-response.json"
```

This command makes no inference/network call and needs no embedding model. It
revalidates the original source, corpus, recovery review/crops and exercise/programme
dependencies against the retained snapshot **even for failed, mismatched or malformed
responses**. Each finish attempt retains separate private diagnostics without
replacing earlier attempts. Changed originals/reviews require a compatible rebuild
and new preparation, not an override of the freshness checks.

The model is instructed to return JSON content
`{"status":"answer","answer":"The source recommends rest [E1]."}` or
`{"status":"abstain","answer":"The supplied evidence does not define that term."}`.
Only one completion with `finish_reason:stop` is accepted. A present response `model`
must equal the configured advertised ID; if absent, `response_model` is null.
Malformed/truncated output is rejected, never repaired into a plausible answer.

Every non-abstaining answer needs supplied `[E1]`-style citations. Comma-separated
groups such as `[E1, E2]` are supported; every identifier is checked individually.
Unknown IDs, missing citations and other square-bracket references are rejected. A labelled
`bbox` array of exactly four signed integer/decimal coordinates is location prose,
not a citation; accepting its syntax does not verify its accuracy. IDs map back to
full evidence in `request.json`. **Membership/syntax is not claim entailment**:
`claim_support_verified` is always false. Human source/answer comparison is still
required. Conflicts (with both source claims), unresolved references, page coverage
and recovery/exercise gaps remain in deterministic `disclosures`, independently of
what the model writes. Source data/instructions are explicitly untrusted prompt
data; author claims must be distinguished from suggestions. These prompt rules
are not a semantic safety verifier.

## Integrated command (authorized route only)

Where the owner has explicitly authorized a route reachable **from the retrieval
machine**, replace `prepare-answer` with `answer` using the same flags. It shares
preparation/validation with the staged workflow and retains artifacts before HTTP.
Do not use the Windows loopback URL from another host expecting it to reach Windows.
The Windows-side route above requires no tunnel or inference-server changes.

Limits are positive/finite: timeout is a per-network-operation timeout (not a total
wall-clock deadline); prompt limit is serialized request bytes, not tokenizer/context
window knowledge; response limit applies to each HTTP body. Evidence is never
silently truncated to fit: excessive context is `inference_context_limit`. A server
context refusal is `inference_http_status`. Escaped response-file overhead is bounded
separately on finish. Choose limits consistent with the existing server, without
changing it. Errors distinguish inference connection, timeout, auth, HTTP status,
advertised/response-model mismatch, malformed protocol, byte limit, bad citations,
response binding, retrieval and post-inference revalidation. CLI error exit is 2.

## Acceptance status

Offline authored tests exercise trusted retrieval → HTTP request → response → current
source revalidation through public answer/prepare/finish interfaces. HTTP transport
responses are synthetic and are **not** live-model acceptance. Real acceptance needs
an actual owner-returned response or an authorized integrated run, followed by
finish validation and private human source comparison. Without either execution
route, real inference acceptance remains owner-action-gated and incomplete.
No network/server/credential is required by ordinary tests.

For reasoning models, the completion allowance can be consumed before a final
answer appears. A `length` finish remains a failed protocol result even if the raw
response contains reasoning. Increase `--max-tokens` explicitly for a fresh attempt
(for example, 4096), retain the original failed run, and compare the actual final
answer with the source. Do not silently truncate evidence or disable reasoning to
manufacture a successful test.
