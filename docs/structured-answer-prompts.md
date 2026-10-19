# Structured evidence in answer prompts

Answer preparation retains the full retrieval result, source report, dependency snapshot,
review history, field provenance and extraction metadata in the private `request.json`.
It constructs separate model-facing copies; it does not edit the corpus or retained records.

For exercises, including resolved exercise records inside programme entries, the prompt:

- Keeps the entire passage text exactly, parent/variation identities and relations, aliases,
  approved instructions, warnings, references, context, quantities and source locations.
- Keeps scalar-field citations, image geometry and `visual_understanding: false`.
- Omits raw extraction `lines` only from recognized field segments with approved text,
  source ID and location. Original text, quotes, joins and unknown fields remain.
- Omits an instructions/warnings/references citation array only if it exactly equals the
  corresponding retained field array. Different citations remain, with their source locations.
- Replaces repetitive global accepted, eligible, complete, gap-free exercise inventory with
  counts in both retrieval and disclosures. Every unavailable or nonroutine record remains
  complete, including unknown warning/qualification fields. Matched records are not inventory
  and are never replaced by counts.

Programme quantities, order, pairings, method definitions, conflicts and unresolved references
remain unchanged. Ordinary prose and same-page context use their existing projection.

Structured prompts additionally bind the full original retrieval and disclosures by digest.
Finish reconstructs the projection from retained originals, checks packet identity, and still
revalidates current original source/review dependencies. Previously prepared structured packets
with the old projection fail reconstruction: prepare a fresh bundle rather than rewriting an old
request or response. Unaffected prose packets need no migration.

The existing completion-body byte guard remains authoritative. If preserving the approved
content still exceeds the configured budget, preparation fails closed; it never truncates
approved evidence or changes completion/server limits to make a request fit. Offline transport
regressions verify the actual completion-body size. They do not establish live-model grounding.
