# Issue tracker: GitHub

Issues and specs live in `JoseHRosario/yayog-rag`. Use the authenticated `gh` CLI;
anonymous URL previews are not authoritative, especially for private repositories.

## Operations

- Create: `gh issue create --repo JoseHRosario/yayog-rag --title "..." --body-file <file>`.
- Read: `gh issue view <number> --repo JoseHRosario/yayog-rag --comments`; include labels and the full discussion.
- List: `gh issue list --repo JoseHRosario/yayog-rag` with appropriate state/label filters.
- Comment: `gh issue comment <number> --repo JoseHRosario/yayog-rag --body-file <file>`.
- Label: `gh issue edit <number> --repo JoseHRosario/yayog-rag --add-label "..."` or `--remove-label "..."`.
- Close only when authorized acceptance is verified: `gh issue close <number> --repo JoseHRosario/yayog-rag --comment "..."`.
- Fall back to `gh api` for an operation blocked by unrelated CLI/GraphQL lookups.

"Publish to the issue tracker" means create a GitHub issue. "Fetch the relevant ticket"
means read its complete body, comments and current state. Search for duplicates first.
Apply the mapping in [triage-labels.md](triage-labels.md). Read back exact bodies,
labels and relationship targets after writes. Keep private evidence out of public records.

## Relationships and wayfinding

- Declare blockers in issue bodies and use GitHub native blocking dependencies when available.
- Dependency creation uses the blocker's database ID, not its issue number; verify actual edges.
- When native dependencies are unavailable, record `Blocked by: #<number>` and check their live states.
- A wayfinder map uses `wayfinder:map`; child types use `wayfinder:<type>` as needed.
- Use native sub-issues only when parent modification is authorized; otherwise reference the parent in child bodies.
- Work the frontier: blockers complete, ticket unclaimed. Claim by assigning the driving developer.
- Do not close or modify a parent while decomposing it into tickets.

## Pull requests as a triage surface

**PRs as a request surface: no.**

PRs are implementation/review artifacts, not additional feature-request inputs to triage.
