# Agent instructions

See [README.md](README.md) for the application overview, setup and public API.

## Commands

```sh
uv sync --extra dev --locked
uv run pytest -q
uv run mypy cited_retrieval
uv run ruff check cited_retrieval tests
uv run ruff format --check cited_retrieval tests
```

## Testing and code

- Use public import/report/retrieval seams; apply red–green TDD for behaviour changes.
- Keep ordinary tests offline with authored synthetic fixtures and inference-boundary doubles.
- Report synthetic tests, private-source checks and live-model acceptance separately.
- Use type annotations and one class/interface per file, including test helpers.
- Keep CLI parsing separate from corpus operations and embedding inference; put shared policy in one place.

## Git workflow

- Use `master` as the base; one dedicated ignored `.worktrees/` checkout and feature branch per issue.
- Fetch and inspect the actual base, working tree and existing PR before editing; preserve unrelated changes.
- Stage owned paths explicitly; audit the entire published history for private content and secrets.
- Land changes through PRs; obtain owner approval before merging. Publishing or service changes need authorization.
- Agent attribution: `Meireles <332697001+xungameireles@users.noreply.github.com>`.

## Boundaries

- **Always:** retain source fingerprints, page labels, section boundaries, provenance and explicit eligibility.
- **Always:** keep private inputs, parsed text/images, corpora, models, indexes and evaluation output outside the repository.
- **Ask first:** change source trust policy, provision remote services, expose endpoints or reconfigure the Windows inference server.
- **Never:** commit secrets, private source content, its title/author/identifying filename, or derived artifacts; publish these in issues/PRs/logs; invent missing source instructions.
- Reuse the existing Windows llama.cpp/Bonsai server; do not assume coiso's localhost reaches it.

## Agent skills

- **Issue tracker:** GitHub Issues; before tracker operations read [issue-tracker.md](docs/agents/issue-tracker.md).
- **Triage labels:** default roles; before triaging read [triage-labels.md](docs/agents/triage-labels.md).
- **Domain docs:** single-context; before domain changes read [domain.md](docs/agents/domain.md).
- **Source readiness:** before extraction, chunking or trust-policy changes read [source-readiness.md](docs/agents/source-readiness.md).

## Living doc

Record recurring corrections as durable rules; prune stale guidance with the change that invalidates it. Keep progress in issues, not here.
