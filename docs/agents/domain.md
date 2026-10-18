# Domain docs

This repository uses a **single-context** layout:

- Root `GLOSSARY.md` defines shared terms, when present.
- `docs/adr/` holds architecture decisions, when present.

Before exploring or specifying domain changes, read the glossary and relevant ADRs.
Use their vocabulary consistently. Flag proposed ADR conflicts explicitly rather than
silently overriding an existing decision.

If these documents do not exist, proceed without requiring their creation. Create them
lazily through domain-modeling when terms or architectural decisions are actually resolved.
Do not create empty placeholders or introduce multi-context scaffolding without need.
