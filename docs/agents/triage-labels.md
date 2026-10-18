# Triage labels

Use these exact GitHub label strings for the five canonical roles:

| Role | Label | Meaning |
| --- | --- | --- |
| Needs evaluation | `needs-triage` | Maintainer must evaluate the request |
| Needs clarification | `needs-info` | Waiting for required information |
| Agent-ready | `ready-for-agent` | Specified and ready for agent implementation |
| Human-ready | `ready-for-human` | Requires human implementation |
| Declined | `wontfix` | Will not be actioned |

Create a missing label when an authorized operation needs it; preserve unrelated labels.
An agent-ready dependent ticket can still be blocked: readiness does not erase dependency edges.
