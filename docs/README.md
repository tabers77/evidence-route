# Documentation

| Document | What it is |
| --- | --- |
| [`EVIDENCEROUTE_PROJECT_SPECIFICATION.md`](EVIDENCEROUTE_PROJECT_SPECIFICATION.md) | The specification **and** the implementation tracker |

## Where documentation lives

Documentation is deliberately split by what it governs, not gathered in one
place. Each document sits next to the thing it describes, so it is hard to
update the code and forget the prose.

| Location | Contents |
| --- | --- |
| `docs/` | The specification and implementation tracker — the whole project |
| `experiments/protocols/` | The research protocol: hypotheses, comparisons, freeze checklist |
| `reports/` | Technical report, system card, human evaluation protocol |
| `data/README.md` | Data card: sources, licenses, split policy, versioning |
| `configs/README.md` | Configuration conventions and the experiment sequence |
| `scripts/README.md` | The pipeline sequence and what each script wraps |
| `tests/fixtures/README.md` | Why fixtures exist and how to re-record them |

## Tracking implementation status

Section 0 of the specification summarises what is built; section 31 tracks it
per backlog item using this legend:

| Marker | Meaning |
| --- | --- |
| ✅ | Implemented and covered by passing tests |
| 🟡 | Partially implemented — usable, but incomplete against the spec |
| 📋 | Designed and documented, not yet implemented |
| ⬜ | Not started |
| 🔒 | Deliberately deferred or out of MVP scope |

The distinction between 📋 and ✅ is the one that matters. A written config, a
documented package or a named scorer is 📋. It becomes ✅ only when code runs and
tests cover it — otherwise the tracker drifts into describing intentions rather
than the repository.
