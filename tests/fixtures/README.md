# Test fixtures

Small, checked-in inputs that let the default test suite and the smoke
experiment run with no network access, no API key and no cost.

| Path | Contents |
| --- | --- |
| `questions/` | A handful of questions with reference answers and evidence |
| `corpus/` | Tiny document set the fixture questions are answerable from |
| `recorded/` | Recorded provider responses, keyed by request hash |

## Why recorded responses

Most CI paths must exercise the real code path without paying for it. Recorded
responses let generation, reranking and judging run deterministically in CI,
while live-model tests are marked `llm` and triggered manually (spec section 24).

A cache miss in the smoke experiment is a hard failure, not a fallback to a live
call — otherwise CI quietly becomes a billable job.

## Re-recording

When a prompt version changes, its recorded responses are stale. Re-record
deliberately, with a live provider, and commit the result alongside the prompt
change so the fixture and the prompt version stay in sync.

Recorded responses contain no credentials: the request hash covers the prompt
and parameters, never the endpoint or key.
