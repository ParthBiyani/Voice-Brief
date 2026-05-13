# VoiceBrief

A personalized daily audio brief that knows what you're building — plus a
knowledge-to-podcast engine for your own documents.

> Early development. See [PRD.md](PRD.md) for scope and the roadmap.

## Quick start

```bash
make install     # editable install with all extras
make up          # Postgres, Qdrant, Redis, MinIO
make migrate     # schema
make seed        # load the source registry
make ingest      # first crawl
```

## Status

| Milestone | State |
|---|---|
| Source registry + Tier 1 ingesters | done — 560 items from 7/7 sources in one pass |
| Dedup, clustering, eval harness | in progress |
| Ranking, personalization, script | planned |
| TTS + web app | planned |

## Notes

`GITHUB_TOKEN` is optional but recommended. Unauthenticated GitHub API access is
capped at 60 requests/hour, which causes partial coverage of the watched-releases
list — the run degrades to whichever repos it reached rather than failing.
