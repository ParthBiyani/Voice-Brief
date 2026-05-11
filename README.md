# VoiceBrief

A personalized daily audio brief that knows what you're building — plus a
knowledge-to-podcast engine for your own documents.

> Early development. See [PRD.md](PRD.md) for scope and the roadmap.

## Quick start

```bash
make install     # editable install with all extras
make up          # Postgres, Qdrant, Redis, MinIO
make migrate     # schema
make test
```

## Status

| Milestone | State |
|---|---|
| Source registry + Tier 1 ingesters | in progress |
| Dedup, clustering, eval harness | planned |
| Ranking, personalization, script | planned |
| TTS + web app | planned |
