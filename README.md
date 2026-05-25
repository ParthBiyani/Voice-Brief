# VoiceBrief

**A personalized daily audio brief that knows what you're building** — plus a
knowledge-to-podcast engine for your own documents.

It reads your repositories' dependency manifests and only tells you about releases
that affect code you have actually written:

> *"LangGraph 0.4 shipped durable execution. You use LangGraph in ContextPilot and
> PlacementPilot — this replaces the checkpoint workaround in both."*

---

## Evaluation

Every number below is measured by `make eval` against a frozen, hand-labelled corpus
of 500 real crawled items. The harness runs the production code paths, offline and
without an API key, and diffs against the last recorded baseline.

| Component | Metric | Result | Target | |
|---|---|---:|---:|---|
| Deduplication | pairwise precision | **1.000** | ≥ 0.90 | ✅ |
| Deduplication | pairwise recall | **0.971** | ≥ 0.80 | ✅ |
| Clustering | homogeneity | **0.811** | ≥ 0.75 | ✅ |
| Ranking | precision@10 | **0.533** | ≥ 0.70 | ❌ |
| Script factuality | sentence attribution | **1.000** | ≥ 0.95 | ✅ |
| Hallucinated links | URLs outside the source set | **0** | 0 | ✅ |
| Latency | end-to-end episode | **3.6 min** | < 6 min | ✅ |
| Cost | per episode | **₹3.77** | < ₹8 | ✅ |

Latency, cost and factuality come from a real 6-minute episode built from live crawl
data. Ranking is the **pre-LLM heuristic path only** and is reported short of target
rather than tuned to it — see [Honest limitations](#honest-limitations).

### Does personalization actually help?

`make ablation` runs the same ranker three ways on identical data:

| Arm | P@10 | nDCG@10 |
|---|---:|---:|
| GitHub stack profile + declared topics | **0.533** | **0.585** |
| Declared topics only | 0.467 | 0.453 |
| No personalization | 0.133 | 0.188 |

Personalization of any kind is worth **+0.40 P@10**. The stack profile adds a further
+0.07 on P@10 but **+0.13 on nDCG** — its real contribution is *ordering*: it puts
the right stories first, not merely in the set.

> The first version of this ablation was unfair — the control arm used a topic
> vocabulary with zero overlap with the corpus, so it could not score at all. Fixing
> it cut the stack profile's apparent advantage from +0.27 to +0.07. The smaller
> number is the true one.

---

## How it works

```
cron ──► ingest ──► filter ──► embed ──► dedup ──► cluster ──► rank ──► script ──► TTS ──► episode
        48 sources   1500→300   local     0.92     HDBSCAN    2-stage  LangGraph  Kokoro   + transcript
                                 bge      cosine    (leaf)              + verify
```

**Ingest** — 48 probe-verified sources (arXiv, Hacker News, GitHub search + releases,
Hugging Face, and 41 RSS feeds). Official APIs and published feeds only: no scraping,
no headless browsers, documented rate limits respected. Sources are database rows, so
adding one is an `INSERT`, not a deploy.

**Filter** — recency decay, source trust, engagement and topic match, plus a
per-source diversity quota. No embeddings, no tokens. This is what makes the cost
target reachable.

**Dedup & cluster** — local `bge-small` embeddings; exact URL/title matching then
cosine ≥ 0.92, grouped with union-find; HDBSCAN with **leaf** selection for stories.

**Rank** — a free heuristic pass over every cluster, then one batched LLM re-rank over
the top 15. The stack profile, built from your repos' `requirements.txt` /
`package.json` / `pubspec.yaml`, is the highest-weighted signal.

**Script** — LangGraph, one call per segment, every sentence traceable to a source
span. A verification node runs *inside* the graph, so no episode escapes unmeasured.

**Audio** — Kokoro locally, per-segment with a content-addressed cache. Transcript
timestamps are measured from the rendered audio, so seeking to a line lands on the
right words.

### Mode 2 — knowledge to podcast

Upload PDFs, Markdown, DOCX, or point it at a repository. Three things separate it
from a generic summarizer:

- **Repository ingestion** — the walker ranks modules by import centrality and
  explains the *architecture*, not files in alphabetical order.
- **Cross-document contrast** — claims are extracted per document, then compared
  across documents to surface where sources actually disagree.
- **Native Hindi** — generated in Hindi, not translated then spoken. Output is
  natural Hinglish: English technical terms, Hindi connective tissue.

---

## Using it

### Prerequisites

Python 3.11+, Docker, and Node 20+ (only for the web app).

### First run

```bash
git clone https://github.com/ParthBiyani/Voice-Brief.git && cd Voice-Brief
cp .env.example .env          # optional: add ANTHROPIC_API_KEY and GITHUB_TOKEN

make install                  # editable install with all extras
make up                       # Postgres, Qdrant, Redis, MinIO (non-default ports)
make migrate                  # schema
make seed                     # load the 48-source registry
```

Verify before going further — `make up` waits for health checks, so if this returns
cleanly the stack is ready:

```bash
docker compose ps             # all four should read "healthy"
```

### Generate your first brief

```bash
make ingest                   # crawl all due sources        → ~560 items, ~60s
make enrich                   # embed, dedup, cluster         → ~200 clusters, ~30s
make brief                    # rank, write, narrate          → episode, ~3–4 min
```

`make brief` prints the episode id, duration, cost, attribution rate and hallucinated
link count. For a personalized brief, pass your GitHub login so the stack profile is
built from your actual repositories:

```bash
python -m voicebrief.cli brief generate \
  --github ParthBiyani \
  --topics "agentic-ai,tooling" \
  --stories 6 --minutes 10
```

### Listen and read

```bash
make api                      # http://localhost:8000/docs
make web                      # http://localhost:5173
```

The web app is the intended surface: episode list, audio player, transcript that
highlights the line currently playing, click-a-timestamp to seek, thumbs per segment,
and "ask about this" scoped to that segment's sources.

### Day to day

Once it works, schedule it and forget it. `deploy/crontab` runs ingest and enrich at
06:00 and 18:00 IST. Overlapping runs are safe — a Postgres advisory lock makes the
second caller a no-op rather than a duplicate crawl.

```bash
make deploy-up                # API + scheduler + datastores
```

### CLI reference

| Command | What it does |
|---|---|
| `voicebrief sources sync` | Upsert the registry from `config/sources.yaml` (idempotent) |
| `voicebrief sources list` | Show the registry as the pipeline sees it |
| `voicebrief ingest run [--source SLUG] [--force]` | One crawl pass; `--force` ignores poll intervals |
| `voicebrief enrich [--keep N] [--threshold 0.92]` | Embed, deduplicate, cluster |
| `voicebrief brief generate [--github LOGIN] [--language hi] [--no-audio]` | Full pipeline |

### API

| Endpoint | Purpose |
|---|---|
| `POST /ingest/run` · `/ingest/enrich` | Cron targets |
| `POST /episodes/generate` → `GET /episodes/{id}/stream` | Start generation (202), follow progress over SSE |
| `GET /episodes` · `/episodes/{id}` · `/episodes/{id}/transcript` | List, detail with presigned audio URL, timestamped transcript |
| `GET /search?q=…` | Semantic search across every past episode |
| `POST /segments/{id}/chat` | Q&A answered only from that segment's sources |
| `POST /feedback` | Thumbs per segment |

Interactive docs at `/docs`.

### Running without an API key

Everything works with no key at all:

```bash
VB_LLM_PROVIDER=ollama   # local model — real output, ₹0
VB_LLM_PROVIDER=echo     # deterministic stub — for CI
VB_TTS_ENGINE=null       # correctly-timed silence, real transcript
```

`echo` is what keeps `make test` hermetic. `ollama` needs `ollama serve` and
`ollama pull qwen2.5:7b-instruct-q4_K_M`.

### Configuration

Copy `.env.example` to `.env`. Host ports deliberately avoid the defaults
(`55432`/`56333`/`56379`/`59000`) so VoiceBrief coexists with other local stacks.

| Variable | Default | Note |
|---|---|---|
| `VB_LLM_PROVIDER` | `anthropic` | `anthropic` \| `ollama` \| `echo` |
| `VB_LLM_SCRIPT_MODEL` | `claude-sonnet-5` | Opus costs ₹7.51/script vs ₹3.01 — see below |
| `VB_DAILY_BUDGET_INR` | `50` | Hard kill-switch, checked before every call |
| `VB_TTS_ENGINE` | `kokoro` | `kokoro` \| `piper` \| `null` |
| `VB_MAX_ITEMS_PER_RUN` | `1500` | Global crawl ceiling |
| `GITHUB_TOKEN` | — | Optional; 60 → 5000 req/hr. Recommended |

### Adding a source

A source is a database row, not a deploy. Add it to `config/sources.yaml` and re-run
`make seed` — the upsert is idempotent and preserves poll history:

```yaml
- slug: my-feed
  name: Some Engineering Blog
  kind: rss
  endpoint: https://example.com/feed.xml
  poll_interval_minutes: 720
  trust_weight: 0.70
  default_topics: [engineering, systems]
  config: {adapter: rss}
```

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `make brief` says "No clusters found" | Run `make ingest` then `make enrich` first |
| Only a few GitHub items appear | Unauthenticated rate limit (60/hr). Set `GITHUB_TOKEN` |
| Episode has no audio | Kokoro weights missing — `pip install 'voicebrief[tts]'`; it degrades to a silent track with a correct transcript |
| Qdrant unhealthy after an image bump | Storage is not forward-compatible; the client pin and server image must move together |
| `BudgetExceeded` | Daily ceiling hit. Raise `VB_DAILY_BUDGET_INR` or wait |

---

## Cost

`< ₹8 per episode` is measured, not estimated: every model call writes a `cost_entry`
row with tokens, latency and rupees. Three decisions got it there:

| | Per episode |
|---|---:|
| Naive (summarize all 60 clusters, Opus script) | ₹20.08 |
| Summarize only the ~15 that survive pre-ranking | ₹13.66 |
| Cache the shared system prompts | ₹9.45 |
| **Sonnet rather than Opus for the script** | **₹4.94** |

The first two are waste removal. The third is a real quality trade-off, defaulted to
meet the budget and reversible in one config line — Opus alone is ₹7.51 of the naive
figure, because script output tokens are the single largest line item in the system.

---

## Architecture

```
React (Vite) ──REST + SSE──► FastAPI ──► LangGraph ──► Postgres · Qdrant · MinIO · Redis
```

Scheduling is a cron hit to `/ingest/run` guarded by a Postgres advisory lock.
Overlapping runs are a no-op, not a duplicate crawl. Celery, Temporal and Kubernetes
are deliberately deferred.

```
src/voicebrief/
├── sources/        adapter per upstream; registry resolves kind → class
├── pipeline/       filter, embed, dedup, cluster, rank, summarize, ground
├── llm/            provider abstraction, pricing, cost ledger, budget
├── graphs/         LangGraph brief pipeline
├── documents/      Mode 2 parsing and claim graph
├── tts/            engines, assembly, timestamps
├── evalkit/        metrics, annotation rubric, harness, ablation
└── api/            FastAPI routes, SSE, episode memory
```

**371 tests** (330 unit, 41 integration), ~8,950 lines of Python.

---

## Honest limitations

- **Ranking is below target (0.533 vs 0.70).** This is the heuristic path measured on
  its own. The LLM re-rank sits on top and is not yet included in the harness number,
  because measuring it needs an API key and a harness that costs money is a harness
  nobody runs. Reported short rather than tuned to the target.
- **Grounding is lexical, not entailment.** It reliably catches fabricated
  specifics — invented version numbers, benchmark figures, product names — which is
  the failure mode that matters in a news brief. It will not catch a sentence that
  reuses source vocabulary while inverting the meaning.
- **ARI is reported but not used as the clustering metric.** The system clusters at
  *story* level while the labels are *subject-area* level, so ARI reads 0.019 even
  when every cluster is correct. Homogeneity is the metric that matches the task;
  ARI is kept in the table rather than quietly dropped.
- **Dependency matching ignores names under four characters.** `dio`, `six` and `ply`
  will never raise a story. A false match would put a confidently wrong claim into
  the audio — "you use this in ContextPilot" when you don't — so precision wins.
- **Labels are rubric-based.** The topic and relevance sets are annotated by a
  documented, deterministic rubric in `evalkit/annotation.py`, not by independent
  human raters. Reproducible and auditable; it inherits the rubric's blind spots.
- **Five of 49 candidate feeds were dropped.** Anthropic, LangChain, LlamaIndex, vLLM
  and Mistral no longer publish RSS; their GitHub releases feeds are used instead.

---

## Commands

| | |
|---|---|
| `make test` | unit suite (fast, hermetic) |
| `make test-all` | including integration against Docker services |
| `make eval` | metrics table + diff against baseline |
| `make ablation` | does personalization help? |
| `make lint` | ruff + mypy |
| `make deploy-up` | full production stack |

## License

MIT
