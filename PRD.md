# VoiceBrief: Product Requirements (v2.0)

**Owner:** Parth Biyani
**Status:** Current product, as running after the October 2026 repair (PRs 1 to 23)
**Supersedes:** PRD v1.0 (the six-week scoped build, May 2026)
**Positioning:** *A daily audio brief that knows what you are building, plus a way to turn your
own documents and repositories into a podcast.*

---

## 1. Summary

VoiceBrief produces a short spoken news brief every morning, chosen for one engineer by what is in
their own code. It reads 43 sources, removes duplicates and anything already heard,
groups items into stories, ranks them against the listener's GitHub dependencies and stated
interests, writes a script in which every sentence is checked against its source, voices it
locally, and delivers it to Slack with the audio attached, where the listener can reply in the
thread to ask about it. A second mode turns the listener's own
documents or a repository into an episode.

Version 1 met its offline evaluation targets but had never run daily for a real person. An audit in
October 2026 found stale and repeated stories, silent failures, under-counted cost and no
automation. Version 2 is the product after those were fixed. This document describes what it does
now, how success is measured, the current values, and what comes next.

---

## 2. Problem

Engineers lose time every day skimming feeds to stay current, and most of what they read does not
touch what they build. (The v1 planning estimate was 45 to 90 minutes a day; it is an estimate, not
a measurement.) Existing tools either show everything (feed readers), are written for everyone
(newsletters), or need the person to do the gathering (chat assistants). None of them know the
listener's codebase, remember what they already heard, or check their own facts.

The question the product answers every morning: **what changed today that affects the code I
write?**

---

## 3. Users

**Primary: the builder.** A student or early-career engineer actively building projects, with code
on GitHub, following many sources loosely, with daily time when their eyes are busy (commute, walk,
gym). Wants to hear what changed that affects their stack. Today this is one user: the owner.

**Secondary: the researcher.** Has a folder of papers or a repository and wants it explained, with
the places where sources disagree called out. This is Mode 2's user.

**Jobs to be done**

1. "Tell me what happened that matters to my projects, in under ten minutes, without my doing
   anything."
2. "Never tell me the same story twice, and never make something up."
3. "Explain these documents, or this repository, and where they disagree."
4. "Show me why you picked what you picked, and what it cost."

---

## 4. Goals and non-goals

### Goals

| # | Goal |
|---|---|
| G1 | A brief arrives every day with no manual step, and a failed day is announced rather than silent |
| G2 | Most stories in a brief are tied to the listener's own stack or interests |
| G3 | Every factual sentence is traceable to source text; no invented links |
| G4 | No story repeats within 30 days; only news first seen since the last brief is a candidate |
| G5 | Cost per episode stays within a few rupees and is measured per call, with a hard daily cap |
| G6 | Every decision (ranking, model call, delivery, backup) is recorded and inspectable |
| G7 | Data that cannot be re-crawled survives hardware failure |

### Non-goals (for this version)

- Real-time or streaming news; the product is one brief a day.
- A native mobile app or push notifications; Slack and a phone-friendly web app cover delivery.
- Multi-tenant sign-up, billing or hosting for other users (the schema is multi-user, the
  operation is not).
- Voice interruption mid-episode; text Q&A on a segment replaces it.
- Scraping any site without an official API or published feed.
- OCR for scanned PDFs.
- An LLM judge for factuality.
- More than two languages (English and Hindi).

---

## 5. Features (as built)

### 5.1 Daily brief (Mode 1)

| Feature | Requirement | Status |
|---|---|---|
| Source registry | Sources are database rows seeded from `config/sources.yaml`; adding one is a config change. Removed sources are disabled, not deleted | Built: 43 sources (5 API, 2 GitHub, 36 RSS), including 11 news feeds added 31 Oct |
| Article text | For sources with `fetch_article: true` (7), items with a summary under 900 chars and no body get the linked page's paragraphs (lxml); robots.txt honoured; ≤ 60 fetches/run, 6 concurrent, 10 s timeout, 2 MB cap; tried once, outcome recorded | Built: 48 of 60 fetched live; HN ~30 → ~5,000 chars |
| Polite, reliable ingest | Official APIs and feeds only; each source fetches from its last successful poll (1 h overlap, capped at 7 days); per-source error boundary and 90 s timeout; backoff capped at 24 h; arXiv fetched serially | Built |
| Visible source failures | A source whose requests are all refused fails with the reason and reset time; partial failures appear in the run summary | Built |
| Cheap filter | Recency, trust, engagement and topic score; window cut to 300 items; no source above 25% | Built |
| Dedup | Exact URL/title, then cosine ≥ 0.92 with union-find; duplicates grouped, not deleted | Built |
| Stories, not topic bags | HDBSCAN (leaf); multi-item clusters must reach mean pairwise similarity ≥ 0.76; template-like sources stay single | Built |
| Freshness | Candidate only if an item was first seen in the window: 36 h, widening up to 7 days after a gap | Built |
| No repeats | Stories cited in the last 30 days excluded, matched by source URL | Built |
| Personal ranking | Heuristic over all candidates (dependencies 3.0 each up to 3; content topics 1.2, source-tag prior 0.4; recency 2.0, 24 h half-life; trust, coverage, interests, feedback ≤ 1.5, thin-story penalty), one batched model re-rank of the top 30, blend 0.4/0.6, tail floor, at least 2 personal stories, no repeated events | Built |
| Stack profile | Dependencies from GitHub manifests (Python, JS, Dart, Go, Rust), forks excluded, common-word names need package-style evidence; cached 7 days; incomplete rebuilds keep the last good profile | Built: 196 dependencies, 36 repos |
| ProjectMind context | Interests, tools moved away from, recent projects, via its public CLI only; fail-soft; cached 24 h; its raw "files and notes" section never read | Built: 6 interests, 3 projects |
| Feedback | Thumbs become smoothed per-source and per-topic affinity, plus recent liked/disliked headlines in the re-rank prompt | Built; effect not yet measured |
| Grounded script | Summaries of cited facts; opens on the agenda (no cold open, no greeting), one segment per story, sign-off; spoken register; no speculation. Documents episodes keep a cold open | Built |
| Verification | Deterministic per-sentence check of figures, versions, names and content words; invented-link check; report stored on the episode | Built |
| Speech | Kokoro (82M) via kokoro-onnx, per-segment cache, timestamps from measured audio, MP3 via soundfile | Built |
| Honest status | `ready` only with stories, stored audio with real sound, and at least half the requested stories; otherwise `degraded` or `failed` with a reason; exit codes 0/2/1 | Built |
| Delivery | Slack bot posts a digest and the MP3 in its thread; optional email; delivered once, failures retried, failed runs announced; dates in local time | Built; verified live 31 Oct |
| Slack Q&A (verified live) | Thread reply = question about that episode (matched by the digest's Slack `ts`); new channel message = latest brief; answers only from the episode and its sources (all scripts + sources of the 3 most similar segments); recorded with channel `slack`; Socket Mode, no public URL | Built |
| Schedule | Windows Task Scheduler "VoiceBrief Daily Brief": 09:00 IST and 3 min after logon; `-NotBefore 09:00` so it is never early; idempotent and exclusive | Built |
| Backups | After every generated brief: verified database dump plus incremental audio copy, outside Docker, newest 7 kept | Built |
| History | `run_log`, `ranking_snapshot` (per-signal breakdown), every model call's prompt and response (90 days), `interaction`; CLI and web views; JSONL export | Built |

### 5.2 Your documents (Mode 2)

| Feature | Requirement | Status |
|---|---|---|
| Inputs | PDF, DOCX, Markdown, text, zip, or a folder read as a repository | Built |
| Repository walk | README and manifests first, then modules by import centrality | Built |
| Contrast | Claims extracted per document, compared only when semantically close; a "Where your sources disagree" segment | Built |
| Shared pipeline | Same graph (knowledge mode), verification, audio, status, history and ledger as the brief | Built |
| Languages | English and Hindi (native generation, Devanagari sentence splitting) | Built |
| Entry points | CLI, upload API with progress stream, web view with drag and drop | Built |

### 5.3 Web app

| Feature | Status |
|---|---|
| FastAPI serves the built React/Vite app and the API on one port | Built |
| Always on: "VoiceBrief Web" task starts `serve` at logon and 09:00 and re-fires every 10 min; `run-serve.ps1` restarts the server if it exits; both tasks launch via `conhost.exe --headless`; embedding model warmed at startup (first search used to take ~15 s) | Built |
| Episode list grouped by day; player with tap-to-seek transcript | Built |
| Audio streamed via `/episodes/{id}/audio` with HTTP Range (seekable from a phone) | Built |
| "Why these stories?", per-story "your stack" badges, degraded/failed banners | Built |
| Segment Q&A answered from the segment's source text | Built |
| Search across past episodes | Built |
| Settings (GitHub login, topics, language, format) with connection cards: GitHub profile, Delivery (Slack status), ProjectMind (recent projects) | Built |
| History page (runs, every candidate with signals, model calls and spend, questions) | Built |
| Token gate (`VB_API_TOKEN`) required to listen beyond localhost | Built |

---

## 6. Success metrics

### 6.1 Product metric

**The listener uses the brief on at least 5 days a week without forcing themselves.** Not yet
measured over a sustained period; the scheduled loop and delivery only became reliable at the end of
October 2026.

### 6.2 Quality and cost metrics (current values)

Live values are from the final run on 30 October 2026 unless noted. Costs were recorded at ₹83.5 per
US dollar; the ₹100/$ figure is shown alongside.

| Area | Metric | Target | Current | Status |
|---|---|---|---|---|
| Personal relevance | Stories tied to the listener's stack or interests | ≥ 2 of 6 (the selection floor) | **5 of 6** (earlier runs: 0 and 1 of 6) | Met (single day) |
| Factuality | Claim sentences attributable to a source | ≥ 0.95 | **0.966** (0.857 before grounding fixes) | Met (single day) |
| Factuality | Invented links | 0 | **0** across all runs | Met |
| Freshness | Already-heard stories in a brief | 0 | 0 (19 excluded on the final run) | Met |
| Cost | Per episode (model calls) | < ₹8 | **₹6.43** at ₹83.5/$ (₹7.70 at ₹100/$) | Met |
| Cost | Daily hard cap | ₹50 | `VB_DAILY_BUDGET_INR=50`, checked before every call | In place |
| Latency | Generation time | < 6 min | 353 s (5.9 min) with 7.3 min of audio | Met |
| Reliability | Brief produced or failure announced every scheduled day | 100% | Not yet measured over weeks | Open |
| Durability | Verified backup after each brief | 100% | Ran and verified on the final run | Met (single run) |
| Dedup (offline) | Pairwise precision / recall | ≥ 0.90 / ≥ 0.80 | **1.000 / 0.971** | Met |
| Clustering (offline) | Homogeneity | ≥ 0.75 | **0.814** | Met |
| Ranking (offline) | Precision@10, full path (heuristic + re-rank top 30), judge labels | ≥ 0.70 | **0.600** (nDCG@10 0.730 to 0.756, two runs) | **Not met** |
| Ranking (offline) | Heuristic only, judge / rubric labels | | 0.533 / 0.660 and 0.500 / 0.599 (before: judge 0.367 / 0.453, rubric 0.533 / 0.585) | |
| Personalisation (offline) | Ablation, judge nDCG: profile + topics / topics / none | profile > topics > none | 0.660 / 0.606 / 0.084 | Met |
| Engineering | Test suite / type errors | green / 0 | 857 tests / 0 mypy errors, enforced in CI | Met |
| Operations | Scheduled runs complete | 100% | Fixed: runs were killed ~25 s in until launched via `conhost.exe --headless`; verified with a 90 s job | Fixed |

**On the ranking miss.** There are two label sets: a keyword rubric, and a model judge (Sonnet 5.5
reading each persona's plain description, no rubric). A change is trusted only if it holds on both.
The rubric's dependency clause is a substring match that rewards the `transformers` false positive,
which is why its score dipped when that was fixed. Tuning stopped deliberately: with 3 personas and
28 judge positives, further gains would be fitting the set. Every number is in `eval/EXPERIMENTS.md`.
The next step is labels from real thumbs via `history export`.

---

## 7. Constraints and budget

| Constraint | Value | Why |
|---|---|---|
| Cost per episode | < ₹8; current ₹6.43 at ₹83.5/$ | A daily product must cost about a coffee a week |
| Daily cap | `VB_DAILY_BUDGET_INR`, default ₹50, enforced before each call | A bug can cost one call, not a night |
| Exchange rate | `VB_USD_TO_INR`, validated 50 to 200, default 100 | Prices are in dollars, the ledger in rupees |
| Models | Sonnet 5.5 ($2/$10 per M tokens) for the script; Haiku 5.5 ($0.10/$0.50) for summaries, re-rank, Q&A, claims | The script is 96% of spend; quality is paid for only where it is heard |
| Unpriced models | Refused at startup | A typo must not cost ₹0 and disarm the cap |
| Embeddings and speech | Local (bge-small, Kokoro via ONNX) | Free per run; embeddings deterministic for the eval harness |
| Python | 3.11 to 3.14 (CI on both ends) | Kokoro via ONNX because the PyTorch package stops at 3.12 |
| Sources | Official APIs and published feeds only; arXiv serial; GitHub token recommended (60 requests/hour without one) | Terms of service and politeness |
| Hosting | One Windows laptop, Docker Compose (Postgres 16, Qdrant 1.12.4 pinned, Redis 7, MinIO), Task Scheduler | One user; local speech; ProjectMind is local |
| Availability | Needs the laptop on and the user logged in; catch-up after logon; Slack Q&A and the web app only while it is on | No stored password; Docker Desktop requirement |
| Slack | Scopes `chat:write`, `files:write`, `channels:read`, `channels:history`; app-level token for Socket Mode | Q&A without exposing the laptop |
| Privacy | ProjectMind's raw "files and notes" never read; prompt and response text pruned after 90 days; web app requires a token beyond localhost | Personal data minimisation |

---

## 8. Architecture (summary)

```
Task Scheduler: Daily Brief (09:00 IST · logon catch-up · -NotBefore 09:00) · Web (serve at logon, restarts)
   └─► voicebrief daily: lock ─► reap ─► ingest ─► enrich ─► rank ─► summarize ─► write
                         ─► verify ─► voice ─► store ─► judge ─► index ─► backup ─► deliver ─► record

Postgres: relational truth, history, ledger     Qdrant: item vectors, episode memory, documents
MinIO: MP3s                                     Redis: provisioned, not yet used (future queue)
Anthropic API · GitHub API · ProjectMind CLI · Slack

Phone / browser ──► voicebrief serve (FastAPI + React, one port, token-gated)
Slack thread reply ──► Slack listener in serve (Socket Mode) ──► answer from the episode's sources
```

Full detail, including the data model and the step-by-step morning run, is in the project deep dive.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| A source changes, rate-limits or dies | Per-source isolation, backfill from last success, visible failure with reason, registry pruning |
| Silent degradation | Honest episode status with reasons, exit codes, run summaries with notes, failure announcements |
| Cost creep | Funnel caps, per-call ledger on its own transaction, pre-call budget check, priced-models-only |
| Hallucination | Facts-only writing, per-sentence verification, invented-link check, stored report; entailment planned |
| Hardware failure | Verified backups outside Docker after each brief; crashed runs closed out automatically |
| ProjectMind changes | Public CLI only, fixture-pinned parser, fail-soft, 24 h cache |
| Ranking quality unproven | History and thumbs collected for real labels; experiments documented, including failures |
| Single-laptop dependency | Catch-up trigger; production compose overlay ready for a cloud move |
| Thin news items | Article text fetched for 7 opted-in sources; paywalled or bot-refusing sites stay at one line |
| Scheduled jobs dying silently | Headless launch; scheduled jobs tested with a 90 s job; next: alarm on runs that start but never finish |

---

## 10. Roadmap

### Now (next few weeks)

- Run the scheduled brief daily and measure the product metric (days listened per week) and
  reliability (briefs produced or announced per scheduled day).
- Accumulate thumbs; build relevance labels from `history export` joined with votes; re-run the
  ranking evaluation (both label sets plus real-thumb labels), including the rejected
  semantic-similarity signal.
- Alarm on any run that logs a start and no finish.
- (Done in October: full article text for news sources, content-aware topics, a measured re-rank
  and an independent judge label set.)
- Measure the feedback signal's effect on precision once enough votes exist.
- Blind listening test: Sonnet vs Haiku for the script. Switch if Haiku passes (about 95% cost cut).
- Offline replay of cached re-rank responses so the model stage can be measured without spending
  ₹0.30 a run.

### Next (one to three months)

- Local entailment check (a small NLI model) alongside the lexical verifier, to catch inverted meaning.
- Ecosystem-qualified dependency matching (for example `pub:dio`) so short names can count safely.
- Distinct voices for the two-host style.
- Move to a small cloud VM using the existing production overlay; GitHub OAuth sign-up for a handful
  of users; S3 instead of MinIO.

### Later (scaling, triggered by measured pressure)

| Phase | Users | Change |
|---|---:|---|
| 1 | 10 to 100 | One VM, per-user daily loop, full per-user script (about ₹23,000/month at 100 users at ₹7.70 each/day) |
| 2 | 1k to 10k | Split shared story work from the per-user layer; Redis queue and workers; shared story segments and audio; CDN; partitioned Postgres |
| 3 | 100k | Autoscaled worker pools; cohort re-ranking; batch precompute; TTS pool; read replicas; Qdrant cluster; metrics, tracing, alerts |
| 4 | 1M | Sharded per-user data; multi-region; time-zone scheduling; second model provider; Slack via the Events API on a public endpoint instead of Socket Mode; SLOs |

Target variable cost at a million users on the shared-story design: about ₹7.5 per user per month
(about ₹1.5 with cohort re-ranking and templated personal lines)
(₹100/$), against about ₹231 for today's per-user design. The derivation is in the deep dive,
section 13.

### Explicitly not planned

Voice barge-in, a native app before there are other users, scraping, paid TTS, and real-time
streaming.
