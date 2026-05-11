# VoiceBrief — PRD v1.0 (Scoped Build)

**Owner:** Parth
**Status:** Draft for build
**Target:** Shippable v1 in 6 weeks, portfolio-grade with evaluation evidence
**Positioning (one line):** *A personalized daily audio brief that knows what you're building — plus a knowledge-to-podcast engine for your own documents.*

---

## 1. Scope Decisions (read this first)

| Decision | Choice | Rationale |
|---|---|---|
| Modes | Both, but Mode 1 is the hero | Mode 2 is genuinely useful and shares 60% of the backend; it just isn't the pitch |
| Crawl model | Scheduled batch (2x/day), not continuous | Cost and complexity collapse; user-visible difference is near zero |
| Interruption | Deferred to v2 | Replaced by transcript + segment Q&A (text) in v1 |
| Languages | English + Hindi | Hindi is the real differentiator vs. NotebookLM |
| Personalization | GitHub-derived context, v1 | Highest demo value per unit of effort |
| Evaluation | Mandatory, ships with v1 | This is the single highest hiring signal in the project |
| Deployment | Web app (React) first, Flutter later | Recruiters click links, not APKs |
| Dropped | LinkedIn scraping, ChatGPT/Claude memory import, Kubernetes, multi-speaker voice cloning, 8 podcast styles | ToS risk, non-existent APIs, or v1 scope bloat |

**Non-goals for v1:** real-time crawling, mid-episode voice interruption, mobile apps, more than 2 languages, multi-tenant billing, more than 2 podcast styles.

---

## 2. Problem & Users

**Problem:** Technical practitioners lose 45–90 min/day skimming feeds to stay current, and most of what they read is irrelevant to what they're actually building. Existing summarizers are per-item and generic; they don't know the user.

**Primary persona — "The builder"**
Final-year CS student or 0–3 yr engineer, actively building side projects, follows 10+ sources loosely, commutes or works out daily. Wants to know *what changed that affects my stack*.

**Secondary persona — "The researcher"**
Grad student or applied scientist with a folder of papers. Wants cross-document synthesis in audio form. (This is Mode 2's user.)

**Jobs to be done**
1. "Tell me what happened that matters to my projects, in 12 minutes, while I walk."
2. "I have 6 papers and 2 hours of commute. Explain how they relate."
3. "You mentioned something about memory architectures last week — find it."

---

## 3. Source Strategy (expanded)

The rule for every source: **official API or published feed only. No HTML scraping, no ToS violations, no headless browsers.** This is both an engineering and a hireability decision — "I respected rate limits and ToS" is a good interview answer.

### Tier 1 — Ship in Week 1 (zero-auth, high signal)

| Source | Access | What it gives | Volume/day |
|---|---|---|---|
| arXiv | Official API (Atom) | cs.AI, cs.CL, cs.LG, cs.CV, cs.SE new submissions | 300–600 |
| Hacker News | Firebase API (official, free) | Top/best stories + comment counts as signal | 100–200 |
| GitHub Trending | `gharchive` / trending via GitHub Search API sorted by stars + date | New and fast-growing repos | 50–150 |
| GitHub Releases | GitHub REST API, watched repos | Version releases for the user's actual dependencies | 10–40 |
| Papers with Code | Public API | Papers with implementations + benchmark deltas | 30–80 |
| Hugging Face Hub | Official `huggingface_hub` API | New/trending models and datasets | 50–200 |

### Tier 2 — Ship in Week 2 (RSS/Atom, one uniform ingester)

A single generic RSS ingester unlocks all of these at near-zero marginal cost. This is why RSS is the highest-leverage source type in the whole system.

- **Company engineering & research blogs:** OpenAI, Anthropic, Google DeepMind, Google AI, Meta AI, Microsoft Research, Mistral, Cohere, Stability, NVIDIA Developer, AWS ML, Netflix Tech, Uber Eng, Stripe Eng, Cloudflare, Figma Eng
- **Framework/tooling release notes:** LangChain/LangGraph, LlamaIndex, vLLM, Ollama, Hugging Face Transformers, PyTorch, FastAPI, Flutter, Qdrant, Weaviate, Pinecone, Supabase, DuckDB
- **Newsletters with public feeds:** Import AI, The Batch (DeepLearning.AI), Ahead of AI (Sebastian Raschka), Hugging Face blog, Simon Willison's blog, Interconnects, The Gradient
- **Standards & specs:** MCP spec changelog, A2A protocol repo, OpenAI/Anthropic API changelogs
- **Aggregators:** Lobste.rs, r/MachineLearning + r/LocalLLaMA + user-chosen subreddits (Reddit's official JSON API with registered app + rate limiting)

### Tier 3 — Ship in Week 4 (needs extra processing)

| Source | Approach | Notes |
|---|---|---|
| YouTube | YouTube Data API for channel uploads + `youtube-transcript-api` for captions | Restrict to a curated channel list (Yannic Kilcher, Two Minute Papers, conference channels, framework channels). Transcript only, never audio download. |
| Podcasts | Public RSS + Whisper on episodes only when title/description passes a relevance gate | Expensive — gate hard |
| Docs changelogs | Git-diff on public docs repos (LangGraph, MCP, FastAPI docs are all in GitHub) | Cheap, high signal, almost nobody does this — good differentiator |
| Conference proceedings | NeurIPS/ICML/ICLR/ACL OpenReview API | Bursty, seasonal |
| Product Hunt | Official API | Startup/product launch signal |
| Crunchbase / funding | Free-tier API or TechCrunch RSS filtered | Only if user opts into "startup" topic |

### Source configuration model

Sources are **not hardcoded**. Store them in Postgres:

```
source(id, name, kind[api|rss|github|youtube], endpoint, auth_ref,
       poll_interval, default_topics[], trust_weight, enabled)
```

Users subscribe to *topics*; topics map to source subsets. This means adding source #47 is a database row, not a deploy. **Design this on day one** — it's what makes "40+ sources" a defensible claim rather than 40 hardcoded scripts.

**Budget guardrail:** cap ingestion at ~1,500 items/day globally. Cheap filters (keyword + recency + source trust weight) cut this to ~300 before any LLM touches it. Only ~40 items reach the ranking LLM. Only ~10 reach the script.

---

## 4. Mode 1 — Daily Brief (Hero Feature)

### Pipeline

```
Scheduled ingest (06:00 / 18:00 IST)
   → Normalize to common Item schema
   → Cheap filter (recency, keyword, source trust)     [1500 → 300]
   → Embed + near-duplicate detection (cosine > 0.92)  [300 → 220]
   → Cluster (HDBSCAN on embeddings)                   [220 → ~60 clusters]
   → Cluster summarization (1 LLM call per cluster)
   → Personalized ranking (user context + LLM scoring) [60 → 8-12]
   → Script generation (LangGraph, one pass, styled)
   → TTS (Kokoro local, per-segment, cached)
   → Episode assembly + transcript with timestamps
```

### Personalization inputs (v1)

1. **Declared topics** — user picks from a taxonomy (Agentic AI, Edge AI, Flutter, Systems, etc.)
2. **GitHub context** — with a user-provided token, read: their repos' languages, `requirements.txt` / `package.json` / `pubspec.yaml` dependencies, starred repos, recent commit messages. Build a *stack profile*.
3. **Feedback signal** — thumbs up/down per segment, stored and fed into the ranking prompt as few-shot examples.

The stack profile is what produces the money line:
> "LangGraph 0.4 shipped durable execution. You use LangGraph in ContextPilot and PlacementPilot — this replaces the checkpoint workaround in both."

### Episode structure

Cold open (10s) → agenda (20s) → 6–10 stories (60–120s each) → "also noted" rapid-fire (60s) → sign-off. Target 10–14 min.

### Styles (v1: two only)
- **Solo anchor** — single voice, brisk, news register
- **Two-host** — question/answer dynamic, better for explaining hard concepts

---

## 5. Mode 2 — Knowledge-to-Podcast

Kept because it's genuinely useful, it reuses the chunking/embedding/retrieval/script/TTS half of the stack, and it makes the product useful on day one before enough sources are wired up.

**How to make it not-a-NotebookLM-clone (pick these three):**
1. **Hindi/Hinglish output** — NotebookLM's non-English support is weaker, and your Indic NLP background makes this credible.
2. **Repository ingestion** — point it at a GitHub repo URL; it walks the tree, reads the README, key modules, and dependency graph, and explains the *architecture*, not just files. NotebookLM cannot do this.
3. **Cross-document contrast mode** — an explicit "where do these sources disagree?" episode type, driven by a claim-extraction step rather than plain summarization.

**Pipeline**

```
Upload (PDF / MD / ZIP / repo URL / .txt / .docx)
   → Parse (PyMuPDF, markdown-it, tree walk for repos)
   → Chunk (structure-aware: sections for papers, files for repos)
   → Embed → Qdrant (per-document collection namespace)
   → Claim/concept extraction per document
   → Cross-document graph (agreements, contradictions, prerequisites)
   → Outline generation
   → Script generation (same generator as Mode 1, different context builder)
   → TTS → Episode
```

**Limits for v1:** max 10 documents or 1 repo per episode, 200 pages total, 50 MB.

---

## 6. Shared Systems

### Interactive layer (v1 — text, not voice)
- Full transcript with clickable timestamps
- "Ask about this segment" — opens a chat scoped to that segment's source documents
- "Explain like I'm five" / "show me the code" / "compare with X" as one-tap prompts
- Answers are text + optionally TTS'd and appended as a bonus clip

This is ~5% of the effort of true barge-in and demos just as well in a screenshot.

### Episode memory & search
Every episode's transcript, segments, and source links go into Postgres + Qdrant. Semantic search across all past episodes: *"which paper about long-term memory did you mention last week?"* → returns the segment, timestamp, and source URL.

### Multilingual
Script generated in English, then either (a) generated natively in Hindi with a language-specific prompt, or (b) translated then TTS'd. **Prefer (a)** — translated-then-spoken audio sounds wrong. Use IndicTTS or Kokoro's multilingual voices; evaluate both.

---

## 7. Evaluation Plan (the resume centerpiece)

This is the part that turns the project from "another RAG app" into evidence of engineering judgment. Build the harness before the UI is pretty.

### Datasets to construct (do this in Week 2)
- **Dedup set:** 500 items from one week of real crawls, hand-labeled into duplicate groups
- **Cluster set:** same corpus, hand-labeled topic clusters
- **Relevance set:** 200 (user-profile, item) pairs labeled relevant / not, across 3 synthetic personas
- **Factuality set:** 100 script sentences with their source spans, labeled supported / unsupported

### Metrics to report in the README

| Component | Metric | Target |
|---|---|---|
| Deduplication | Pairwise precision / recall | P ≥ 0.90, R ≥ 0.80 |
| Clustering | Adjusted Rand Index vs. hand labels | ≥ 0.65 |
| Ranking | nDCG@10, Precision@10 | P@10 ≥ 0.70 |
| Script factuality | % sentences attributable to a source span | ≥ 0.95 |
| Hallucinated links | Count of URLs not in source set | 0 |
| Latency | End-to-end episode generation | < 6 min |
| Cost | Per episode (LLM + TTS) | < ₹8 |

### Regression harness
A `make eval` command that runs the full pipeline against the frozen labeled set and prints a comparison against the last recorded run. Every prompt change gets a before/after. **Screenshot this table for your portfolio.** It is worth more than any UI.

### Ablations worth running
- With vs. without GitHub stack profile (does personalization actually help ranking?)
- Embedding dedup vs. LLM dedup (cost/quality tradeoff)
- Cluster-then-summarize vs. summarize-then-cluster

---

## 8. Architecture & Stack

```
React (Vite) web app
        │  REST + SSE
        ▼
FastAPI
   ├── /ingest     (source registry, scheduled runs)
   ├── /episodes   (generate, list, stream audio)
   ├── /chat       (segment Q&A)
   └── /search     (episode memory)
        │
   LangGraph orchestrator
   (ingest graph, brief graph, doc graph)
        │
   ┌────┴────┬──────────┬──────────┐
Postgres   Qdrant    Object store  Redis
(items,   (embeds,   (audio,       (cache,
 sources,  episode    uploads)      job locks)
 episodes) memory)
```

**Deliberately deferred:** Kubernetes, Celery/Temporal (a `cron` → FastAPI endpoint with a Postgres advisory lock is enough for v1), microservices split, Flutter.

**LLMs:** BYOK. Claude for script generation (best long-form register), a small/cheap model for cluster summarization and ranking. Every LLM call goes through one wrapper with retries, token accounting, and a cost ledger written to Postgres — so the cost metric above is real, not estimated.

**TTS:** Kokoro local (free, good quality, runs on your 3060). ElevenLabs behind a feature flag for a single showcase episode.

**Observability:** LangSmith or Langfuse from day one. Trace every graph run. This is itself a hiring signal — "evaluation and observability" was one of your identified differentiators.

---

## 9. Roadmap

### Week 1 — Skeleton + Tier 1 sources
- Repo, Docker Compose (Postgres, Qdrant, Redis, MinIO), FastAPI skeleton
- `source` registry schema + generic `Item` normalization
- Ingesters: arXiv, Hacker News, GitHub Search/Releases, Papers with Code, HF Hub
- Store raw items; no LLM yet
- **Exit:** `python -m voicebrief.ingest` pulls 500+ items/day into Postgres

### Week 2 — Dedup, clustering, and the eval harness
- Embeddings (BGE or `all-MiniLM` locally) → Qdrant
- Near-dup detection + HDBSCAN clustering
- Generic RSS ingester → all of Tier 2 (~40 feeds)
- **Hand-label the eval sets.** This is a full day. Do not skip it.
- `make eval` harness printing the metrics table
- **Exit:** measured dedup precision/recall and ARI on real data

### Week 3 — Ranking, personalization, script
- GitHub stack profile builder
- LLM ranking with user context; nDCG measured against the labeled relevance set
- LangGraph script generation, solo-anchor style, English
- Cost ledger + LangSmith tracing
- **Exit:** a readable, factually-grounded 12-minute script generated end-to-end

### Week 4 — Audio + web app
- Kokoro TTS, per-segment with caching, episode assembly
- React app: episode list, player, timestamped transcript, thumbs up/down
- Tier 3: YouTube transcripts, docs-changelog diffing
- **Exit:** you listen to your own real brief on your commute

### Week 5 — Mode 2 + Hindi
- Document/repo upload, structure-aware chunking, cross-document claim graph
- Repo ingestion (the differentiated bit)
- Hindi script generation + TTS; side-by-side quality eval
- **Exit:** upload 5 papers → get a coherent contrast episode; generate one Hindi brief

### Week 6 — Memory, polish, ship
- Episode semantic search, segment Q&A chat
- Scheduling (morning/evening), two-host style
- Deploy (Railway/Fly/single EC2), demo video, README with the eval table
- **Exit:** public URL, 2-minute demo video, evaluation results in the README

### v2 backlog (explicitly not now)
Voice barge-in, Flutter app, more styles, podcast/Whisper ingestion, multi-user billing, ContextPilot deep integration, additional Indic languages.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Source APIs rate-limit or change | Source registry + per-source adapter isolation; failures degrade gracefully, never block the episode |
| LLM cost creep | Hard item caps per stage; cost ledger with a daily budget kill-switch |
| Script hallucination | Every sentence must cite a source span; factuality metric gates releases |
| Audio quality disappoints | Evaluate Kokoro vs. Piper vs. OpenAI TTS in Week 4, pick on a blind listen test |
| Scope creep back to the original brief | This document is the contract. v2 backlog exists to absorb ideas without absorbing time. |
| Hindi TTS sounds robotic | Test early (Week 2, not Week 5) with a throwaway script; if unusable, downgrade to English-only and say so honestly |

---

## 11. Success Criteria

**Product:** You personally listen to your own brief 5 days a week without forcing yourself. If you don't, the ranking is wrong and no amount of polish fixes it.

**Portfolio:** README opens with the evaluation table, a 2-minute demo video, and one sentence that lands in an interview — *"it reads my repos' dependency files and only tells me about releases that affect code I've actually written."*

**Interview readiness:** You can answer, without notes: how dedup works and its measured precision, why you chose batch over streaming and what it saved, what your per-episode cost is, and what your ablation showed about whether personalization actually helped.
