# VoiceBrief

**A daily audio brief that knows what you're building, plus a way to turn your own
documents and repositories into a podcast.**

Every morning it reads the sources you care about, drops what you already heard, picks the
stories that matter to *your* stack, writes a spoken script in which every sentence is
checked against its source, voices it, and sends it to your phone.

> *"Kokoro can now clone a voice. A new recipe pairs it with OpenVoice's
> ToneColorConverter. This touches Voice-Brief directly, because you use Kokoro there."*
> — from a real brief, generated from live sources and the author's own GitHub profile.

| | |
|---|---|
| **Daily brief** | 32 live sources → ~6 stories, ~6 minutes of audio, ₹3–7, ready before you are awake |
| **Your documents** | PDFs, notes, Word files or a whole repository → an episode that explains them, with a segment on where they disagree |
| **Where you listen** | Slack (audio in the thread) or email, and a web app that works from a phone |
| **Why it is trustworthy** | Every claim is traced to a source passage; the full reasoning behind every pick is kept |

---

## A day with it

1. **07:00, on its own.** A Windows scheduled task runs `voicebrief daily`: ingest, enrich,
   rank, write, voice, deliver. A laptop that was off or asleep catches up when it wakes.
2. **A message arrives** (Slack and/or email) with the stories, why each one is yours, source
   links, and the audio attached. Or open the web app on any device.
3. **While listening**, tap a line to jump to it, give a thumbs up or down, or ask a
   question about a segment; the answer comes from that segment's sources only.
4. **Whenever**, drop your own PDFs or a repo into the app and get an episode explaining them.
5. **Later**, open *History* to see exactly why a story was picked or skipped, what each
   model call said and cost, and everything you asked.

---

## Measured, not claimed

### The pipeline (offline, reproducible: `make eval`)

Frozen, hand-labelled corpus of 500 real crawled items; production code paths, no API key.

| Component | Metric | Result | Target | |
|---|---|---:|---:|---|
| Deduplication | pairwise precision | **1.000** | ≥ 0.90 | ✅ |
| Deduplication | pairwise recall | **0.971** | ≥ 0.80 | ✅ |
| Clustering | homogeneity | **0.814** | ≥ 0.75 | ✅ |
| Ranking | precision@10 (heuristic only) | **0.533** | ≥ 0.70 | ❌ |

Ranking misses its target and is reported as such. This number is the free heuristic stage
alone; the model re-rank that sits on top (now confirmed running: it scores every
candidate) is not part of the offline harness, because measuring it needs an API key.

**Does personalization help?** `make ablation`, same ranker three ways on identical data:

| Arm | P@10 | nDCG@10 |
|---|---:|---:|
| GitHub stack profile + declared topics | **0.533** | **0.585** |
| Declared topics only | 0.467 | 0.453 |
| No personalization | 0.133 | 0.188 |

### Live runs (real sources, real model, this laptop)

| | Result |
|---|---|
| Cost per 6-story brief | **₹3.4 – ₹7.2** (Sonnet 5.5 writes, Haiku 5.5 does the bulk work) |
| Time, end to end | 45 s without audio, about 3 min with ~6 min of speech |
| Hallucinated links | **0** across every run |
| Sentences traced to a source | **0.85 – 0.96** on briefs; 1.0 on a two-document episode |
| Prompt caching | ~21% of input tokens served from cache |
| Where the money goes | script writing is **94%** of spend |

These are single-user, hand-checked runs, not a benchmark. The honest reading is "it works
and costs about a coffee a week", not a statistical claim.

---

## How it works

```
 schedule ─► ingest ─► filter ─► embed ─► dedup ─► cluster ─► rank ─► re-rank ─► script ─► verify ─► voice ─► deliver
            32 sources  recency   bge     0.92     stories     stack    model       one      every     Kokoro   Slack /
            + backfill  + trust   small   cosine   not topics  + you    (top 15)    segment  sentence  + MP3    email
```

- **Ingest.** arXiv, Hacker News, GitHub trending and watched releases, Hugging Face, and
  RSS. Official APIs and published feeds only. Each source fetches from its last successful
  poll, so a skipped run loses nothing.
- **Fresh, never repeated.** Only stories first seen since your last brief are candidates,
  and anything you already heard (matched by source URL, so re-clustering cannot hide a
  repeat) is skipped. Skip a weekend and the next brief catches up.
- **Stories, not topics.** Clusters are broken apart unless their members are genuinely about
  one thing (mean pairwise similarity ≥ 0.76). Trending-list entries stay individual.
- **Rank.** Recency, source trust, your GitHub dependencies, declared topics, and (when
  connected) what [ProjectMind](#projectmind) knows about how you work. A dependency named
  like an ordinary English word needs package-style evidence to count. A story about one
  event is picked once, however many sources covered it.
- **Script.** One spoken segment per story, written only from cited facts. Short sources
  are demoted rather than padded.
- **Verify.** Every sentence's figures, versions and names are checked against its source
  text; invented figures are caught. See [limitations](#honest-limitations).
- **Voice.** Kokoro locally (through ONNX Runtime, so it runs on current Python), encoded to
  MP3, with timestamps measured from the audio so tapping a line lands on the right words.
- **Remember.** Every run, every candidate with each ranking signal's contribution, every
  model call (prompt and response) and every question is stored. See [History](#history).

### Your documents

`python -m voicebrief.cli docs podcast notes.md paper.pdf ./my-repo`

One document becomes a walk through it, in order. A folder is read as a repository:
README and manifests first, then modules ordered by how much the rest of the code imports
them, so it explains what the project *is* before the details. With two or more documents,
claims are extracted and compared, and where sources contradict each other becomes its own
segment. Same verification, same audio, same history; nothing is read from the web.

---

## Set it up (Windows)

**You need:** Python 3.11+ (3.14 tested), Docker Desktop, Node 22 (for the web app), and an
Anthropic API key. Everything else is optional.

```powershell
git clone https://github.com/ParthBiyani/Voice-Brief.git; cd Voice-Brief
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ml,llm,docs,tts,dev]"

copy .env.example .env            # put ANTHROPIC_API_KEY in it
docker compose up -d --wait       # Postgres, Qdrant, Redis, MinIO (non-default ports)
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m voicebrief.cli sources sync
.\.venv\Scripts\python.exe -m voicebrief.cli tts setup     # one-off, ~350 MB voice model

cd web; npm ci; npm run build; cd ..                        # the web app

.\.venv\Scripts\python.exe -m voicebrief.cli daily --github YOUR_GITHUB_LOGIN
.\scripts\install-schedule.ps1                              # every morning at 07:00
```

`daily` remembers your GitHub login and topics, so the scheduled run needs no arguments.
Run it again the same day and it does nothing; `--force` makes another.

> On machines where Windows blocks the generated `.exe` launchers, use
> `python -m voicebrief.cli ...` as above. Where they work, `voicebrief ...` is the same.

**Recommended:** a GitHub token in `.env` (`GITHUB_TOKEN`, fine-grained, *Contents: read-only*).
Without one GitHub allows 60 requests an hour, which a profile of dozens of repositories
exhausts; VoiceBrief detects that, keeps your last good profile and says so, but a token
makes it complete.

### On your phone

```powershell
# in .env:  VB_API_TOKEN=pick-a-long-random-string     VB_PUBLIC_URL=http://<laptop-ip>:8000
python -m voicebrief.cli serve --host 0.0.0.0
```

Open `http://<laptop-ip>:8000` on the phone and enter the token once. Audio is streamed
through the app (seekable, never expires), so it plays from any device that can reach it.
`serve` refuses to listen on the network without a token: anyone who could reach it could
generate briefs, which cost money. For use away from home, a private network such as
Tailscale is the sensible way to reach the laptop; do not expose it to the internet.

### Delivery

| Channel | Set in `.env` |
|---|---|
| Slack | `SLACK_BOT_TOKEN` (scopes `chat:write`, `files:write`, `channels:read`) and `SLACK_CHANNEL_ID` |
| Email | `VB_SMTP_HOST`, `VB_MAIL_FROM`, `VB_EMAIL_TO`, plus user and password (Gmail: an app password) |

Then `python -m voicebrief.cli deliver test` sends a short message with no model calls. A
brief is delivered once; a failed delivery is retried next time; and a failed *run* sends a
short "could not make today's brief" message, so silence never means confusion.

---

## Using it

| | |
|---|---|
| `daily` | The whole morning loop; idempotent; what the scheduler runs |
| `serve` | The web app and API on one port |
| `docs podcast <files or folders>` | Your own material as an episode |
| `history show [--date]` / `history export` | What happened and why; JSONL for analysis |
| `personal show [--refresh]` | What ProjectMind has told VoiceBrief about you |
| `deliver status / test / episode <id>` | Slack and email |
| `brief generate`, `ingest run`, `enrich` | The same steps, individually |
| `sources sync / list` | The source registry |
| `tts setup / test` | The voice model |

Exit codes from `daily`: **0** done or nothing to do, **2** a brief was made but is
degraded (for example silent audio), **1** failure. Logs: `%LOCALAPPDATA%\VoiceBrief\logs`.

### What "ready" means

An episode is `ready` only with stories, stored audio that has sound, and a reasonable share
of what was asked for. Otherwise it is `degraded` (a usable transcript, something missing,
reason stored) or `failed`. A run cut short by a crash is marked `interrupted` the next time
it starts. A brief never silently passes for complete.

### History

Every run is recorded: per-source ingest results, what was considered and why it was picked,
skipped or already heard (each ranking signal's contribution and the model's stated reason),
every model call with its full prompt, response, tokens and cost, and every question you
asked and the answer you got. Prompt and response text is kept 90 days (`VB_HISTORY_RETENTION_DAYS`);
costs and metadata are kept. `history export` writes flat JSONL, which with your thumbs
makes labelled examples of what you found relevant.

### Backups

Episodes, history, feedback and the cost ledger cannot be re-crawled, and they live inside
Docker's disk image, which a hard shutdown can corrupt (it happened). So every `daily` run
that generates a brief ends with a dump of the database and a copy of any new audio into
`%LOCALAPPDATA%\VoiceBrief\backups` (`VB_BACKUP_DIR`), keeping the newest seven
(`VB_BACKUP_KEEP`). Each dump is verified before it counts, and a failed backup is recorded
in the run's history without failing the brief.

```powershell
.\.venv\Scripts\python.exe -m voicebrief.cli backup create    # now, on demand
.\.venv\Scripts\python.exe -m voicebrief.cli backup list
.\.venv\Scripts\python.exe -m voicebrief.cli backup restore   # newest dump; stop serve/daily first
```

### ProjectMind

[ProjectMind](https://github.com/ParthBiyani/ProjectMind) is a separate project: a memory of
how its owner works and what they have been building. VoiceBrief can ask it three questions
a day and use the answers to rank (tools you reach for up, tools you left behind down) and
to brief the writer, which may then say "this is the tradeoff you made in another project".
It uses only ProjectMind's public CLI, never its database, and degrades to "no personal
context" if it is absent or changes. Set `VB_PROJECTMIND_CMD` to enable it, then
`personal show --refresh`.

### Configuration

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | | Required for real briefs |
| `VB_LLM_SCRIPT_MODEL` / `VB_LLM_UTILITY_MODEL` | `claude-sonnet-5-5` / `claude-haiku-5-5` | Writing and bulk work |
| `VB_DAILY_BUDGET_INR` | `50` | Hard stop, checked before every model call |
| `VB_TTS_ENGINE` | `kokoro` | `kokoro` \| `piper` \| `null` (silent, for tests) |
| `VB_TIMEZONE` | `Asia/Kolkata` | What "today" means |
| `GITHUB_TOKEN` | | Recommended (see above) |
| `VB_API_TOKEN`, `VB_PUBLIC_URL` | | For the web app beyond localhost |
| `SLACK_*`, `VB_SMTP_*` | | Delivery |
| `VB_PROJECTMIND_CMD` | | Enable ProjectMind |
| `VB_HISTORY_RETENTION_DAYS` | `90` | How long prompt text is kept |
| `VB_BACKUP_DIR`, `VB_BACKUP_KEEP` | per-user folder, `7` | Where backups go, and how many dumps to keep |

`VB_LLM_PROVIDER=echo` runs everything with a deterministic stub, for tests. `ollama` runs a
local model for ₹0.

### Adding a source

A source is a database row. Add it to `config/sources.yaml` and re-run `sources sync`
(idempotent; sources removed from the file are disabled, not deleted). `cluster: solo` in a
source's config marks a trending-list source whose entries are each their own story.

---

## Under the hood

```
React (Vite)  ──REST + SSE──►  FastAPI  ──►  Postgres · Qdrant · MinIO · Redis
                                  │
 Windows Task Scheduler ─► voicebrief daily ─► ingest · enrich · rank · write · voice · deliver
```

```
src/voicebrief/
├── sources/        one adapter per upstream; the registry resolves kind -> class
├── pipeline/       ingest, filter, embed, dedup, cluster, rank, summarize, ground, daily, episodes
├── graphs/         the writing graph (brief and knowledge modes)
├── documents/      parsing, claim extraction, cross-document contrast
├── llm/            providers, pricing, cost ledger, budget
├── personalization/  GitHub profile (cached), ProjectMind adapter
├── delivery/       digest, Slack, email
├── tts/            speech engines, assembly, timestamps
├── history.py      the audit trail
├── evalkit/        offline metrics and the ablation
└── api/            routes, progress streaming, audio, settings, hosting
```

**~760 tests** run on every push against real Postgres and Qdrant on Python 3.11 and 3.14,
alongside lint and the web build. `make test` runs the hermetic subset; `make test-all`
adds integration tests against the Docker services.

---

## Honest limitations

- **Ranking is below target offline (0.533 vs 0.70).** That is the free heuristic stage.
  The model re-rank runs on top and visibly improves what gets picked, but it is not in the
  offline harness. The eval labels come from a documented rubric, not independent raters;
  `history export` plus your thumbs is the route to real labels.
- **Your thumbs are a gentle nudge, not yet a learned model.** Each vote becomes a smoothed
  per-source and per-topic affinity worth at most 1.5 points (a matched dependency is 3), and
  your latest liked and disliked headlines are shown to the re-rank. It needs a few days of
  real votes to show, and it has not been measured against held-out labels.
- **Verification is lexical, not entailment.** It reliably catches invented figures,
  versions and names, and tolerates faithful rewording (units, plurals, spoken numbers). It
  cannot catch a sentence that reuses a source's words while inverting its meaning. Small
  spoken counts (up to ten, such as "two releases") are not held to the strict figure check,
  a deliberate trade against flagging every sentence that counts what is in front of it.
- **The GitHub profile includes transitive dependencies** from frozen requirements files, so
  a few names are noise; common-word names are filtered, but it is not perfect.
- **Slack and email are tested against mocked services, not live ones.** The first real run
  is `deliver test`. ProjectMind was integrated against its documented output and fixtures,
  not run live, because it was being changed in parallel.
- **The web app was typechecked, built and exercised over real HTTP but not viewed in a
  browser** by the automation that built it. Look at it once.
- **It needs your laptop on and logged in.** The scheduled task runs while you are signed in
  (no password is stored) and catches up after a missed start. A phone can only reach it
  while the laptop is on and reachable.
- **A hard power-off can cost the last minutes of database writes** on Docker Desktop. One did,
  during development; migrations were re-applied and nothing else was lost. Runs a crash left
  `running` are closed out automatically.
- **Mode 2** cannot read scanned PDFs (no OCR), and a very short note is rejected rather than
  padded. `mypy` is not clean (65 pre-existing findings) and is not enforced in CI.
- The exchange rate for rupee costs is a constant in `llm/pricing.py`. Update it when it matters.

## License

MIT
