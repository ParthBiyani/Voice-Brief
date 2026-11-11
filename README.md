# VoiceBrief

[![CI](https://github.com/ParthBiyani/Voice-Brief/actions/workflows/ci.yml/badge.svg)](https://github.com/ParthBiyani/Voice-Brief/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue)
![Types](https://img.shields.io/badge/mypy-clean-brightgreen)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A daily audio brief that knows what you're building.** Every morning it reads 44 tech
sources, drops what you've already heard, picks the stories that matter to *your* stack,
writes a script in which every claim is checked against its source, voices it in a natural
human voice (English, Hindi or a Hindi-English mix, one host or two), and sends it to Slack,
where you can ask it follow-up questions. It is as long as the news deserves: a minute or
two on a quiet day, up to 15 on a big one.

![The VoiceBrief web app: today's episode with player, cost, sourcing and transcript](docs/images/web-app.jpg)

> *"Kokoro can now clone a voice … This touches Voice-Brief directly, because you use
> Kokoro there."* (from a real brief, generated from live sources and the author's GitHub)

**Run it yourself:** [set it up on your own machine](#set-it-up-on-your-own-machine) takes one
script and an API key.

## Highlights

| | |
|---|---|
| **Personal** | Ranks by your GitHub dependencies, declared topics, your thumbs, and (optionally) [ProjectMind](https://github.com/ParthBiyani/ProjectMind)'s memory of how you work. 5 of 6 stories matched the owner's stack on the latest run |
| **Fresh, never repeated** | Only what's new since your last brief; anything already aired is skipped, even if re-clustered |
| **Grounded** | Every sentence's figures, versions and names are checked against the source; 0 invented links across every recorded run |
| **Hands-off** | Task Scheduler (Windows) or cron runs it at 09:00 and catches up after sleep; the web app stays up on its own |
| **Ask anything** | Reply in Slack or tap a line in the web app; answers come only from that brief's sources |
| **Your documents too** | PDFs, notes or a whole repo become an episode, with a segment on where they disagree |
| **Worth the time** | Big news first (whether or not it touches you), then a rising repository, then what changes your own stack, and only when the impact is real. Small library updates are left out |
| **Cheap** | A few rupees on a short day, about ₹15 at the 15-minute ceiling, capped by a daily budget checked before every model call |

## How it works

```
 09:00 ─► ingest ─► filter ─► embed ─► dedup ─► cluster ─► rank ─► re-rank ─► write ─► verify ─► voice ─► deliver
          43 feeds   fresh +   bge-     0.92     stories    stack,   LLM on     one      every     voice     Slack +
          & APIs,    trusted   small    cosine   not topics content  the top    segment  claim     → MP3    web app
          full text                                         topics,  30         a story  checked
                                                            thumbs
```

| Layer | Choice |
|---|---|
| Sources | arXiv, Hacker News, GitHub (trending + your dependencies' releases), Hugging Face, AI lab and engineering blogs, tech news (Techmeme, TechCrunch, The Verge, Ars Technica, WIRED, MIT Tech Review, InfoQ, Inc42 …). Official APIs and published feeds; where a feed gives one line, the linked article's text is fetched (robots.txt honoured) |
| Pipeline | Python 3.14 · LangGraph · sentence-transformers (bge-small) · HDBSCAN |
| Models | Claude Sonnet 5.5 writes the script; Haiku 5.5 does ranking, summaries and Q&A (both configurable) |
| Speech | Microsoft neural voices by default (free); OpenAI, ElevenLabs or local Kokoro by setting. Spoken sentence by sentence with pauses that follow the speech, encoded to MP3 |
| Storage | Postgres 16 · Qdrant · MinIO (audio) · all in Docker, published on localhost only |
| App | FastAPI + React/Vite, served together on one port |
| Delivery | Slack bot (digest, audio in thread, Q&A over Socket Mode) · optional email |

Deep dive: [PRD.md](PRD.md) for requirements and success metrics; [eval/EXPERIMENTS.md](eval/EXPERIMENTS.md) for what was tried and rejected.

## Results

**Offline, reproducible** (`make eval`; a frozen, hand-labelled corpus of 500 crawled items, no API key):

| Component | Metric | Result | Target |
|---|---|---:|---:|
| Deduplication | pairwise precision / recall | **1.000 / 0.971** | 0.90 / 0.80 ✅ |
| Clustering | homogeneity | **0.814** | 0.75 ✅ |
| Ranking, full path (heuristic + model re-rank) | precision@10 / nDCG@10, independent judge labels | **0.600 / 0.74** | 0.70 ❌ |
| Ranking, heuristic stage | precision@10, judge / rubric labels | 0.533 / 0.500 | |
| Personalization ablation | nDCG@10, judge labels: stack + topics / topics / none | **0.660** / 0.606 / 0.084 | |

**Live** (real sources, real models, one laptop):

| | |
|---|---|
| Claims traced to a source | **0.966** on the latest brief (0.857 before the verifier fixes) |
| Stories matched to the listener | **5 of 6** (0–1 before personal context was wired in) |
| Cost | ₹6.4 recorded at ₹83.5/$, which is **₹7.7 at ₹100/$**; the script is ~95% of it |
| Time | ~45 s without audio, ~3 min with 6–7 min of speech |
| Invented links | **0** |

These are single-user runs, not a benchmark. Ranking misses its target and says so. It is
measured on two label sets, a keyword rubric and an independent model judgement, because
each alone flatters a different kind of ranker. The details and what was tried are in
[eval/EXPERIMENTS.md](eval/EXPERIMENTS.md).

## Set it up on your own machine

VoiceBrief runs entirely on your computer: your data, your keys, your schedule. You need
**Docker** (for Postgres, Qdrant and MinIO), **Python 3.11+**, **Node 20+** (only for the web
app) and an **Anthropic API key**. Windows, macOS and Linux all work.

### 1. Install

```powershell
# Windows (PowerShell)
git clone https://github.com/ParthBiyani/Voice-Brief.git; cd Voice-Brief
.\scripts\setup.ps1
```

```bash
# macOS / Linux
git clone https://github.com/ParthBiyani/Voice-Brief.git && cd Voice-Brief
./scripts/setup.sh
```

The script creates a virtual environment, installs the package, creates `.env` from
`.env.example`, starts the three services in Docker, applies the database migrations,
registers the news sources, builds the web app and ends by running `voicebrief doctor`.
It can be run again at any time. Pass `-SkipWeb` / `--skip-web` if you only want the CLI and Slack.

### 2. Add your key

Open `.env` and set `ANTHROPIC_API_KEY`. Strongly recommended: a `GITHUB_TOKEN`
(a fine-grained token with *Contents: read-only* is enough), which lifts GitHub's limit from
60 to 5,000 requests an hour. Then check everything:

```bash
voicebrief doctor        # one line per requirement, and what to type if one is missing
```

### 3. Make your first brief

```bash
voicebrief daily --github YOUR_GITHUB_LOGIN --topics "agentic-ai,tooling"
voicebrief serve         # the web app at http://localhost:8000
```

Your GitHub login is how the brief learns what you build: it reads the dependency files of your
public repositories (and private ones, if your token allows) and ranks what is new against
them. Language, format and topics can be changed any time on the web app's *Settings* page.
`daily` does nothing if today's brief already exists (`--force` makes another), and exits
**0** ok, **2** degraded, **1** failed.

### 4. Schedule it

| System | How |
|---|---|
| Windows | `.\scripts\install-schedule.ps1` runs the brief at 09:00 (and catches up after sleep) and keeps the web app up |
| macOS / Linux | `crontab -e`, then `0 9 * * * /path/to/Voice-Brief/scripts/run-daily.sh >> ~/.local/share/voicebrief/daily.log 2>&1` |

Your computer must be on (and Docker running) at that time; a missed day is caught up on
the next run, and a Slack alarm tells you if a brief could not be made.

## Choose how it sounds

| Setting | Options |
|---|---|
| **Voice engine** (`VB_TTS_ENGINE`) | `edge` (default): Microsoft's neural voices, free, no key, the most human-sounding without paying. Needs internet; each sentence is sent to Microsoft. `openai` or `elevenlabs`: paid, with a key, the most natural delivery. `kokoro`: fully local and private, noticeably more synthetic. |
| **Language** (web *Settings*) | English, Hindi, or **Hindi + English (Hinglish)**: Hindi grammar with the technical words kept in English, read by Indian voices |
| **Format** | One voice, or **two hosts**: a woman asks, a man answers, as a real conversation |
| **Length** | Set by the news, not by a number: a quiet day is a minute or two, a big one up to 15 minutes. Only stories with real impact are told |

Voices are chosen per engine and language (see `.env.example`); a voice name that belongs to
another engine is ignored rather than breaking the run. In the web app, click any sentence
(or word) of the transcript to start the audio from there.

## Choose the models

`VB_LLM_SCRIPT_MODEL` writes what you hear and `VB_LLM_UTILITY_MODEL` finds, ranks and
summarises the stories. Defaults are Sonnet for the script and Haiku for the rest. To measure
other combinations on your own stories without sending or storing anything:

```bash
python scripts/compare_models.py          # Haiku+Sonnet, Sonnet+Sonnet, Haiku+Haiku
```

It writes a side-by-side report (cost, length, share of claims traced to a source, and the
scripts themselves) to `eval/runs/`. Spend is capped by `VB_DAILY_BUDGET_INR`, checked before
every model call.

## What leaves your machine

| Goes to | What | Avoid with |
|---|---|---|
| Anthropic | Story text and your stack profile, to rank and write | `VB_LLM_PROVIDER=ollama` (local, lower quality) |
| Microsoft (Edge voices) | Each sentence of the script | `VB_TTS_ENGINE=kokoro` |
| OpenAI / ElevenLabs | Script sentences, only if you choose them | leave them unset |
| GitHub | Your login and token, to read repositories and releases | leave `--github` unset |
| Slack / email | The finished brief, only if configured | leave them unset |

Everything else (the database, audio, history, your thumbs) stays in Docker volumes and a
per-user folder on your computer.

## Slack

1. Create a Slack app with bot scopes `chat:write`, `files:write`, `channels:read`,
   `channels:history`; enable **Socket Mode** and subscribe to the `message.channels` bot event.
2. Install it, invite the bot to a channel, and set `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID`
   and `SLACK_APP_TOKEN` (the `xapp-` token, scope `connections:write`) in `.env`.
3. `python -m voicebrief.cli deliver test`

Each brief arrives as a digest with the audio in its thread. **Reply in the thread** to ask
about that brief, or post in the channel to ask about the latest one. Socket Mode means no
public URL is needed; the listener runs inside the web app's process. A failed run posts a
short "could not make today's brief" message, so silence never means confusion.

## Usage

| Command | What it does |
|---|---|
| `doctor` | Checks the machine is ready and says what to run if not |
| `daily` | The whole morning loop (what the scheduler runs) |
| `serve [--host 0.0.0.0]` | Web app + API on one port; also answers in Slack |
| `docs podcast <files/folders>` | Your own material as an episode |
| `history show [--date]` · `history export` | What happened and why; JSONL for analysis |
| `deliver status / test / episode <id>` | Slack and email |
| `backup create / list / restore` | Database dump + audio copy |
| `personal show [--refresh]` | What ProjectMind told VoiceBrief |
| `ingest run [-s slug]` · `enrich` · `brief generate` | Individual steps |
| `sources sync / list` · `tts setup / test` | Source registry, voice model |

All commands: `python -m voicebrief.cli <command>` (or `voicebrief <command>` where Windows allows the launcher).

## Configuration

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | | Required |
| `GITHUB_TOKEN` | | Recommended |
| `VB_DAILY_BUDGET_INR` | `50` | Hard stop, checked before every model call |
| `VB_USD_TO_INR` | `100` | Rate for the rupee cost ledger |
| `VB_LLM_SCRIPT_MODEL` / `VB_LLM_UTILITY_MODEL` | `claude-sonnet-5-5` / `claude-haiku-5-5` | Script / ranking and summaries |
| `VB_TTS_ENGINE` | `edge` | `edge`, `openai`, `elevenlabs`, `kokoro` |
| `OPENAI_API_KEY`, `ELEVENLABS_API_KEY` | | Only for those voice engines |
| `VB_TIMEZONE` | `Asia/Kolkata` | What "today" means |
| `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID`, `SLACK_APP_TOKEN` | | Delivery and Q&A |
| `VB_SMTP_*`, `VB_MAIL_FROM`, `VB_EMAIL_TO` | | Optional email delivery |
| `VB_API_TOKEN`, `VB_PUBLIC_URL` | | Needed to serve beyond localhost |
| `VB_PROJECTMIND_CMD` | | Enables ProjectMind |
| `VB_BACKUP_DIR`, `VB_BACKUP_KEEP` | per-user folder, `7` | |
| `VB_HISTORY_RETENTION_DAYS` | `90` | How long prompt text is kept |

`VB_LLM_PROVIDER=echo` and `VB_TTS_ENGINE=null` run everything offline for tests;
`ollama` runs a local model for ₹0. Full list: [.env.example](.env.example).

<details>
<summary><b>More: phone access, history, backups, ProjectMind, adding a source</b></summary>

### On your phone
Set `VB_API_TOKEN` (a long random string) and `VB_PUBLIC_URL=http://<laptop-ip>:8000`, run
`serve --host 0.0.0.0` (or `run-serve.ps1 -Address 0.0.0.0`), open the URL and enter the
token once. `serve` refuses to listen on the network without a token, because anyone who
reached it could spend money. Away from home, use a private network such as Tailscale.

### What "ready" means
An episode is `ready` only with stories, stored audio that actually has sound, and at least
half the stories asked for; otherwise `degraded` (usable, something missing, reason stored)
or `failed`. A crash leaves `interrupted`, closed out on the next run.

### History
Every run, every candidate considered (each ranking signal's contribution, the model's
reason, picked or skipped and why), every model call (prompt, response, tokens, cost) and
every question asked on the web or in Slack is stored. Prompt text is kept 90 days; costs
and metadata forever. The *History* page and `history show` replay any day.

### Backups
After every generated brief, the database is dumped (verified with `pg_restore --list`)
and new audio copied to `%LOCALAPPDATA%\VoiceBrief\backups`, outside Docker's disk image,
keeping the newest seven. This exists because a hard freeze once corrupted that image.
`backup restore` puts the newest dump back.

### ProjectMind
[ProjectMind](https://github.com/ParthBiyani/ProjectMind) is the author's cross-project
memory. When `VB_PROJECTMIND_CMD` is set, VoiceBrief asks it about your technologies, uses
the answers in ranking and lets the writer say "this is the tradeoff you made in another
project". Only its public CLI is used; its raw private notes are never read.

### Adding a source
Add an entry to `config/sources.yaml` and run `sources sync` (idempotent; removed entries are
disabled, not deleted). `cluster: solo` marks a trending-list source whose entries are each
their own story.

</details>

## Development

```powershell
make test         # hermetic unit tests (echo LLM, silent TTS)
make test-all     # + integration tests against the Docker services
make lint; make typecheck; make eval; make ablation
cd web && npm test      # transcript timing
```

**900+ tests**; CI runs lint, mypy, the web build and the backend suite against real Postgres
and Qdrant on Python 3.11 and 3.14 for every push.

```
src/voicebrief/
├── sources/          one adapter per upstream
├── pipeline/         ingest, filter, embed, dedup, cluster, rank, summarize, ground, daily, episodes
├── graphs/           the writing graphs (brief, documents)
├── personalization/  GitHub profile, ProjectMind adapter, feedback affinity
├── delivery/         digest, Slack (send + listen), email
├── llm/  tts/  documents/  api/  evalkit/
├── qa.py             answering questions about a segment or a whole episode
└── history.py        the audit trail
```

## Limitations

- **Ranking is below its offline target** (P@10 0.600 vs 0.70 for the full path). The
  labels are synthetic (3 personas); real thumbs are the route to a trustworthy number.
- **Verification is lexical, not entailment:** it catches invented figures, versions and
  names, not a sentence that reuses the source's words with the meaning flipped.
- **Some articles can't be fetched:** full text comes from the linked page (robots.txt
  honoured), so paywalls and sites that refuse bots stay at their one-line summary.
- **Thumbs are a gentle nudge** (at most 1.5 points vs 3 for a dependency match), not a learned model.
- **It runs on your own computer**, which must be on and signed in; Docker must be running. There is no hosted version.
- **Word-level click-to-play is approximate.** Sentence times are measured from the audio; the words inside a sentence are timed by length.
- **The default voices need internet** and send each sentence to Microsoft; `VB_TTS_ENGINE=kokoro` keeps speech on your machine.
- **Documents mode** can't read scanned PDFs (no OCR).

## License

[MIT](LICENSE) © Parth Biyani
