# Contributing

Thanks for looking. VoiceBrief is a one-person project that runs on one computer, and it is
meant to be easy to run, read and change.

## Run it

Follow [Set it up on your own machine](README.md#set-it-up-on-your-own-machine), then:

```bash
make test        # hermetic unit tests: no API key, no network, no speech
make lint        # ruff check and format --check, what CI enforces
make typecheck   # mypy
cd web && npm test && npm run build
```

Unit tests use the `echo` model provider and the silent speech engine, so they run anywhere.
Integration tests need the Docker services (`make up`) and are marked `integration`.

## Changing things

- One change per pull request, with a branch named for it (`feat/…`, `fix/…`, `docs/…`).
- Commit messages are one line, conventional style, imperative: `fix(ranking): keep thin repos out`.
- Add a test with a behaviour change. A bug fix starts with a test that fails.
- Keep comments for the *why*; the code says what it does.
- A change to ranking or to the writer's prompts should come with the measurement that
  motivated it (`make eval`, or `scripts/compare_models.py` for models). See
  [eval/EXPERIMENTS.md](eval/EXPERIMENTS.md) for what has already been tried.

## Adding a source

Add an entry to `config/sources.yaml` and run `voicebrief sources sync`. A new kind of
source is one adapter in `src/voicebrief/sources/` registered with `@registry.register`.

## Adding a voice engine

Subclass `TTSEngine` in `src/voicebrief/tts/engines.py`, give it default voices in
`DEFAULT_VOICES`, and add it to `build_engine`. Set `concurrency` above 1 if it waits on a
network. Rendering, pacing and click-to-play timings are shared and need no changes.

## Ground rules

Do not commit `.env`, API keys, tokens or audio. Report security problems privately: see
[SECURITY.md](SECURITY.md).
