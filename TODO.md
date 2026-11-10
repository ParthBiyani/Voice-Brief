# VoiceBrief TODO

Written 2026-11-10 after a full audit (read the PRD, ran the stack, ran the daily brief through
the real scheduled task). Items are in priority order. Unfinished items are not deleted; new
work is added at the bottom of the right section.

## Audit snapshot (2026-11-10)

- 857 tests pass, mypy and ruff are clean, the web API is healthy.
- Scheduled run through Task Scheduler worked end to end: 1,495 items ingested, 9.3 min episode in
  264 s, delivered to Slack with audio, verified backup, done flag written.
- Misses against the PRD targets: attribution 0.820 (target 0.95; 2 Nov was 0.854), cost ₹8.43
  (target under ₹8; 2 Nov was ₹8.24), ranking precision@10 0.600 (target 0.70).
- 2 Nov: the episode was generated, then Postgres vanished, so delivery crashed and the backup
  failed. The retry then failed because Qdrant exited (143), so that day may never have reached Slack.
- 10 Nov: Docker Desktop's engine crashed right after the run (WSL error
  `Wsl/CallMsi/Install/ERROR_NOT_FOUND`); a restart of Docker Desktop fixed it.
- Not yet tested: the web UI by eye, Slack Q&A, Mode 2 documents, Hindi.

## Reliability (do first)

- [x] **1. Retry delivery and backup on their own.** A rerun that finds today's episode returns
  "already done" and never re-delivers or re-backs-up (`pipeline/daily.py`, `already_done` path;
  `scripts/run-daily.ps1` writes the done flag on exit 0). Why: a generated brief that never
  reaches Slack is a silent failure, which goal G1 forbids.
- [x] **2. Alarm on runs that start but never finish, and on days with no brief.** Task
  Scheduler's result code is always 0 under headless conhost. The tasks were disabled for a week
  without anyone noticing. Why: nothing currently announces a missed day.
- [x] **3. Survive Docker Desktop crashes.** Detect the engine dying mid-run, wait for it to come
  back, and resume or retry. Consider a Docker Desktop update or `wsl --update` if the WSL error
  recurs. Why: Docker failed on 2 Nov and 10 Nov, both right after generation.
- [x] **4. Work offline and tolerate flaky networks.** The embedding model calls Hugging Face on
  every start (set offline mode), and a network drop on 2 Nov produced DNS errors across dozens of
  sources. Add retries on transient errors. Why: a bad network should not kill the web app or the
  09:00 brief.

## Quality

- [x] **5. Find out why attribution dropped to 0.82.** The flagged sentences look like spoken
  numbers ("sixteen six nine seven") not matching digits ("16697"), so the verifier may be giving
  false positives. Fix the verifier or the writer, whichever is wrong. Why: the PRD's 0.966 was one
  day, and the metric cannot be trusted until this is understood.
- [x] **6. Stop reading PR numbers and version strings aloud; cap GitHub release items per
  brief.** Today's brief was mostly release notes. Why: poor listening, and it dilutes "what
  matters to my code".
- [x] **7. Bring cost back under ₹8.** Trim the script prompt or input, or run the blind listening
  test of Haiku for the script (roadmap item). Why: the target was missed on the last two days.
- [ ] **8. Collect real thumbs and rerun the ranking evaluation** (`history export` joined with
  votes). Why: precision@10 is 0.600 against 0.70, and only synthetic labels exist; more tuning
  would overfit.

## Security and ops hygiene

- [x] **9. Bind datastores to localhost and replace default passwords.** `docker-compose.yml`
  publishes Postgres, Redis, Qdrant and MinIO on all interfaces; passwords are "voicebrief" and
  Redis and Qdrant have none. Why: this is a laptop that joins public Wi-Fi.
- [ ] **10. Fill in settings (needs the owner).** Topics are empty and the email is the placeholder
  `me@example.com`. The GitHub stack profile does refresh on its own (rebuilt 10 Nov). Topics are a personal
  choice; the email is the internal user key, so changing it would orphan history and feedback.
  Why: declared interests feed the ranker.
- [ ] **11. Replace the `bitnamilegacy/minio` image.** Redis is dropped (it was unused). Tried
  `cgr.dev/chainguard/minio`: it reads the existing data and works as root, but fails with
  "file access denied" as its non-root user even after chown. Needs a proper migration (fresh
  volume, then `voicebrief backup restore` of the media copy). Why: the legacy image is unmaintained.

## Testing and later

- [ ] **12. Test what the audit did not:** web UI by eye on desktop and phone, a real Slack Q&A
  answer, Mode 2 documents, the Hindi path.
- [ ] **13. Add frontend tests.** CI only builds the web app. Why: the UI has no regression
  protection.
- [ ] **14. Move to a small cloud VM (roadmap).** Why: everything depends on this laptop being on
  and signed in.

## Added later

- [x] **Two real voices and a real conversation (owner request, 10 Nov).** Two-host briefs had one
  voice reading the HOST labels aloud. Now A (woman) asks and B (man) answers, each in their own
  voice, and speech is paced sentence by sentence. Needs a listen by the owner: voice and pacing
  settings are `VB_TTS_VOICE_HOST_*` and `VB_TTS_SPEED`.

_(new items go here)_
