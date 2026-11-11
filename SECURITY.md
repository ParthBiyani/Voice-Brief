# Security

VoiceBrief runs on your own computer and spends your API credits, so the things that matter
are keys and network exposure.

- Secrets live in `.env`, which is git-ignored. Never commit it.
- The database, vector store and audio storage are published on `127.0.0.1` only.
  Change the default passwords (`VB_PG_PASSWORD`, `VB_S3_SECRET_KEY`) if other people share your machine.
- `voicebrief serve --host 0.0.0.0` refuses to start without `VB_API_TOKEN`, because anyone
  who could reach it could generate briefs and spend money. Away from home, use a private
  network such as Tailscale rather than exposing the port.
- The default voice engine sends each sentence of the script to Microsoft; see
  [What leaves your machine](README.md#what-leaves-your-machine).

To report a vulnerability, open a private security advisory on GitHub
(*Security → Report a vulnerability*) rather than a public issue.
