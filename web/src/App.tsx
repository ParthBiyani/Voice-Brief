/**
 * Episode list, player, and search.
 *
 * Deliberately one screen. The PRD's v1 interactive layer is transcript + scoped
 * Q&A + feedback; a router and multiple pages would be scaffolding around a product
 * that has exactly one thing to show.
 */

import { useCallback, useEffect, useState } from "react";
import { Player } from "./components/Player";
import { Search } from "./components/Search";
import type { EpisodeDetail, EpisodeSummary, ProgressEvent } from "./lib/api";
import { api, formatDuration, watchProgress } from "./lib/api";

export default function App() {
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [selected, setSelected] = useState<EpisodeDetail | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const rows = await api.listEpisodes();
      setEpisodes(rows);
      setError(null);
      return rows;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not reach the API");
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh().then((rows) => {
      const ready = rows.find((r) => r.status === "ready");
      if (ready) void open(ready.id);
    });
    // `refresh` is stable; opening the first episode is a mount-time concern only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refresh]);

  async function open(id: string, seekTo?: number) {
    try {
      const detail = await api.getEpisode(id);
      setSelected(detail);
      if (seekTo !== undefined) {
        // Let the audio element mount before seeking into it.
        requestAnimationFrame(() => {
          const audio = document.querySelector<HTMLAudioElement>(".player audio");
          if (audio) audio.currentTime = seekTo;
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load episode");
    }
  }

  async function generate() {
    setProgress({ stage: "starting", at: new Date().toISOString() });
    try {
      const { stream_url } = await api.generate({
        max_stories: 6,
        target_minutes: 10,
        topics: ["agentic-ai", "tooling"],
      });
      const stop = watchProgress(stream_url, (event) => {
        setProgress(event);
        if (event.stage === "ready") {
          void refresh();
          if (typeof event.episode_id === "string") void open(event.episode_id);
        }
      });
      // Stop listening if the user leaves before it finishes.
      window.addEventListener("beforeunload", stop, { once: true });
    } catch (err) {
      setProgress({
        stage: "failed",
        at: new Date().toISOString(),
        error: err instanceof Error ? err.message : "failed",
      });
    }
  }

  return (
    <div className="app">
      <header className="app-head">
        <div>
          <h1>VoiceBrief</h1>
          <p className="tagline">A daily audio brief that knows what you're building.</p>
        </div>
        <button className="primary" onClick={generate} disabled={progress?.stage === "writing"}>
          Generate today's brief
        </button>
      </header>

      {error && <p className="error banner">{error}</p>}

      {progress && progress.stage !== "ready" && (
        <p className={`banner ${progress.stage === "failed" ? "error" : "notice"}`}>
          {progress.stage === "failed"
            ? `Generation failed: ${String(progress.error ?? "unknown")}`
            : `Generating — ${progress.stage}…`}
        </p>
      )}

      <Search onOpen={(episodeId, seconds) => void open(episodeId, seconds)} />

      <div className="layout">
        <aside className="episodes">
          <h2>Episodes</h2>
          {loading && <p className="notice">Loading…</p>}
          {!loading && episodes.length === 0 && (
            <p className="notice">
              No episodes yet. Run <code>make ingest</code>, then <code>make enrich</code>,
              then generate.
            </p>
          )}
          <ul>
            {episodes.map((episode) => (
              <li key={episode.id}>
                <button
                  className={selected?.id === episode.id ? "is-selected" : ""}
                  onClick={() => void open(episode.id)}
                >
                  <span className="episode-title">{episode.title ?? "Untitled"}</span>
                  <span className="episode-meta">
                    {new Date(episode.created_at).toLocaleDateString()} ·{" "}
                    {formatDuration(episode.duration_seconds)}
                    {episode.status !== "ready" && ` · ${episode.status}`}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <main>
          {selected ? (
            <Player episode={selected} />
          ) : (
            <p className="notice">Select an episode.</p>
          )}
        </main>
      </div>
    </div>
  );
}
