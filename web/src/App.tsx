/**
 * VoiceBrief: today's brief, the history behind it, and settings.
 *
 * Three views behind a hash route (#/, #/history, #/settings) so a phone's back button
 * and a reload both behave; no router library is needed for that.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { History } from "./components/History";
import { Player } from "./components/Player";
import { Search } from "./components/Search";
import { Settings } from "./components/Settings";
import type { EpisodeDetail, EpisodeSummary, ProgressEvent } from "./lib/api";
import {
  AuthRequired,
  adoptTokenFromUrl,
  api,
  dayLabel,
  formatDuration,
  saveToken,
  watchProgress,
} from "./lib/api";

type View = "today" | "history" | "settings";

function viewFromHash(): View {
  const hash = location.hash.replace(/^#\/?/, "");
  return hash === "history" || hash === "settings" ? hash : "today";
}

const STAGE_TEXT: Record<string, string> = {
  starting: "Starting",
  started: "Starting",
  writing: "Reading the news, picking stories and writing the script. This takes a few minutes",
};

function TokenPrompt({ onDone }: { onDone: () => void }) {
  const [value, setValue] = useState("");
  return (
    <form
      className="token-prompt"
      onSubmit={(e) => {
        e.preventDefault();
        if (!value.trim()) return;
        saveToken(value.trim());
        onDone();
      }}
    >
      <h2>Access token needed</h2>
      <p className="notice-text">
        This VoiceBrief is protected. Enter the token set as VB_API_TOKEN on the server. It is
        remembered on this device.
      </p>
      <input
        type="password"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Access token"
        autoFocus
      />
      <button className="primary" type="submit">
        Continue
      </button>
    </form>
  );
}

export default function App() {
  const [view, setView] = useState<View>(viewFromHash());
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [selected, setSelected] = useState<EpisodeDetail | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [needsToken, setNeedsToken] = useState(false);
  const stopWatching = useRef<(() => void) | null>(null);
  const openedFirst = useRef(false);

  const handle = useCallback((err: unknown, fallback: string) => {
    if (err instanceof AuthRequired) setNeedsToken(true);
    else setError(err instanceof Error ? err.message : fallback);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const rows = await api.listEpisodes();
      setEpisodes(rows);
      setError(null);
      return rows;
    } catch (err) {
      handle(err, "Could not reach the server");
      return [];
    } finally {
      setLoading(false);
    }
  }, [handle]);

  const open = useCallback(
    async (id: string, seekTo?: number) => {
      try {
        const detail = await api.getEpisode(id);
        setSelected(detail);
        setView("today");
        if (seekTo !== undefined) {
          // Let the audio element mount before seeking into it.
          requestAnimationFrame(() => {
            const audio = document.querySelector<HTMLAudioElement>(".player audio");
            if (audio) audio.currentTime = seekTo;
          });
        }
      } catch (err) {
        handle(err, "Could not load the episode");
      }
    },
    [handle],
  );

  const follow = useCallback(
    (streamUrl: string) => {
      stopWatching.current?.();
      stopWatching.current = watchProgress(streamUrl, (event) => {
        setProgress(event);
        if (event.stage === "ready") {
          void refresh();
          if (typeof event.episode_id === "string") void open(event.episode_id);
        }
      });
    },
    [open, refresh],
  );

  // First load: the list, the newest listenable episode, and any run already in flight
  // (a page opened or reloaded mid-generation picks the stream back up).
  useEffect(() => {
    adoptTokenFromUrl();
    void (async () => {
      const rows = await refresh();
      if (!openedFirst.current) {
        openedFirst.current = true;
        const listenable = rows.find((r) => r.status === "ready" || r.status === "degraded");
        if (listenable) await open(listenable.id);
      }
      try {
        const [active] = await api.activeGenerations();
        if (active) follow(`/episodes/${active}/stream`);
      } catch (err) {
        handle(err, "");
      }
    })();
    return () => stopWatching.current?.();
  }, [refresh, open, follow, handle]);

  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  function go(next: View) {
    location.hash = next === "today" ? "/" : `/${next}`;
    setView(next);
  }

  async function generate() {
    setProgress({ stage: "starting", at: new Date().toISOString() });
    try {
      const { stream_url } = await api.generate();
      follow(stream_url);
    } catch (err) {
      if (err instanceof AuthRequired) setNeedsToken(true);
      setProgress({
        stage: "failed",
        at: new Date().toISOString(),
        error: err instanceof Error ? err.message : "failed",
      });
    }
  }

  const generating = progress !== null && progress.stage !== "ready" && progress.stage !== "failed";

  if (needsToken) {
    return (
      <div className="app">
        <TokenPrompt
          onDone={() => {
            setNeedsToken(false);
            location.reload();
          }}
        />
      </div>
    );
  }

  // Group the list by day so "today's brief" is obvious at a glance.
  const groups: { label: string; items: EpisodeSummary[] }[] = [];
  for (const episode of episodes) {
    const label = dayLabel(episode.created_at);
    const last = groups[groups.length - 1];
    if (last && last.label === label) last.items.push(episode);
    else groups.push({ label, items: [episode] });
  }

  return (
    <div className="app">
      <header className="app-head">
        <div>
          <h1>VoiceBrief</h1>
          <p className="tagline">A daily audio brief that knows what you're building.</p>
        </div>
        <button className="primary" onClick={generate} disabled={generating}>
          {generating ? "Generating…" : "Make a brief now"}
        </button>
      </header>

      <nav className="tabs" aria-label="Sections">
        {(["today", "history", "settings"] as View[]).map((name) => (
          <button
            key={name}
            className={view === name ? "is-on" : ""}
            onClick={() => go(name)}
            aria-current={view === name ? "page" : undefined}
          >
            {name === "today" ? "Brief" : name[0].toUpperCase() + name.slice(1)}
          </button>
        ))}
      </nav>

      {error && <p className="banner error">{error}</p>}

      {progress && progress.stage !== "ready" && (
        <p className={`banner ${progress.stage === "failed" ? "error" : "notice"}`}>
          {progress.stage === "failed"
            ? `Could not make a brief: ${String(progress.error ?? "unknown error")}`
            : `${STAGE_TEXT[progress.stage] ?? progress.stage}…`}
        </p>
      )}

      {view === "history" && <History />}
      {view === "settings" && <Settings />}

      {view === "today" && (
        <>
          <Search onOpen={(episodeId, seconds) => void open(episodeId, seconds)} />

          <div className="layout">
            <aside className="episodes">
              <h2>Episodes</h2>
              {loading && <p className="notice">Loading…</p>}
              {!loading && episodes.length === 0 && (
                <p className="notice">
                  No briefs yet. Press “Make a brief now”, or run <code>voicebrief daily</code>.
                </p>
              )}
              {groups.map((group) => (
                <div key={group.label} className="episode-group">
                  <h3>{group.label}</h3>
                  <ul>
                    {group.items.map((episode) => (
                      <li key={episode.id}>
                        <button
                          className={selected?.id === episode.id ? "is-selected" : ""}
                          onClick={() => void open(episode.id)}
                        >
                          <span className="episode-title">{episode.title ?? "Untitled"}</span>
                          <span className="episode-meta">
                            {new Date(episode.created_at).toLocaleTimeString([], {
                              hour: "2-digit",
                              minute: "2-digit",
                            })}{" "}
                            · {formatDuration(episode.duration_seconds)}
                            {episode.status !== "ready" && (
                              <span className={`badge badge-${episode.status}`}>
                                {episode.status}
                              </span>
                            )}
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </aside>

            <main>
              {selected ? (
                <Player episode={selected} />
              ) : (
                <p className="notice">Select an episode.</p>
              )}
            </main>
          </div>
        </>
      )}
    </div>
  );
}
