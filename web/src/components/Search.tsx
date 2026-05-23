/**
 * Semantic search across every past episode.
 *
 * Answers the PRD's third job-to-be-done: "which paper about long-term memory did
 * you mention last week?" Each hit carries the episode and timestamp, so the result
 * is a place to jump to rather than just a reminder.
 */

import { useState } from "react";
import type { SearchHit } from "../lib/api";
import { api } from "../lib/api";

interface Props {
  onOpen: (episodeId: string, startSeconds: number) => void;
}

export function Search({ onOpen }: Props) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [busy, setBusy] = useState(false);

  async function run(event: React.FormEvent) {
    event.preventDefault();
    if (query.trim().length < 2) return;
    setBusy(true);
    try {
      const result = await api.search(query.trim());
      setHits(result.hits);
    } catch {
      setHits([]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="search">
      <form onSubmit={run}>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search past episodes — e.g. durable execution"
        />
        <button type="submit" disabled={busy}>
          {busy ? "Searching…" : "Search"}
        </button>
      </form>

      {hits !== null && hits.length === 0 && <p className="notice">Nothing matched.</p>}

      {hits && hits.length > 0 && (
        <ul className="hits">
          {hits.map((hit) => (
            <li key={hit.segment_id}>
              <button onClick={() => onOpen(hit.episode_id, hit.start_seconds)}>
                <span className="hit-head">
                  <strong>{hit.heading ?? "Segment"}</strong>
                  <span className="hit-meta">
                    {hit.timestamp} · {hit.score.toFixed(2)}
                  </span>
                </span>
                <span className="hit-episode">{hit.episode_title}</span>
                <span className="hit-excerpt">{hit.excerpt}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
