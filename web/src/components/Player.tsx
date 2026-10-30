/**
 * Audio player with a transcript that follows along.
 *
 * The two-way binding is the whole point of the interactive layer in the PRD: the
 * transcript highlights whatever is playing, and clicking a line seeks the audio to
 * it. Both directions rely on the segment timestamps being measured from the
 * rendered audio rather than estimated, which is why assembly does it that way.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { EpisodeDetail, RankingRow, Segment } from "../lib/api";
import { api, formatTimestamp, mediaUrl, percent } from "../lib/api";
import { SegmentChat } from "./SegmentChat";

interface Props {
  episode: EpisodeDetail;
}

export function Player({ episode }: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [openChat, setOpenChat] = useState<string | null>(null);
  const [votes, setVotes] = useState<Record<string, 1 | -1>>({});
  const [why, setWhy] = useState<RankingRow[] | null>(null);
  const [showWhy, setShowWhy] = useState(false);

  // A different episode starts fresh: no stale explanation from the previous one.
  useEffect(() => {
    setWhy(null);
    setShowWhy(false);
  }, [episode.id]);

  async function toggleWhy() {
    if (!showWhy && why === null) {
      try {
        setWhy(await api.ranking(episode.id));
      } catch {
        setWhy([]);
      }
    }
    setShowWhy((open) => !open);
  }

  const activeId = useMemo(() => {
    const current = episode.segments.find(
      (s) =>
        s.start_seconds !== null &&
        s.end_seconds !== null &&
        position >= s.start_seconds &&
        position < s.end_seconds,
    );
    return current?.id ?? null;
  }, [episode.segments, position]);

  // Keep the active transcript line in view without yanking the page while the
  // user is deliberately scrolling elsewhere.
  useEffect(() => {
    if (!activeId || !playing) return;
    document
      .getElementById(`segment-${activeId}`)
      ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [activeId, playing]);

  function seek(seconds: number) {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = seconds;
    void audio.play().catch(() => undefined);
  }

  async function vote(segmentId: string, value: 1 | -1) {
    setVotes((prev) => ({ ...prev, [segmentId]: value }));
    try {
      await api.vote(segmentId, value);
    } catch {
      setVotes((prev) => {
        const next = { ...prev };
        delete next[segmentId];
        return next;
      });
    }
  }

  return (
    <div className="player">
      <header className="player-head">
        <h2>{episode.title ?? "Untitled episode"}</h2>
        <div className="meta">
          <span>{formatTimestamp(episode.duration_seconds ?? 0)}</span>
          {episode.cost_inr !== null && <span>₹{episode.cost_inr.toFixed(2)}</span>}
          {episode.grounding?.attribution_rate !== undefined && (
            <span title="Share of factual claims traced to a source passage">
              {percent(episode.grounding.attribution_rate)} sourced
            </span>
          )}
          <span className={`badge badge-${episode.status}`}>{episode.status}</span>
        </div>
      </header>

      {episode.status === "degraded" && episode.error && (
        <p className="banner error">This brief is incomplete: {episode.error}</p>
      )}
      {episode.status === "failed" && (
        <p className="banner error">This brief failed: {episode.error ?? "unknown error"}</p>
      )}

      {episode.audio_url ? (
        <div className="audio-bar">
          <audio
            ref={audioRef}
            src={mediaUrl(episode.audio_url)}
            controls
            preload="metadata"
            onTimeUpdate={(e) => setPosition(e.currentTarget.currentTime)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
          />
          <a className="download" href={`${mediaUrl(episode.audio_url)}?download=1`} download>
            Download
          </a>
        </div>
      ) : (
        <p className="notice">
          No audio for this episode. The transcript below is still complete.
        </p>
      )}

      <div className="why-bar">
        <button onClick={() => void toggleWhy()} aria-expanded={showWhy}>
          {showWhy ? "Hide" : "Why these stories?"}
        </button>
      </div>
      {showWhy && <Why rows={why} />}

      <ol className="transcript">
        {episode.segments.map((segment) => (
          <TranscriptLine
            key={segment.id}
            segment={segment}
            active={segment.id === activeId}
            vote={votes[segment.id]}
            onSeek={seek}
            onVote={vote}
            chatOpen={openChat === segment.id}
            onToggleChat={() =>
              setOpenChat((current) => (current === segment.id ? null : segment.id))
            }
          />
        ))}
      </ol>
    </div>
  );
}

interface LineProps {
  segment: Segment;
  active: boolean;
  vote: 1 | -1 | undefined;
  onSeek: (seconds: number) => void;
  onVote: (segmentId: string, value: 1 | -1) => void;
  chatOpen: boolean;
  onToggleChat: () => void;
}

function TranscriptLine({
  segment,
  active,
  vote,
  onSeek,
  onVote,
  chatOpen,
  onToggleChat,
}: LineProps) {
  const start = segment.start_seconds ?? 0;
  const isStory = segment.kind === "story";

  return (
    <li
      id={`segment-${segment.id}`}
      className={`segment segment-${segment.kind}${active ? " is-active" : ""}`}
    >
      <div className="segment-head">
        <button className="timestamp" onClick={() => onSeek(start)}>
          {formatTimestamp(start)}
        </button>
        {segment.heading && <h3>{segment.heading}</h3>}
        {segment.personal && (
          <span className="chip chip-personal">
            {segment.matched.length > 0 ? `your stack: ${segment.matched.join(", ")}` : "for you"}
          </span>
        )}
      </div>

      <p className="script">{segment.script}</p>

      {segment.citations.length > 0 && (
        <ul className="citations">
          {segment.citations.map((citation) => (
            <li key={citation.url}>
              {citation.url.startsWith("doc://") ? (
                // Your own document: a section to look up, not a web address to open.
                <span className="doc-source" title="From your documents">
                  {citation.title}
                </span>
              ) : (
                <a href={citation.url} target="_blank" rel="noreferrer noopener">
                  {citation.title || citation.url}
                </a>
              )}
            </li>
          ))}
        </ul>
      )}

      {isStory && (
        <div className="segment-actions">
          <button
            className={vote === 1 ? "is-on" : ""}
            onClick={() => onVote(segment.id, 1)}
            aria-label="More like this"
          >
            More like this
          </button>
          <button
            className={vote === -1 ? "is-on" : ""}
            onClick={() => onVote(segment.id, -1)}
            aria-label="Less like this"
          >
            Less like this
          </button>
          <button onClick={onToggleChat}>{chatOpen ? "Close" : "Ask about this"}</button>
        </div>
      )}

      {chatOpen && <SegmentChat segmentId={segment.id} />}
    </li>
  );
}

function signalText(signals: Record<string, number>): string {
  return Object.entries(signals)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3)
    .map(([name, value]) => `${name.replace("_", " ")} ${value > 0 ? "+" : ""}${value.toFixed(1)}`)
    .join(" · ");
}

function Why({ rows }: { rows: RankingRow[] | null }) {
  if (rows === null) return <p className="notice">Loading…</p>;
  if (rows.length === 0) {
    return <p className="notice">No ranking record for this episode (it predates history).</p>;
  }
  const shown = rows.filter((r) => r.outcome !== "aired").slice(0, 14);
  return (
    <div className="why">
      <p className="notice-text">
        The top candidates VoiceBrief considered, with what each signal contributed.
      </p>
      <ul className="plain">
        {shown.map((row, index) => (
          <li key={`${row.title}-${index}`} className={`row-${row.outcome}`}>
            <span className="why-mark">{row.outcome === "picked" ? "●" : "○"}</span>
            <div>
              <div>{row.title}</div>
              <div className="notice-text">
                {signalText(row.signals)}
                {row.stack_match.length > 0 && ` · stack: ${row.stack_match.join(", ")}`}
              </div>
              {row.reason && <div className="notice-text">{row.reason}</div>}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
