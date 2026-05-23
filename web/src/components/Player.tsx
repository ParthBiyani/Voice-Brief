/**
 * Audio player with a transcript that follows along.
 *
 * The two-way binding is the whole point of the interactive layer in the PRD: the
 * transcript highlights whatever is playing, and clicking a line seeks the audio to
 * it. Both directions rely on the segment timestamps being measured from the
 * rendered audio rather than estimated, which is why assembly does it that way.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { EpisodeDetail, Segment } from "../lib/api";
import { api, formatTimestamp } from "../lib/api";
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
          <span className={`badge badge-${episode.status}`}>{episode.status}</span>
        </div>
      </header>

      {episode.audio_url ? (
        <audio
          ref={audioRef}
          src={episode.audio_url}
          controls
          preload="metadata"
          onTimeUpdate={(e) => setPosition(e.currentTarget.currentTime)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />
      ) : (
        <p className="notice">
          No audio for this episode. The transcript below is still complete.
        </p>
      )}

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
      </div>

      <p className="script">{segment.script}</p>

      {segment.citations.length > 0 && (
        <ul className="citations">
          {segment.citations.map((citation) => (
            <li key={citation.url}>
              <a href={citation.url} target="_blank" rel="noreferrer noopener">
                {citation.title || citation.url}
              </a>
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
