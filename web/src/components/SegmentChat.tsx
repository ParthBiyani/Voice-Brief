/**
 * Scoped Q&A for one segment.
 *
 * The preset buttons are the PRD's one-tap prompts. Answers are grounded in that
 * segment's sources only, so the citations shown are the ones the answer can draw
 * from — the listener can check every claim without leaving the page.
 */

import { useState } from "react";
import type { Citation } from "../lib/api";
import { api } from "../lib/api";

const PRESETS: { key: string; label: string }[] = [
  { key: "explain_simply", label: "Explain simply" },
  { key: "show_code", label: "Show me the code" },
  { key: "compare_with", label: "Compare with alternatives" },
];

export function SegmentChat({ segmentId }: { segmentId: string }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function ask(text: string, preset?: string) {
    if (!text && !preset) return;
    setBusy(true);
    setError(null);
    setAnswer(null);
    try {
      const result = await api.askSegment(segmentId, text, preset);
      setAnswer(result.answer);
      setCitations(result.citations);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="chat">
      <div className="chat-presets">
        {PRESETS.map((preset) => (
          <button key={preset.key} disabled={busy} onClick={() => ask("", preset.key)}>
            {preset.label}
          </button>
        ))}
      </div>

      <form
        className="chat-form"
        onSubmit={(event) => {
          event.preventDefault();
          void ask(question);
        }}
      >
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask about this segment…"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !question.trim()}>
          {busy ? "Thinking…" : "Ask"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}
      {answer && (
        <div className="chat-answer">
          <p>{answer}</p>
          {citations.length > 0 && (
            <p className="chat-sources">
              Answered from:{" "}
              {citations.map((citation, index) => (
                <span key={citation.url}>
                  {index > 0 && ", "}
                  <a href={citation.url} target="_blank" rel="noreferrer noopener">
                    {citation.title || "source"}
                  </a>
                </span>
              ))}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
