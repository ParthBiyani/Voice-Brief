/**
 * What happened on a day, and why.
 *
 * Runs, what was picked or skipped with each ranking signal's contribution, the model
 * calls and their cost, and every question asked and answered. The same record is what
 * the system will be improved from.
 */

import { useEffect, useState } from "react";
import type { HistoryDay } from "../lib/api";
import { api } from "../lib/api";

function today(): string {
  const now = new Date();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${month}-${day}`;
}

function time(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function signalText(signals: Record<string, number>): string {
  return Object.entries(signals)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3)
    .map(([name, value]) => `${name.replace("_", " ")} ${value > 0 ? "+" : ""}${value.toFixed(1)}`)
    .join(" · ");
}

function asked(request: Record<string, unknown>): string {
  const value = request.question ?? request.q ?? request.vote ?? JSON.stringify(request);
  return String(value);
}

export function History() {
  const [date, setDate] = useState(today());
  const [data, setData] = useState<HistoryDay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showSkipped, setShowSkipped] = useState(false);

  useEffect(() => {
    setData(null);
    setError(null);
    api
      .history(date)
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load history"));
  }, [date]);

  const empty = data && !data.runs.length && !data.episodes.length && !data.interactions.length;

  return (
    <div className="history">
      <div className="history-head">
        <h2>History</h2>
        <input
          type="date"
          value={date}
          max={today()}
          onChange={(e) => e.target.value && setDate(e.target.value)}
          aria-label="Day"
        />
      </div>

      {error && <p className="banner error">{error}</p>}
      {!data && !error && <p className="notice">Loading…</p>}
      {empty && <p className="notice">Nothing recorded for this day.</p>}

      {data && data.runs.length > 0 && (
        <section>
          <h3>Runs</h3>
          <ul className="plain">
            {data.runs.map((run) => {
              const summary = run.summary as {
                ingest?: { new_items?: number; sources_failed?: number };
                candidates?: { considered?: number; already_aired?: number };
              };
              return (
                <li key={run.id}>
                  <span className={`badge badge-${run.status}`}>{run.status}</span>{" "}
                  {run.kind} · {run.trigger} · {time(run.started_at)}
                  {summary.ingest?.new_items !== undefined &&
                    ` · ${summary.ingest.new_items} new items`}
                  {summary.ingest?.sources_failed
                    ? ` (${summary.ingest.sources_failed} sources failed)`
                    : ""}
                  {summary.candidates &&
                    ` · ${summary.candidates.considered} candidates, ${summary.candidates.already_aired} already heard`}
                  {run.error && <span className="error-text"> · {run.error}</span>}
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {data?.episodes.map((episode) => {
        const rows = episode.ranking.filter((r) => showSkipped || r.outcome === "picked");
        const spend = episode.llm_calls.reduce((sum, call) => sum + call.cost_inr, 0);
        return (
          <section key={episode.id} className="history-episode">
            <h3>{episode.title ?? "Untitled"}</h3>
            <p className="notice-text">
              <span className={`badge badge-${episode.status}`}>{episode.status}</span> ₹
              {episode.cost_inr.toFixed(2)} · {episode.llm_calls.length} model calls (₹
              {spend.toFixed(2)})
              {episode.grounding.attribution_rate !== undefined &&
                ` · ${Math.round(episode.grounding.attribution_rate * 100)}% of claims traced to a source`}
            </p>
            {episode.error && <p className="error-text">{episode.error}</p>}

            {episode.ranking.length > 0 && (
              <>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={showSkipped}
                    onChange={(e) => setShowSkipped(e.target.checked)}
                  />{" "}
                  Show stories that were considered but not picked
                </label>
                <div className="table-wrap">
                  <table className="rank">
                    <thead>
                      <tr>
                        <th />
                        <th>Story</th>
                        <th>Score</th>
                        <th>Why</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, index) => (
                        <tr key={`${row.title}-${index}`} className={`row-${row.outcome}`}>
                          <td>
                            {row.outcome === "picked"
                              ? "●"
                              : row.outcome === "aired"
                                ? "heard"
                                : "○"}
                          </td>
                          <td>
                            {row.title}
                            <div className="notice-text">{row.source}</div>
                          </td>
                          <td className="num">{row.score === null ? "" : row.score.toFixed(1)}</td>
                          <td>
                            {row.stack_match.length > 0 && (
                              <span className="chip">stack: {row.stack_match.join(", ")}</span>
                            )}
                            <div className="notice-text">{signalText(row.signals)}</div>
                            {row.why && <div className="notice-text">{row.why}</div>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </section>
        );
      })}

      {data && data.interactions.length > 0 && (
        <section>
          <h3>Questions and actions</h3>
          <ul className="plain">
            {data.interactions.map((item) => (
              <li key={item.id}>
                {time(item.at)} · <span className="chip">{item.kind}</span> {asked(item.request)}
                {typeof item.response.answer === "string" && (
                  <div className="notice-text">→ {item.response.answer.slice(0, 220)}</div>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
