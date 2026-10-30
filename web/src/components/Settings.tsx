/**
 * What VoiceBrief knows about you, and what you can change.
 *
 * Everything here used to be a command-line flag repeated on every run. Saved once, it
 * applies to every brief, scheduled or manual.
 */

import { useEffect, useState } from "react";
import type { Integration, Settings as SettingsData } from "../lib/api";
import { api } from "../lib/api";

function Status({ title, info, hint }: { title: string; info: Integration; hint: string }) {
  return (
    <div className="integration">
      <div className="integration-head">
        <strong>{title}</strong>
        <span className={`badge ${info.configured ? "badge-ready" : ""}`}>
          {info.configured ? "connected" : "not connected"}
        </span>
      </div>
      <p className="notice-text">{info.detail}</p>
      {info.fetched_at && (
        <p className="notice-text">Updated {new Date(info.fetched_at).toLocaleString()}</p>
      )}
      {!info.configured && <p className="notice-text">{hint}</p>}
    </div>
  );
}

export function Settings() {
  const [loaded, setLoaded] = useState<SettingsData | null>(null);
  const [login, setLogin] = useState("");
  const [topics, setTopics] = useState("");
  const [language, setLanguage] = useState("en");
  const [style, setStyle] = useState("solo_anchor");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  function apply(data: SettingsData) {
    setLoaded(data);
    setLogin(data.github_login ?? "");
    setTopics(data.topics.join(", "));
    setLanguage(data.language);
    setStyle(data.style);
  }

  useEffect(() => {
    api
      .settings()
      .then(apply)
      .catch((err) =>
        setMessage({ ok: false, text: err instanceof Error ? err.message : "Could not load" }),
      );
  }, []);

  async function save(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setMessage(null);
    try {
      const saved = await api.saveSettings({
        github_login: login.trim(),
        topics: topics
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
        language,
        style,
      });
      apply(saved);
      setMessage({ ok: true, text: "Saved. It applies from the next brief." });
    } catch (err) {
      setMessage({ ok: false, text: err instanceof Error ? err.message : "Could not save" });
    } finally {
      setBusy(false);
    }
  }

  if (!loaded) return <p className="notice">{message?.text ?? "Loading…"}</p>;

  return (
    <div className="settings">
      <h2>Settings</h2>

      <form onSubmit={save} className="form">
        <label>
          <span>GitHub username</span>
          <input
            value={login}
            onChange={(e) => setLogin(e.target.value)}
            placeholder="your-username"
            autoComplete="off"
          />
          <small>
            Your repositories' dependencies decide which releases count as "yours". Public and,
            with a token configured on the server, private repositories.
          </small>
        </label>

        <label>
          <span>Topics you follow</span>
          <input
            value={topics}
            onChange={(e) => setTopics(e.target.value)}
            placeholder="agentic-ai, tooling, edge-ai"
          />
          <small>Comma separated. Stories on these topics rank higher.</small>
        </label>

        <div className="form-row">
          <label>
            <span>Language</span>
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="en">English</option>
              <option value="hi">Hindi</option>
            </select>
          </label>
          <label>
            <span>Format</span>
            <select value={style} onChange={(e) => setStyle(e.target.value)}>
              <option value="solo_anchor">One voice</option>
              <option value="two_host">Two hosts</option>
            </select>
          </label>
        </div>

        <div className="form-actions">
          <button className="primary" type="submit" disabled={busy}>
            {busy ? "Saving…" : "Save"}
          </button>
          {message && (
            <span className={message.ok ? "notice-text" : "error-text"}>{message.text}</span>
          )}
        </div>
      </form>

      <h3>Connections</h3>
      <Status
        title="GitHub stack profile"
        info={loaded.github}
        hint="Add your username above."
      />
      <Status
        title="ProjectMind memory"
        info={loaded.projectmind}
        hint="Set VB_PROJECTMIND_CMD in the server's .env to let your ProjectMind memory shape the brief."
      />

      <h3>System</h3>
      <p className="notice-text">
        Writing: {loaded.llm_provider} · Speech: {loaded.tts_engine} · Daily spend cap: ₹
        {loaded.daily_budget_inr}
      </p>
    </div>
  );
}
