/**
 * Turn your own material into an episode.
 *
 * Drop PDFs, Word files, notes or a zipped repository. One document becomes a walk
 * through it; several also get a segment on where they disagree.
 */

import { useRef, useState } from "react";
import { api } from "../lib/api";

const ACCEPT = ".pdf,.docx,.md,.txt,.rst,.zip";
const MAX_FILES = 8;

function size(bytes: number): string {
  return bytes > 1_000_000 ? `${(bytes / 1_000_000).toFixed(1)} MB` : `${Math.ceil(bytes / 1000)} KB`;
}

interface Props {
  /** Called with the progress stream once the server has accepted the documents. */
  onStarted: (streamUrl: string) => void;
  busy: boolean;
}

export function Documents({ onStarted, busy }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [language, setLanguage] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  function add(list: FileList | File[]) {
    const next = [...files];
    for (const file of Array.from(list)) {
      if (!next.some((f) => f.name === file.name && f.size === file.size)) next.push(file);
    }
    if (next.length > MAX_FILES) {
      setError(`At most ${MAX_FILES} documents per episode.`);
      return;
    }
    setError(null);
    setFiles(next);
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (files.length === 0) return;
    setSending(true);
    setError(null);
    try {
      const { stream_url } = await api.uploadDocuments(files, language || undefined);
      setFiles([]);
      onStarted(stream_url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="documents">
      <h2>Documents to podcast</h2>
      <p className="notice-text">
        Papers, notes, Word files, or a zipped repository. Everything said comes from what you
        supply, and nothing is read from the web.
      </p>

      <form onSubmit={submit}>
        <div
          className={`dropzone${dragging ? " is-dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            add(e.dataTransfer.files);
          }}
          onClick={() => input.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        >
          <p>Drop files here, or tap to choose</p>
          <small>PDF · DOCX · Markdown · text · zip (a repository)</small>
          <input
            ref={input}
            type="file"
            accept={ACCEPT}
            multiple
            hidden
            onChange={(e) => e.target.files && add(e.target.files)}
          />
        </div>

        {files.length > 0 && (
          <ul className="file-list">
            {files.map((file) => (
              <li key={`${file.name}-${file.size}`}>
                <span>{file.name}</span>
                <span className="notice-text">{size(file.size)}</span>
                <button
                  type="button"
                  aria-label={`Remove ${file.name}`}
                  onClick={() => setFiles(files.filter((f) => f !== file))}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="form-actions">
          <label className="inline">
            <span>Language</span>
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="">My default</option>
              <option value="en">English</option>
              <option value="hi">Hindi</option>
            </select>
          </label>
          <button
            className="primary"
            type="submit"
            disabled={files.length === 0 || sending || busy}
          >
            {sending ? "Uploading…" : busy ? "Busy with another episode…" : "Make the episode"}
          </button>
        </div>
        {error && <p className="error-text">{error}</p>}
      </form>
    </div>
  );
}
