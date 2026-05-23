/**
 * API client.
 *
 * One place that knows the wire format, so a route change breaks the build rather
 * than a component at runtime.
 */

const BASE = "/api";

export interface Citation {
  url: string;
  title: string;
  item_id: string | null;
}

export interface Segment {
  id: string;
  position: number;
  kind: "cold_open" | "agenda" | "story" | "also_noted" | "sign_off";
  heading: string | null;
  script: string;
  start_seconds: number | null;
  end_seconds: number | null;
  citations: Citation[];
}

export interface EpisodeSummary {
  id: string;
  title: string | null;
  status: "pending" | "running" | "ready" | "failed";
  mode: string;
  language: string;
  style: string;
  duration_seconds: number | null;
  cost_inr: number | null;
  created_at: string;
}

export interface EpisodeDetail extends EpisodeSummary {
  segments: Segment[];
  audio_url: string | null;
  generation_seconds: number | null;
  error: string | null;
}

export interface SearchHit {
  segment_id: string;
  episode_id: string;
  episode_title: string | null;
  heading: string | null;
  excerpt: string;
  timestamp: string;
  start_seconds: number;
  score: number;
  citations: Citation[];
}

export interface ProgressEvent {
  stage: string;
  at: string;
  [key: string]: unknown;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}${detail ? `: ${detail}` : ""}`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; llm_provider: string; tts_engine: string }>("/health"),

  listEpisodes: (limit = 20) => request<EpisodeSummary[]>(`/episodes?limit=${limit}`),

  getEpisode: (id: string) => request<EpisodeDetail>(`/episodes/${id}`),

  search: (query: string) =>
    request<{ query: string; hits: SearchHit[] }>(`/search?q=${encodeURIComponent(query)}`),

  vote: (segmentId: string, vote: 1 | -1) =>
    request<void>("/feedback", {
      method: "POST",
      body: JSON.stringify({ segment_id: segmentId, vote }),
    }),

  askSegment: (segmentId: string, question: string, preset?: string) =>
    request<{ answer: string; citations: Citation[]; cost_inr: number }>(
      `/segments/${segmentId}/chat`,
      { method: "POST", body: JSON.stringify({ question, preset: preset ?? null }) },
    ),

  generate: (body: Record<string, unknown>) =>
    request<{ episode_id: string; stream_url: string }>("/episodes/generate", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

/**
 * Subscribe to generation progress.
 *
 * Returns an unsubscribe function. The caller must call it — an EventSource left
 * open keeps reconnecting after the component unmounts.
 */
export function watchProgress(
  streamUrl: string,
  onEvent: (event: ProgressEvent) => void,
): () => void {
  const source = new EventSource(`${BASE}${streamUrl}`);
  source.onmessage = (message) => {
    try {
      const event = JSON.parse(message.data) as ProgressEvent;
      onEvent(event);
      if (event.stage === "ready" || event.stage === "failed") source.close();
    } catch {
      // A malformed frame is not worth tearing the stream down for.
    }
  };
  source.onerror = () => source.close();
  return () => source.close();
}

export function formatTimestamp(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

export function formatDuration(seconds: number | null): string {
  if (!seconds) return "—";
  return `${Math.round(seconds / 60)} min`;
}
