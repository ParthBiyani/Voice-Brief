/**
 * API client.
 *
 * One place that knows the wire format, so a route change breaks the build rather
 * than a component at runtime.
 */

const BASE = "/api";

/** The single user this app serves. Multi-user would replace this with a login. */
export const USER_EMAIL = "me@example.com";

export type EpisodeStatus = "pending" | "running" | "ready" | "degraded" | "failed";

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
  /** Dependencies from the listener's own repositories this story touches. */
  matched: string[];
  /** Picked for a personal reason: a stack match or a stated interest. */
  personal: boolean;
}

export interface EpisodeSummary {
  id: string;
  title: string | null;
  status: EpisodeStatus;
  mode: string;
  language: string;
  style: string;
  duration_seconds: number | null;
  cost_inr: number | null;
  created_at: string;
}

export interface Grounding {
  attribution_rate?: number;
  sentences?: number;
  supported?: number;
  warnings?: string[];
  tts_engine?: string;
}

export interface EpisodeDetail extends EpisodeSummary {
  segments: Segment[];
  /** Relative to the API root; use `mediaUrl`. */
  audio_url: string | null;
  generation_seconds: number | null;
  error: string | null;
  grounding: Grounding | null;
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

export interface Integration {
  configured: boolean;
  detail: string;
  fetched_at: string | null;
  items: number;
}

export interface Settings {
  email: string;
  github_login: string | null;
  topics: string[];
  language: string;
  style: string;
  github: Integration;
  projectmind: Integration;
  delivery: Integration;
  llm_provider: string;
  tts_engine: string;
  daily_budget_inr: number;
}

export interface SettingsUpdate {
  github_login?: string;
  topics?: string[];
  language?: string;
  style?: string;
}

export interface RankingRow {
  outcome: "picked" | "not_picked" | "aired";
  title: string;
  url?: string | null;
  source: string | null;
  position: number | null;
  score: number | null;
  llm_score: number | null;
  signals: Record<string, number>;
  stack_match: string[];
  reason: string | null;
}

export interface HistoryDay {
  date: string;
  runs: {
    id: string;
    kind: string;
    trigger: string;
    status: string;
    started_at: string;
    finished_at: string | null;
    summary: Record<string, unknown>;
    error: string | null;
  }[];
  episodes: {
    id: string;
    title: string | null;
    status: string;
    error: string | null;
    cost_inr: number;
    duration_seconds: number | null;
    grounding: Grounding;
    ranking: {
      outcome: string;
      title: string;
      source: string | null;
      score: number | null;
      llm: number | null;
      signals: Record<string, number>;
      stack_match: string[];
      why: string | null;
    }[];
    llm_calls: { stage: string; model: string; tokens: number; cost_inr: number }[];
  }[];
  interactions: {
    id: string;
    kind: string;
    channel: string;
    at: string;
    request: Record<string, unknown>;
    response: Record<string, unknown>;
  }[];
}

/** Raised on a 401, so the app can ask for the access token instead of showing an error. */
export class AuthRequired extends Error {
  constructor() {
    super("An access token is required");
  }
}

/** Store the access token in a cookie, which is what lets <audio> and EventSource send it. */
export function saveToken(token: string): void {
  const secure = location.protocol === "https:" ? "; Secure" : "";
  document.cookie = `vb_token=${encodeURIComponent(token)}; path=/; max-age=31536000; SameSite=Strict${secure}`;
}

/** A `?token=` in the address (a shared link) is stored and removed from the URL. */
export function adoptTokenFromUrl(): void {
  const url = new URL(location.href);
  const token = url.searchParams.get("token");
  if (!token) return;
  saveToken(token);
  url.searchParams.delete("token");
  history.replaceState(null, "", url.toString());
}

export function mediaUrl(path: string): string {
  return path.startsWith("http") ? path : `${BASE}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (response.status === 401) throw new AuthRequired();
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}${detail ? `: ${detail}` : ""}`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

const user = `email=${encodeURIComponent(USER_EMAIL)}`;

export const api = {
  health: () => request<{ status: string; llm_provider: string; tts_engine: string }>("/health"),

  listEpisodes: (limit = 30) => request<EpisodeSummary[]>(`/episodes?limit=${limit}`),

  getEpisode: (id: string) => request<EpisodeDetail>(`/episodes/${id}`),

  ranking: (id: string) => request<RankingRow[]>(`/episodes/${id}/ranking`),

  activeGenerations: () => request<string[]>("/episodes/active/now"),

  search: (query: string) =>
    request<{ query: string; hits: SearchHit[] }>(`/search?q=${encodeURIComponent(query)}`),

  vote: (segmentId: string, vote: 1 | -1) =>
    request<void>(`/feedback?${user}`, {
      method: "POST",
      body: JSON.stringify({ segment_id: segmentId, vote }),
    }),

  askSegment: (segmentId: string, question: string, preset?: string) =>
    request<{ answer: string; citations: Citation[]; cost_inr: number }>(
      `/segments/${segmentId}/chat`,
      { method: "POST", body: JSON.stringify({ question, preset: preset ?? null }) },
    ),

  /** Anything left out falls back to the settings saved on the server. */
  generate: (body: Record<string, unknown> = {}) =>
    request<{ episode_id: string; stream_url: string }>("/episodes/generate", {
      method: "POST",
      body: JSON.stringify({ email: USER_EMAIL, ...body }),
    }),

  settings: () => request<Settings>(`/settings?${user}`),

  saveSettings: (update: SettingsUpdate) =>
    request<Settings>(`/settings?${user}`, { method: "PUT", body: JSON.stringify(update) }),

  history: (date?: string) => request<HistoryDay>(`/history${date ? `?date=${date}` : ""}`),

  /** Documents become an episode; progress arrives on the returned stream. */
  uploadDocuments: async (files: File[], language?: string) => {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    form.append("email", USER_EMAIL);
    if (language) form.append("language", language);
    // No Content-Type header: the browser must set the multipart boundary itself.
    const response = await fetch(`${BASE}/documents/podcast`, { method: "POST", body: form });
    if (response.status === 401) throw new AuthRequired();
    if (!response.ok) {
      let detail = response.statusText;
      try {
        detail = ((await response.json()) as { detail?: string }).detail ?? detail;
      } catch {
        // keep the status text
      }
      throw new Error(detail);
    }
    return (await response.json()) as {
      episode_id: string;
      stream_url: string;
      documents: { title: string; kind: string; sections: number }[];
    };
  },
};

/**
 * Follow generation progress.
 *
 * The server replays every event so far, so a dropped connection is simply
 * re-established: nothing is lost. Returns an unsubscribe function, which the caller
 * must call; an EventSource left open keeps reconnecting after the component unmounts.
 */
export function watchProgress(
  streamUrl: string,
  onEvent: (event: ProgressEvent) => void,
): () => void {
  let source: EventSource | null = null;
  let stopped = false;
  let attempts = 0;

  const connect = () => {
    source = new EventSource(`${BASE}${streamUrl}`);
    source.onmessage = (message) => {
      attempts = 0;
      try {
        const event = JSON.parse(message.data) as ProgressEvent;
        onEvent(event);
        if (event.stage === "ready" || event.stage === "failed") {
          stopped = true;
          source?.close();
        }
      } catch {
        // A malformed frame is not worth tearing the stream down for.
      }
    };
    source.onerror = () => {
      source?.close();
      if (stopped) return;
      if (attempts >= 5) {
        onEvent({
          stage: "failed",
          at: new Date().toISOString(),
          error: "Lost the connection to the server",
        });
        return;
      }
      attempts += 1;
      setTimeout(() => !stopped && connect(), 1500 * attempts);
    };
  };

  connect();
  return () => {
    stopped = true;
    source?.close();
  };
}

export function formatTimestamp(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

export function formatDuration(seconds: number | null): string {
  if (!seconds) return "—";
  return `${Math.round(seconds / 60)} min`;
}

/** "Today", "Yesterday", or a short date, in the viewer's own timezone. */
export function dayLabel(iso: string, now = new Date()): string {
  const day = new Date(iso);
  const startOf = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const diff = Math.round((startOf(now) - startOf(day)) / 86_400_000);
  if (diff === 0) return "Today";
  if (diff === 1) return "Yesterday";
  return day.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
}

export function percent(value: number | undefined | null): string {
  return value === undefined || value === null ? "—" : `${Math.round(value * 100)}%`;
}
