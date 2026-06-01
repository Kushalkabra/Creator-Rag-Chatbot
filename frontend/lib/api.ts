const LOCAL_API_URL = "http://localhost:8000";

function normalizeBaseUrl(url: string): string {
  return url.trim().replace(/\/$/, "");
}

function isValidExternalBackendUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return false;
    }
    if (typeof window !== "undefined" && parsed.origin === window.location.origin) {
      // e.g. NEXT_PUBLIC_API_URL mistakenly set to the Vercel site URL
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

function resolveApiUrl(): string {
  const configured = normalizeBaseUrl(process.env.NEXT_PUBLIC_API_URL ?? "");
  const forceProxy = process.env.NEXT_PUBLIC_USE_API_PROXY === "true";

  if (typeof window !== "undefined") {
    if (forceProxy || !isValidExternalBackendUrl(configured)) {
      return "/api";
    }
    return configured;
  }

  // Server-side (build/SSR): talk to backend directly when proxying via rewrites
  if (forceProxy || !isValidExternalBackendUrl(configured)) {
    const backend = normalizeBaseUrl(
      process.env.BACKEND_URL ?? process.env.RAILWAY_BACKEND_URL ?? ""
    );
    return backend || LOCAL_API_URL;
  }

  return configured;
}

export const API_URL = resolveApiUrl();

export type VideoMeta = {
  label: string;
  url: string;
  creator: string;
  views: number;
  likes: number;
  comments: number;
  engagement_rate: number;
  hashtags: string[];
  duration: number;
  upload_date: string;
};

export type IngestResponse = {
  A: VideoMeta;
  B: VideoMeta;
};

export type EvalResult = {
  question: string;
  passed: boolean;
  recall_at_k: number;
  matched_keywords: string[];
  missing_keywords: string[];
  video_ids_retrieved: string[];
  notes: string;
  retrieved_excerpts: string[];
};

export type EvalSummary = {
  pass_rate: number;
  average_recall_at_k: number;
  passed: number;
  total_cases: number;
  summary: string;
  results: EvalResult[];
};

export type MetricsResponse = {
  session: {
    chunks_stored: number;
    chat_turns: number;
    ingestion_count: number;
    embedding_tokens_estimated: number;
    chat_input_tokens_estimated: number;
    chat_output_tokens_estimated: number;
    whisper_minutes: number;
  };
  cost: {
    embeddings_usd: number;
    chat_usd: number;
    whisper_usd: number;
    total_session_usd: number;
    cost_per_creator_usd: number;
  };
  model_info: {
    llm: string;
    embeddings: string;
    whisper: string;
    llm_input_cost_per_1m: number;
    llm_output_cost_per_1m: number;
    embedding_cost_per_1m: number;
  };
  vs_openai_gpt4o: {
    estimated_cost_usd: number;
    our_cost_usd: number;
    savings_usd: number;
    savings_percent: number;
  };
  scale_projections: {
    cost_per_creator: number;
    daily: Record<string, number>;
    monthly: Record<string, number>;
    vs_gpt4o_monthly_1000_creators: number;
  };
};

export type ChatSource = {
  video_id: string;
  tag: string;
  chunk_index?: number;
  source_url?: string;
  creator?: string;
  engagement_rate?: number;
  excerpt?: string;
};

type RawVideoMeta = {
  video_label?: string;
  label?: string;
  url?: string;
  creator?: string;
  views?: number;
  likes?: number;
  comments?: number;
  engagement_rate?: number;
  hashtags?: string[];
  duration?: number;
  upload_date?: string;
};

function normalizeVideoMeta(raw: RawVideoMeta, fallbackLabel: "A" | "B"): VideoMeta {
  return {
    label: raw.label ?? raw.video_label ?? fallbackLabel,
    url: raw.url ?? "",
    creator: raw.creator ?? "Unknown",
    views: raw.views ?? 0,
    likes: raw.likes ?? 0,
    comments: raw.comments ?? 0,
    engagement_rate: raw.engagement_rate ?? 0,
    hashtags: raw.hashtags ?? [],
    duration: raw.duration ?? 0,
    upload_date: raw.upload_date ?? "",
  };
}

function sourceTagsFromPayload(sources: ChatSource[]): string[] {
  const tags = sources.map(
    (source) => source.tag || `[Video ${source.video_id}]`
  );
  return Array.from(new Set(tags));
}

async function parseApiError(response: Response, fallback: string): Promise<string> {
  const text = await response.text();
  if (!text) return fallback;

  if (text.trimStart().startsWith("<")) {
    return (
      "Request reached the Next.js frontend (404 HTML), not the FastAPI backend. " +
      "On Vercel set BACKEND_URL to your Railway URL (e.g. https://xxx.up.railway.app) " +
      "and redeploy — the app proxies /api/* to that host. " +
      "Remove NEXT_PUBLIC_API_URL if it points at your Vercel domain."
    );
  }

  try {
    const json = JSON.parse(text) as { detail?: string | Array<{ msg?: string }> };
    if (typeof json.detail === "string") return json.detail;
    if (Array.isArray(json.detail)) {
      return json.detail
        .map((item) => item.msg ?? JSON.stringify(item))
        .join(", ");
    }
  } catch {
    // Plain-text error body from the API.
  }

  return text;
}

export async function ingestVideos(
  youtubeUrl: string,
  instagramUrl: string
): Promise<IngestResponse> {
  const response = await fetch(`${API_URL}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      youtube_url: youtubeUrl,
      instagram_url: instagramUrl,
    }),
  });

  if (!response.ok) {
    throw new Error(
      await parseApiError(response, `Ingest failed (${response.status})`)
    );
  }

  const data = (await response.json()) as { A: RawVideoMeta; B: RawVideoMeta };
  return {
    A: normalizeVideoMeta(data.A, "A"),
    B: normalizeVideoMeta(data.B, "B"),
  };
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const response = await fetch(`${API_URL}/metrics`);

  if (!response.ok) {
    throw new Error(
      await parseApiError(response, `Metrics failed (${response.status})`)
    );
  }

  return (await response.json()) as MetricsResponse;
}

export async function runEval(): Promise<EvalSummary> {
  const response = await fetch(`${API_URL}/eval`);

  if (!response.ok) {
    throw new Error(
      await parseApiError(response, `Eval failed (${response.status})`)
    );
  }

  return (await response.json()) as EvalSummary;
}

export async function streamChat(
  message: string,
  sessionId: string,
  onToken: (token: string) => void,
  onDone: (sources: string[]) => void
): Promise<void> {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    throw new Error(
      await parseApiError(response, `Chat failed (${response.status})`)
    );
  }

  if (!response.body) {
    throw new Error("Chat response has no body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let lastSourceTags: string[] = [];
  let streamCompleted = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;

        const jsonStr = trimmed.slice(5).trim();
        if (!jsonStr) continue;

        let payload: { token?: string; sources?: ChatSource[]; done?: boolean };
        try {
          payload = JSON.parse(jsonStr) as {
            token?: string;
            sources?: ChatSource[];
            done?: boolean;
          };
        } catch {
          continue;
        }

        if (payload.done) {
          streamCompleted = true;
          onDone(lastSourceTags);
          return;
        }

        if (payload.sources?.length) {
          lastSourceTags = sourceTagsFromPayload(payload.sources);
        }

        if (payload.token) {
          onToken(payload.token);
        }
      }
    }
  } catch (error) {
    if (!streamCompleted) {
      if (error instanceof Error) throw error;
      throw new Error("Network error while streaming chat response");
    }
    throw error;
  }

  if (!streamCompleted) {
    throw new Error("Network error: chat stream ended unexpectedly");
  }
}
