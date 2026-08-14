// Typed API client for the DocuMind FastAPI backend.
// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.ts).
const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export type SourceType = "pdf" | "docx" | "text";

export interface DocumentInfo {
  doc_id: string;
  doc_name: string;
  pages: number;
  chunks: number;
  source_type: SourceType;
  summary: string | null;
  suggested_questions: string[];
}

export interface Citation {
  marker: number;
  doc_id: string;
  doc_name: string;
  page: number;
  section: string | null;
  snippet: string;
  score: number;
  source_type: SourceType;
}

export interface Source {
  marker: number;
  chunk_id?: string;
  doc_id: string;
  doc_name: string;
  page: number;
  section: string | null;
  score: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface AskMeta {
  query_used: string;
  rewritten: boolean;
}

export interface TraceEntry {
  chunk_id: string;
  doc_name: string;
  page: number;
  rank: number;
  score: number;
}

export interface RetrievalTrace {
  query_used: string;
  rewritten: boolean;
  dense: TraceEntry[];
  bm25: TraceEntry[];
  fused: TraceEntry[];
  reranked: TraceEntry[];
  timings_ms: Record<string, number>;
}

export interface AnswerResponse {
  answer: string;
  citations: Citation[];
  grounded: boolean;
  retrieval_ms: number;
  generation_ms: number;
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = (body as any).detail;
    // detail is either a plain string or a structured {code, message} object.
    const msg =
      typeof detail === "string"
        ? detail
        : detail?.message ?? `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  return jsonOrThrow(await fetch(`${BASE}/documents`));
}

export async function uploadDocuments(files: File[]): Promise<void> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  await jsonOrThrow(await fetch(`${BASE}/ingest`, { method: "POST", body: form }));
}

export async function deleteDocument(docId: string): Promise<void> {
  await jsonOrThrow(
    await fetch(`${BASE}/documents/${docId}`, { method: "DELETE" })
  );
}

// URL of the original uploaded file (rendered by the citation PDF viewer).
export function documentFileUrl(docId: string): string {
  return `${BASE}/documents/${docId}/file`;
}

export interface Health {
  status: string;
  qdrant_ok: boolean;
  indexed_chunks: number;
  models_ready: boolean;
}

export async function health(): Promise<Health> {
  return jsonOrThrow(await fetch(`${BASE}/health`));
}

// Streaming ask via SSE-over-POST. We parse the event stream manually because
// EventSource doesn't support POST bodies.
export interface StreamHandlers {
  onMeta?: (meta: AskMeta) => void;
  onSources?: (sources: Source[]) => void;
  onTrace?: (trace: RetrievalTrace) => void;
  onToken?: (token: string) => void;
  onDone?: (payload: { grounded: boolean; citations: Citation[] }) => void;
  onError?: (err: Error) => void;
}

export interface AskStreamBody {
  question: string;
  doc_ids: string[] | null;
  history?: ChatTurn[];
  include_trace?: boolean;
}

export async function askStream(
  body: AskStreamBody,
  handlers: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  try {
    const res = await fetch(`${BASE}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok || !res.body) throw new Error(`Stream failed (${res.status})`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      const frames = buf.split("\n\n");
      buf = frames.pop() ?? "";
      for (const frame of frames) {
        const ev = /event:\s*(.*)/.exec(frame)?.[1]?.trim();
        const dataLine = /data:\s*([\s\S]*)/.exec(frame)?.[1];
        if (!ev || dataLine == null) continue;
        const data = JSON.parse(dataLine);
        if (ev === "meta") handlers.onMeta?.(data);
        else if (ev === "sources") handlers.onSources?.(data);
        else if (ev === "trace") handlers.onTrace?.(data);
        else if (ev === "token") handlers.onToken?.(data);
        else if (ev === "done") handlers.onDone?.(data);
      }
    }
  } catch (err) {
    if ((err as Error).name !== "AbortError") {
      handlers.onError?.(err as Error);
    }
  }
}
