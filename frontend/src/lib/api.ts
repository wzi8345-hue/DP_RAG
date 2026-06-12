import type {
  ChatRequest,
  ChatResponse,
  ChunkBboxResponse,
  CollectionInfo,
  CollectionsListResponse,
  DocumentsListResponse,
  DocSummaryResponse,
  FileUploadResponse,
  HealthResponse,
  LogSessionDetail,
  LogSessionListResponse,
  QueryRequest,
  QueryResponse,
  SkillListResponse,
  SkillSpec,
  SkillSummary,
  SkillTemplate,
  StatsResponse,
  StreamEvent,
  TaskResponse,
} from "./types";
import { DEFAULT_COLLECTION } from "./types";
import type { Settings } from "./settings";

const API_PREFIX = "/api/v1";

function baseFor(settings: Settings): string {
  // Empty baseUrl => same-origin (dev proxy handles /api). Otherwise absolute.
  return (settings.baseUrl || "").replace(/\/$/, "");
}

function headers(settings: Settings, json = true): HeadersInit {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  if (settings.apiKey) h["Authorization"] = `Bearer ${settings.apiKey}`;
  return h;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = `${res.status}: ${body.detail}`;
    } catch {
      /* non-json error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export class ApiClient {
  constructor(private settings: Settings) {}

  private url(path: string) {
    return `${baseFor(this.settings)}${API_PREFIX}${path}`;
  }

  async health(): Promise<HealthResponse> {
    const res = await fetch(this.url("/health"), {
      headers: headers(this.settings, false),
    });
    return handle<HealthResponse>(res);
  }

  async stats(): Promise<StatsResponse> {
    const res = await fetch(this.url("/stats"), {
      headers: headers(this.settings, false),
    });
    return handle<StatsResponse>(res);
  }

  async docSummary(docId: string): Promise<DocSummaryResponse> {
    const res = await fetch(
      this.url(`/doc_summary?doc_id=${encodeURIComponent(docId)}`),
      { headers: headers(this.settings, false) }
    );
    return handle<DocSummaryResponse>(res);
  }

  // ── 原文 PDF + 定位框 (引用溯源高亮) ──────────────────────────────
  /** 原始 PDF 的可取回 URL (供 pdf.js getDocument 加载)。 */
  pdfUrl(collection: string, docId: string): string {
    const c = collection || DEFAULT_COLLECTION;
    return this.url(
      `/files/pdf?collection=${encodeURIComponent(c)}&doc_id=${encodeURIComponent(docId)}`
    );
  }

  /** pdf.js getDocument 需要的鉴权头 (未配置 apiKey 时为空)。 */
  authHeaders(): Record<string, string> {
    return this.settings.apiKey
      ? { Authorization: `Bearer ${this.settings.apiKey}` }
      : {};
  }

  /** 取某 chunk 在 PDF 中的定位框; 旧集合无 bboxes 字段时返回空列表。 */
  async getChunkBbox(
    collection: string,
    docId: string,
    chunkId: string
  ): Promise<ChunkBboxResponse> {
    const c = collection || DEFAULT_COLLECTION;
    const res = await fetch(
      this.url(
        `/files/chunk_bbox?collection=${encodeURIComponent(c)}` +
          `&doc_id=${encodeURIComponent(docId)}` +
          `&chunk_id=${encodeURIComponent(chunkId)}`
      ),
      { headers: headers(this.settings, false) }
    );
    return handle<ChunkBboxResponse>(res);
  }

  async query(req: QueryRequest): Promise<QueryResponse> {
    const res = await fetch(this.url("/query"), {
      method: "POST",
      headers: headers(this.settings),
      body: JSON.stringify(req),
    });
    return handle<QueryResponse>(res);
  }

  async chat(req: ChatRequest): Promise<ChatResponse> {
    const res = await fetch(this.url("/chat"), {
      method: "POST",
      headers: headers(this.settings),
      body: JSON.stringify(req),
    });
    return handle<ChatResponse>(res);
  }

  async createSession(): Promise<{ session_id: string }> {
    const res = await fetch(this.url("/sessions"), {
      method: "POST",
      headers: headers(this.settings),
    });
    return handle<{ session_id: string }>(res);
  }

  async deleteSession(id: string): Promise<{ deleted: boolean }> {
    const res = await fetch(this.url(`/sessions/${id}`), {
      method: "DELETE",
      headers: headers(this.settings, false),
    });
    return handle<{ deleted: boolean }>(res);
  }

  async uploadFile(file: File): Promise<FileUploadResponse> {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(this.url("/files/upload"), {
      method: "POST",
      headers: headers(this.settings, false),
      body: fd,
    });
    return handle<FileUploadResponse>(res);
  }

  // ── 知识库 (Collection) 管理 ──────────────────────────────────────

  async listCollections(prefix = "kb_"): Promise<CollectionsListResponse> {
    const res = await fetch(
      this.url(`/collections?prefix=${encodeURIComponent(prefix)}`),
      { headers: headers(this.settings, false) }
    );
    return handle<CollectionsListResponse>(res);
  }

  async listDocuments(name: string): Promise<DocumentsListResponse> {
    const res = await fetch(
      this.url(`/collections/${encodeURIComponent(name)}/documents`),
      { headers: headers(this.settings, false) }
    );
    return handle<DocumentsListResponse>(res);
  }

  async createCollection(name: string): Promise<CollectionInfo> {
    const res = await fetch(this.url("/collections"), {
      method: "POST",
      headers: headers(this.settings),
      body: JSON.stringify({ name }),
    });
    return handle<CollectionInfo>(res);
  }

  async deleteCollection(name: string): Promise<{ deleted: boolean; name: string }> {
    const res = await fetch(this.url(`/collections/${encodeURIComponent(name)}`), {
      method: "DELETE",
      headers: headers(this.settings, false),
    });
    return handle<{ deleted: boolean; name: string }>(res);
  }

  async rebuildCollection(name: string): Promise<TaskResponse> {
    const res = await fetch(
      this.url(`/collections/${encodeURIComponent(name)}/rebuild`),
      { method: "POST", headers: headers(this.settings, false) }
    );
    return handle<TaskResponse>(res);
  }

  // ── 专家技能 (Skill) 管理 ──────────────────────────────────────────

  async listSkills(): Promise<SkillListResponse> {
    const res = await fetch(this.url("/skills"), {
      headers: headers(this.settings, false),
    });
    return handle<SkillListResponse>(res);
  }

  async getSkillTemplate(): Promise<SkillTemplate> {
    const res = await fetch(this.url("/skills/template"), {
      headers: headers(this.settings, false),
    });
    return handle<SkillTemplate>(res);
  }

  async saveSkill(spec: SkillSpec): Promise<{ saved: boolean; id: string; skill: SkillSummary }> {
    const res = await fetch(this.url("/skills"), {
      method: "POST",
      headers: headers(this.settings),
      body: JSON.stringify(spec),
    });
    return handle<{ saved: boolean; id: string; skill: SkillSummary }>(res);
  }

  async deleteSkill(id: string): Promise<{ deleted: boolean; id: string }> {
    const res = await fetch(this.url(`/skills/${encodeURIComponent(id)}`), {
      method: "DELETE",
      headers: headers(this.settings, false),
    });
    return handle<{ deleted: boolean; id: string }>(res);
  }

  async uploadAndIngest(
    files: File[],
    collection: string,
    backend?: string
  ): Promise<TaskResponse> {
    const fd = new FormData();
    for (const f of files) {
      fd.append("files", f);
    }
    fd.append("collection", collection);
    if (backend) fd.append("backend", backend);
    const res = await fetch(this.url("/ingest/upload"), {
      method: "POST",
      headers: headers(this.settings, false),
      body: fd,
    });
    return handle<TaskResponse>(res);
  }

  async ingest(
    kind: "rebuild" | "append" | "parse" | "load-vec",
    body: Record<string, unknown>
  ): Promise<TaskResponse> {
    const res = await fetch(this.url(`/ingest/${kind}`), {
      method: "POST",
      headers: headers(this.settings),
      body: JSON.stringify(body),
    });
    return handle<TaskResponse>(res);
  }

  async getTask(id: string): Promise<TaskResponse> {
    const res = await fetch(this.url(`/tasks/${id}`), {
      headers: headers(this.settings, false),
    });
    return handle<TaskResponse>(res);
  }

  /** Streaming chat via SSE. Calls onEvent for each parsed event. */
  async chatStream(
    req: ChatRequest,
    onEvent: (ev: StreamEvent) => void,
    signal?: AbortSignal
  ): Promise<void> {
    const res = await fetch(this.url("/chat/stream"), {
      method: "POST",
      headers: headers(this.settings),
      body: JSON.stringify({ ...req, stream: true }),
      signal,
    });
    if (!res.ok || !res.body) {
      await handle(res); // throws with detail
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const dataLines = frame
          .split("\n")
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.slice(5).trim());
        if (dataLines.length === 0) continue;
        const payload = dataLines.join("\n");
        try {
          onEvent(JSON.parse(payload) as StreamEvent);
        } catch {
          /* skip malformed frame */
        }
      }
    }
  }

  // ── 日志查看 (LogViewer) ────────────────────────────────────────────

  async listLogSessions(): Promise<LogSessionListResponse> {
    const res = await fetch(this.url("/logs/sessions"), {
      headers: headers(this.settings, false),
    });
    if (!res.ok) return { sessions: [] };
    return handle<LogSessionListResponse>(res);
  }

  async getLogSession(
    sessionId: string,
    tail?: number
  ): Promise<LogSessionDetail> {
    const params = tail ? `?tail=${tail}` : "";
    const res = await fetch(
      this.url(`/logs/sessions/${sessionId}${params}`),
      { headers: headers(this.settings, false) }
    );
    if (!res.ok) {
      return {
        session_id: sessionId,
        query: "",
        created_at: 0,
        updated_at: 0,
        line_count: 0,
        lines: [],
      };
    }
    return handle<LogSessionDetail>(res);
  }

  /** 返回 SSE EventSource URL, 供 LogViewer 实时追踪日志。 */
  logStreamUrl(sessionId: string): string {
    return this.url(`/logs/sessions/${sessionId}/stream`);
  }
}
