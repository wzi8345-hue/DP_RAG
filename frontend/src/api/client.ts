import { consumeSSE } from './stream'
import type {
  ChatRequest,
  ChatAppendResponse,
  CollectionInfo,
  CollectionsListResponse,
  ConversationGetResponse,
  ConversationListResponse,
  DocSummaryResponse,
  DocumentInfo,
  HealthResponse,
  IngestTask,
  IngestTaskEvent,
  PdfUrlResponse,
  ResourceCopyResponse,
  RunStatusResponse,
  SkillListResponse,
  SkillSpec,
  StatsResponse,
  StreamEvent,
  TaskResponse,
  Visibility,
  ConversationShareResponse,
} from './types'

const API_PREFIX = '/api/v1'

export interface ApiConfig {
  /** 后端 base，如 https://funmg.dp.tech/sci-loop-api；空串=同源（dev proxy）。可随设置变化。 */
  base: () => string
  /** 返回 Logto access_token（JWT）；未登录返回 undefined。 */
  getToken: () => Promise<string | undefined>
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = `${res.status}: ${body.detail}`
    } catch {
      /* non-json */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export class ApiClient {
  constructor(private cfg: ApiConfig) {}

  private url(path: string): string {
    return `${this.cfg.base().replace(/\/$/, '')}${API_PREFIX}${path}`
  }

  private async headers(json = true): Promise<HeadersInit> {
    const h: Record<string, string> = {}
    if (json) h['Content-Type'] = 'application/json'
    const token = await this.cfg.getToken()
    if (token) h['Authorization'] = `Bearer ${token}`
    return h
  }

  // ── 运维 ────────────────────────────────────────────────
  async health(): Promise<HealthResponse> {
    return handle(await fetch(this.url('/health'), { headers: await this.headers(false) }))
  }

  async stats(): Promise<StatsResponse> {
    return handle(await fetch(this.url('/stats'), { headers: await this.headers(false) }))
  }

  async docSummary(docId: string): Promise<DocSummaryResponse> {
    return handle(
      await fetch(this.url(`/doc_summary?doc_id=${encodeURIComponent(docId)}`), {
        headers: await this.headers(false),
      }),
    )
  }

  // ── 知识库 / 文献 ───────────────────────────────────────
  async listCollections(prefix = 'kb_'): Promise<CollectionsListResponse> {
    return handle(
      await fetch(this.url(`/collections?prefix=${encodeURIComponent(prefix)}`), {
        headers: await this.headers(false),
      }),
    )
  }

  async createCollection(name: string): Promise<CollectionInfo> {
    return handle(
      await fetch(this.url('/collections'), {
        method: 'POST',
        headers: await this.headers(),
        body: JSON.stringify({ name }),
      }),
    )
  }

  async deleteCollection(name: string): Promise<{ deleted: boolean; name: string }> {
    return handle(
      await fetch(this.url(`/collections/${encodeURIComponent(name)}`), {
        method: 'DELETE',
        headers: await this.headers(false),
      }),
    )
  }

  async rebuildCollection(name: string): Promise<TaskResponse> {
    return handle(
      await fetch(this.url(`/collections/${encodeURIComponent(name)}/rebuild`), {
        method: 'POST',
        headers: await this.headers(false),
      }),
    )
  }

  /** 设置文献库可见性（后端 M5；当前若 404 调用方降级处理）。 */
  async setCollectionVisibility(name: string, visibility: Visibility): Promise<void> {
    const res = await fetch(this.url(`/collections/${encodeURIComponent(name)}/visibility`), {
      method: 'PATCH',
      headers: await this.headers(),
      body: JSON.stringify({ visibility }),
    })
    if (!res.ok) await handle(res)
  }

  async copyCollectionToMine(id: string): Promise<ResourceCopyResponse> {
    return handle(
      await fetch(this.url('/collections/copy-to-mine'), {
        method: 'POST',
        headers: await this.headers(),
        body: JSON.stringify({ id }),
      }),
    )
  }

  /** 列出某库下文献（后端 M5；当前后端可能未实现 → 调用方 try/catch 降级）。 */
  async listDocuments(collection: string): Promise<DocumentInfo[]> {
    const res = await fetch(
      this.url(`/collections/${encodeURIComponent(collection)}/documents`),
      { headers: await this.headers(false) },
    )
    if (!res.ok) return []
    const data = (await res.json()) as { documents?: DocumentInfo[] }
    return data.documents ?? []
  }

  async deleteDocument(collection: string, docId: string): Promise<void> {
    const res = await fetch(
      this.url(
        `/collections/${encodeURIComponent(collection)}/documents/${encodeURIComponent(docId)}`,
      ),
      { method: 'DELETE', headers: await this.headers(false) },
    )
    if (!res.ok) await handle(res)
  }

  async getDocumentPdfUrl(
    docId: string,
    collection?: string | null,
    expiresIn = 900,
  ): Promise<PdfUrlResponse> {
    return handle(
      await fetch(this.url('/documents/pdf-url'), {
        method: 'POST',
        headers: await this.headers(),
        body: JSON.stringify({ doc_id: docId, collection, expires_in: expiresIn }),
      }),
    )
  }

  async uploadAndIngest(
    files: File[],
    collection: string,
    backend?: string,
  ): Promise<TaskResponse> {
    const fd = new FormData()
    for (const f of files) fd.append('files', f)
    fd.append('collection', collection)
    if (backend) fd.append('backend', backend)
    return handle(
      await fetch(this.url('/ingest/upload'), {
        method: 'POST',
        headers: await this.headers(false),
        body: fd,
      }),
    )
  }

  async getTask(id: string): Promise<TaskResponse> {
    return handle(
      await fetch(this.url(`/tasks/${id}`), { headers: await this.headers(false) }),
    )
  }

  async getIngestTask(id: string): Promise<IngestTask> {
    return handle(
      await fetch(this.url(`/ingest/tasks/${encodeURIComponent(id)}`), {
        headers: await this.headers(false),
      }),
    )
  }

  async streamIngestTask(
    id: string,
    onEvent: (ev: IngestTaskEvent) => void,
    signal?: AbortSignal,
    afterSeq = 0,
  ): Promise<void> {
    const res = await fetch(
      this.url(`/ingest/tasks/${encodeURIComponent(id)}/stream?after_seq=${encodeURIComponent(afterSeq)}`),
      {
        headers: await this.headers(false),
        signal,
      },
    )
    if (!res.ok || !res.body) {
      await handle(res)
      return
    }
    await consumeSSE(res, onEvent as (ev: StreamEvent) => void)
  }

  async cancelIngestTask(id: string): Promise<IngestTask> {
    return handle(
      await fetch(this.url(`/ingest/tasks/${encodeURIComponent(id)}/cancel`), {
        method: 'POST',
        headers: await this.headers(false),
      }),
    )
  }

  // ── 技能 ────────────────────────────────────────────────
  async listSkills(): Promise<SkillListResponse> {
    return handle(await fetch(this.url('/skills'), { headers: await this.headers(false) }))
  }

  async getSkillTemplate(): Promise<Record<string, unknown>> {
    return handle(
      await fetch(this.url('/skills/template'), { headers: await this.headers(false) }),
    )
  }

  async saveSkill(spec: SkillSpec): Promise<unknown> {
    return handle(
      await fetch(this.url('/skills'), {
        method: 'POST',
        headers: await this.headers(),
        body: JSON.stringify(spec),
      }),
    )
  }

  async deleteSkill(id: string): Promise<{ deleted: boolean; id: string }> {
    return handle(
      await fetch(this.url(`/skills/${encodeURIComponent(id)}`), {
        method: 'DELETE',
        headers: await this.headers(false),
      }),
    )
  }

  async setSkillVisibility(id: string, visibility: Visibility): Promise<void> {
    const res = await fetch(this.url(`/skills/${encodeURIComponent(id)}/visibility`), {
      method: 'PATCH',
      headers: await this.headers(),
      body: JSON.stringify({ visibility }),
    })
    if (!res.ok) await handle(res)
  }

  async copySkillToMine(id: string): Promise<ResourceCopyResponse> {
    return handle(
      await fetch(this.url('/skills/copy-to-mine'), {
        method: 'POST',
        headers: await this.headers(),
        body: JSON.stringify({ id }),
      }),
    )
  }

  // ── 对话分享 ─────────────────────────────────────────────
  async shareConversation(conversationId: string): Promise<ConversationShareResponse> {
    return handle(
      await fetch(this.url('/conversations/share'), {
        method: 'POST',
        headers: await this.headers(),
        body: JSON.stringify({ conversation_id: conversationId }),
      }),
    )
  }

  async unshareConversation(conversationId: string): Promise<void> {
    const res = await fetch(this.url('/conversations/unshare'), {
      method: 'POST',
      headers: await this.headers(),
      body: JSON.stringify({ conversation_id: conversationId }),
    })
    if (!res.ok) await handle(res)
  }

  async getSharedConversation(token: string): Promise<ConversationGetResponse> {
    return handle(
      await fetch(this.url('/conversations/shared/get'), {
        method: 'POST',
        headers: await this.headers(),
        body: JSON.stringify({ token }),
      }),
    )
  }

  async listConversations(): Promise<ConversationListResponse> {
    return handle(
      await fetch(this.url('/conversations'), {
        headers: await this.headers(false),
      }),
    )
  }

  async getConversation(conversationId: string): Promise<ConversationGetResponse> {
    return handle(
      await fetch(this.url(`/conversations/${encodeURIComponent(conversationId)}`), {
        headers: await this.headers(false),
      }),
    )
  }

  async copySharedConversationToMine(token: string): Promise<{ conversation_id: string }> {
    return handle(
      await fetch(this.url('/conversations/copy-to-mine'), {
        method: 'POST',
        headers: await this.headers(),
        body: JSON.stringify({ token }),
      }),
    )
  }
  // ── 生产级 run-based 流式问答 ───────────────────────────
  async appendChatRun(req: ChatRequest): Promise<ChatAppendResponse> {
    return handle(
      await fetch(this.url('/chat/append'), {
        method: 'POST',
        headers: await this.headers(),
        body: JSON.stringify(req),
      }),
    )
  }

  async streamRun(
    runId: string,
    onEvent: (ev: StreamEvent) => void,
    signal?: AbortSignal,
    afterSeq = 0,
  ): Promise<void> {
    const res = await fetch(
      this.url(`/runs/${encodeURIComponent(runId)}/stream?after_seq=${encodeURIComponent(afterSeq)}`),
      {
        headers: await this.headers(false),
        signal,
      },
    )
    if (!res.ok || !res.body) {
      await handle(res)
      return
    }
    await consumeSSE(res, onEvent)
  }

  async getRunStatus(runId: string): Promise<RunStatusResponse> {
    return handle(
      await fetch(this.url(`/runs/${encodeURIComponent(runId)}/status`), {
        headers: await this.headers(false),
      }),
    )
  }

  async stopRun(runId: string): Promise<RunStatusResponse> {
    return handle(
      await fetch(this.url(`/runs/${encodeURIComponent(runId)}/stop`), {
        method: 'POST',
        headers: await this.headers(false),
      }),
    )
  }

}
