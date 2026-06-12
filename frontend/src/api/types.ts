// 镜像后端 pipeline/api/models.py 与计划中的对话消息树模型。

export type RetrievalMode = 'hybrid' | 'vector' | 'metadata'
export type Visibility = 'private' | 'org' | 'public'

/** 检索源 key（预留多源：当前仅 literature，enterprise_sql 为占位）。 */
export type RetrievalSourceKey = 'literature' | 'enterprise_sql'

export interface Hit {
  pk?: string
  kb_id?: string
  chunk_id?: string
  doc_id?: string
  doc_name?: string
  type?: string
  section?: string
  page_start?: number
  paragraph_index?: number
  publication_year?: number
  bbox?: Record<string, number>
  bboxes?: Array<Record<string, number>>
  page_width?: number
  page_height?: number
  content?: string
  context?: string
  related_assets?: Array<Record<string, unknown>>
  score?: number
  rrf_score?: number
  sources?: string[]
  matched_keywords?: string[]
  rerank_score?: number | null
  subquery_id?: string
  [key: string]: unknown
}

export interface Usage {
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
  [key: string]: unknown
}

export interface ResearchMeta {
  status?: 'complete' | 'insufficient' | 'clarify' | 'error' | string
  rounds: number
  evidence_docs: number
  evidence_chunks: number
  gaps: string[]
  covered: string[]
}

export interface ChatRequest {
  query: string
  session_id?: string | null
  mode?: RetrievalMode | null
  top_k?: number | null
  use_agentic?: boolean
  professional?: boolean
  /** 单库检索；为空用默认库 */
  collection?: string | null
  /** 可选高级筛选；为空表示全部可访问知识 */
  kb_ids?: string[] | null
  /** 限定到具体文献（预留：后端需支持 doc_id 过滤） */
  doc_ids?: string[] | null
  /** 是否启用文献检索（关闭则纯生成） */
  enable_retrieval?: boolean
  /** 启用的检索源（预留多源融合） */
  sources?: RetrievalSourceKey[]
  stream?: boolean
  conversation_id?: string | null
  parent_message_id?: string | null
  client_user_message_id?: string | null
  client_assistant_message_id?: string | null
}

export interface ChatResponse {
  query: string
  answer: string
  hits: Hit[]
  context: string
  usage?: Usage | null
  latency_s: number
  session_id: string
  error?: string | null
  needs_clarify: boolean
  needs_reuse: boolean
  no_answer: boolean
  retry_count: number
  correlation_id: string
  research?: ResearchMeta | null
}

export interface ChatAppendResponse {
  conversation_id: string
  user_message_id: string
  assistant_message_id: string
  run_id: string
  status: string
}

export interface RunStatusResponse {
  run_id: string
  conversation_id: string
  user_message_id: string
  assistant_message_id: string
  status: string
  error?: string | null
  cancel_requested: boolean
}

export type StreamEvent =
  | { type: 'status'; stage: string; seq?: number; run_id?: string }
  | { type: 'thinking'; content: string; round?: number; phase?: string; seq?: number; run_id?: string }
  | { type: 'text'; content: string; seq?: number; run_id?: string }
  | {
      type: 'done'
      answer?: string
      hits?: Hit[]
      context?: string
      session_id?: string
      message_id?: string
      conversation_id?: string
      latency_s?: number
      usage?: Usage
      research?: ResearchMeta | null
      needs_clarify?: boolean
      no_answer?: boolean
      retry_count?: number
      seq?: number
      run_id?: string
      [key: string]: unknown
    }
  | { type: 'error'; message: string; seq?: number; run_id?: string }

// ── 知识库 / 文献 ──────────────────────────────────────────
export interface CollectionInfo {
  name: string
  display_name?: string
  row_count: number
  doc_count?: number
  /** 归属/可见性（后端 M5 落地后返回；当前可能缺省） */
  owner_id?: string
  visibility?: Visibility
  mine?: boolean
  org_id?: string | null
  can_manage?: boolean
}

export interface CollectionsListResponse {
  collections: CollectionInfo[]
}

export interface DocSummaryResponse {
  doc_id: string
  doc_name: string
  title: string
  year?: number | null
  summary: string
  found: boolean
}

export interface DocumentInfo {
  id?: string
  doc_id: string
  collection_name?: string
  title?: string
  filename?: string
  year?: number | null
  pdf_object_key?: string | null
  artifact_prefix?: string | null
  status?: 'parsing' | 'ready' | 'failed' | string
  chunk_count?: number
}

export interface TaskResponse {
  id: string
  status: 'queued' | 'pending' | 'running' | 'done' | 'failed' | 'cancelled' | string
  progress: number
  result?: unknown
  error?: string | null
  created_at: number
}

export interface IngestTaskItem {
  id: string
  doc_id: string
  filename?: string | null
  status: 'pending' | 'running' | 'ready' | 'failed' | 'cancelled' | 'skipped' | string
  error?: string | null
  chunk_count: number
}

export interface IngestTask {
  id: string
  collection_name: string
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | string
  progress: number
  total_items: number
  completed_items: number
  failed_items: number
  skipped_items: number
  cancel_requested: boolean
  result?: Record<string, unknown> | null
  error?: string | null
  created_at: number
  items: IngestTaskItem[]
}

export type IngestTaskEvent =
  | { type: 'status'; status: string; task_id?: string; seq?: number }
  | { type: 'progress'; progress: number; completed_items?: number; failed_items?: number; skipped_items?: number; total_items?: number; task_id?: string; seq?: number }
  | { type: 'item'; item_id: string; doc_id: string; status: string; error?: string | null; chunk_count?: number; task_id?: string; seq?: number }
  | { type: 'done' | 'failed' | 'cancelled' | 'error'; status?: string; message?: string; result?: Record<string, unknown>; task_id?: string; seq?: number }

// ── 技能 ───────────────────────────────────────────────────
export interface SkillSpec {
  id: string
  name: string
  description?: string
  priority?: number
  triggers?: string[]
  prefer_first_paths?: string[]
  sufficiency?: Record<string, unknown>
  tuning?: Record<string, unknown>
  guards?: string[]
  plan?: string
  policy?: string
  synthesis_system?: string
  synthesis_thinking?: string
  synthesis_user?: string
}

export interface SkillSummary extends SkillSpec {
  priority: number
  editable: boolean
  owner_id?: string | null
  org_id?: string | null
  visibility?: Visibility
  mine?: boolean
  can_manage?: boolean
}

export interface SkillListResponse {
  enabled: boolean
  router_mode: string
  upload_dir: string
  skills: SkillSummary[]
}

// ── 运维 ───────────────────────────────────────────────────
export interface HealthResponse {
  status: string
  milvus: string
  llm: string
  embedding: string
  reranker: string
  reflection: string
}

export interface StatsResponse {
  stats: Record<string, unknown>
}

// ── 对话消息树（前端模型；后端 M4 落地后对齐） ─────────────
export interface ChatMessage {
  id: string
  parentId: string | null
  role: 'user' | 'assistant'
  content: string
  /** 同层兄弟分支顺序（分叉重生成产生） */
  hits?: Hit[]
  context?: string
  thinking?: string
  research?: ResearchMeta | null
  latency?: number
  usage?: Usage | null
  status?: 'streaming' | 'done' | 'failed' | 'stopped'
  /** 流式期间的阶段标签（检索中/生成中…），完成后清空。 */
  stage?: string
  error?: string
  expert?: boolean
  createdAt: number
}

export interface Conversation {
  id: string
  title: string
  sessionId: string | null
  visibility: Visibility
  /** 消息树：id -> message */
  messages: Record<string, ChatMessage>
  rootIds: string[]
  activeLeafId: string | null
  updatedAt: number
  ownerId?: string | null
  mine?: boolean
  forkedFrom?: string | null
  shareToken?: string | null
}

export interface BackendConversationPayload {
  id: string
  title?: string
  sessionId?: string | null
  visibility?: Visibility
  messages?: Record<string, Partial<ChatMessage>>
  rootIds?: string[]
  activeLeafId?: string | null
  updatedAt?: number
  ownerId?: string | null
  orgId?: string | null
  mine?: boolean
  forkedFrom?: string | null
}

export interface ConversationListResponse {
  conversations: BackendConversationPayload[]
}

export interface ConversationGetResponse {
  conversation: BackendConversationPayload
}

export interface PdfUrlResponse {
  url: string
  doc_id: string
  collection: string
  expires_in: number
}

export interface ConversationShareResponse {
  token: string
  url: string
}

export interface ResourceCopyResponse {
  id: string
  name?: string | null
}

export type AuthzRole = 'user' | 'admin' | 'root'

export interface AdminMeResponse {
  user_id: string
  org_id?: string | null
  role: AuthzRole
  organizations: string[]
  organization_roles: string[]
  is_admin: boolean
  is_root: boolean
}

export interface AdminAuditLog {
  id: number
  actor_id: string
  actor_role: string
  actor_org_id?: string | null
  target_owner_id?: string | null
  resource_type: string
  resource_id: string
  action: string
  metadata?: Record<string, unknown> | null
  created_at?: string | null
}

export interface AdminIngestTask extends Omit<IngestTask, 'items' | 'created_at'> {
  owner_id: string
  org_id?: string | null
  created_at?: string | null
}

export interface AdminGenerationRun {
  id: string
  conversation_id: string
  user_message_id: string
  assistant_message_id: string
  owner_id: string
  org_id?: string | null
  status: string
  error?: string | null
  cancel_requested: boolean
  created_at?: string | null
}

export interface AdminResourcesResponse {
  collections?: CollectionInfo[]
  conversations?: BackendConversationPayload[]
  skills?: SkillSummary[]
  tasks?: AdminIngestTask[]
  runs?: AdminGenerationRun[]
  logs?: AdminAuditLog[]
}
