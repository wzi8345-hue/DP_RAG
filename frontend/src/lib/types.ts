// Mirrors pipeline/api/models.py and the Hit dataclass (retrieval/retrievers.py).

export type RetrievalMode = "hybrid" | "vector" | "metadata";

export interface Hit {
  pk?: string;
  chunk_id?: string;
  doc_id?: string;
  doc_name?: string;
  type?: string;
  section?: string;
  page_start?: number;
  paragraph_index?: number;
  publication_year?: number;
  content?: string;
  context?: string;
  related_assets?: Array<Record<string, unknown>>;
  score?: number;
  rrf_score?: number;
  sources?: string[];
  matched_keywords?: string[];
  rerank_score?: number | null;
  subquery_id?: string;
  subquery_rewrite?: string;
  stage?: string;
  [key: string]: unknown;
}

export interface Usage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  [key: string]: unknown;
}

export interface QueryRequest {
  query: string;
  mode?: RetrievalMode | null;
  top_k?: number | null;
  use_agentic?: boolean;
  /** 专业研究模式: 多轮递进式文献检索 + 综述综合 */
  professional?: boolean;
  /** 目标 Milvus 集合名 (null = 使用配置默认) */
  collection?: string | null;
}

/** 专业研究模式执行概要 (普通模式为 null/缺省) */
export interface ResearchMeta {
  /** complete=正常完成 | insufficient=证据不足无法完成 | clarify=需澄清 | error */
  status?: "complete" | "insufficient" | "clarify" | "error" | string;
  rounds: number;
  evidence_docs: number;
  evidence_chunks: number;
  gaps: string[];
  covered: string[];
}

export interface QueryResponse {
  query: string;
  answer: string;
  hits: Hit[];
  context: string;
  usage?: Usage | null;
  latency_s: number;
  error?: string | null;
  needs_clarify: boolean;
  needs_reuse: boolean;
  no_answer: boolean;
  retry_count: number;
  correlation_id: string;
  research?: ResearchMeta | null;
}

export interface ChatRequest extends QueryRequest {
  session_id?: string | null;
  stream?: boolean;
}

export interface ChatResponse extends QueryResponse {
  session_id: string;
}

export interface TaskResponse {
  id: string;
  status: "pending" | "running" | "done" | "failed" | string;
  progress: number;
  result?: unknown;
  error?: string | null;
  created_at: number;
}

export interface HealthResponse {
  status: string;
  milvus: string;
  llm: string;
  embedding: string;
  reranker: string;
  reflection: string;
}

export interface StatsResponse {
  stats: Record<string, unknown>;
}

export interface FileUploadResponse {
  file_id: string;
  filename: string;
  size_bytes: number;
}

export interface DocSummaryResponse {
  doc_id: string;
  doc_name: string;
  title: string;
  year?: number | null;
  summary: string;
  found: boolean;
}

// ---------------------------------------------------------------------------
// 知识库 (Collection) 管理
// ---------------------------------------------------------------------------

export interface CollectionInfo {
  name: string;
  display_name?: string;
  row_count: number;
  doc_count?: number;
}

export interface CollectionsListResponse {
  collections: CollectionInfo[];
}

// ---------------------------------------------------------------------------
// 专家技能 (Skill) 管理
// ---------------------------------------------------------------------------

export interface SkillSufficiency {
  min_docs?: number;
  need_conflict_check?: boolean;
  need_quantitative_data?: boolean;
  [k: string]: unknown;
}

export interface SkillTuning {
  max_rounds?: number | null;
  max_batches?: number | null;
  gap_stall_limit?: number | null;
  stall_quality_floor?: number | null;
}

export interface SkillSpec {
  id: string;
  name: string;
  description?: string;
  priority?: number;
  triggers?: string[];
  prefer_first_paths?: string[];
  sufficiency?: SkillSufficiency;
  tuning?: SkillTuning;
  guards?: string[];
  plan?: string;
  policy?: string;
  synthesis_system?: string;
  synthesis_thinking?: string;
  synthesis_user?: string;
}

export interface SkillSummary extends SkillSpec {
  id: string;
  name: string;
  priority: number;
  editable: boolean;
}

export interface SkillListResponse {
  enabled: boolean;
  router_mode: string;
  upload_dir: string;
  skills: SkillSummary[];
}

export interface SkillTemplateField {
  key: string;
  label: string;
  type: string;
  required: boolean;
  help: string;
}

export interface SkillTemplate {
  fields: SkillTemplateField[];
  valid_paths: string[];
  valid_guards: string[];
  example: SkillSpec;
}

// ---------------------------------------------------------------------------
// 日志查看 (LogViewer)
// ---------------------------------------------------------------------------

export interface LogSessionSummary {
  session_id: string;
  query: string;
  created_at: number;
  updated_at: number;
  line_count: number;
}

export interface LogLineEntry {
  ts: number;
  timestamp: string;
  level: string;
  logger: string;
  message: string;
}

export interface LogSessionDetail {
  session_id: string;
  query: string;
  created_at: number;
  updated_at: number;
  line_count: number;
  lines: LogLineEntry[];
}

export interface LogSessionListResponse {
  sessions: LogSessionSummary[];
}

// SSE stream events emitted by POST /api/v1/chat/stream
export type StreamEvent =
  | { type: "status"; stage: string }
  | { type: "thinking"; content: string; round?: number; phase?: string }
  | { type: "text"; content: string }
  | {
      type: "done";
      answer?: string;
      hits?: Hit[];
      context?: string;
      session_id?: string;
      latency_s?: number;
      usage?: Usage;
      session_meta?: Record<string, unknown>;
      needs_clarify?: boolean;
      no_answer?: boolean;
      research?: ResearchMeta | null;
      [key: string]: unknown;
    }
  | { type: "error"; message: string };
