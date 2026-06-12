"""Pydantic 请求/响应模型。"""

from __future__ import annotations

from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")
Visibility = Literal["private", "org", "public"]


class APIResponse(BaseModel, Generic[T]):
    """统一响应封装（非 SSE 接口）。

    业务接口统一返回 {code, data, msg}：code=0 成功，非 0 表示业务/系统错误。
    SSE 流式接口不走此封装（仍是 text/event-stream）。
    """

    code: int = 0
    data: T | None = None
    msg: str = ""

    @classmethod
    def ok(cls, data: T | None = None, msg: str = "") -> "APIResponse[T]":
        return cls(code=0, data=data, msg=msg)

    @classmethod
    def fail(cls, code: int = 1, msg: str = "", data: T | None = None) -> "APIResponse[T]":
        return cls(code=code, data=data, msg=msg)


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = None  # hybrid / vector / metadata
    top_k: Optional[int] = None
    use_agentic: bool = True
    professional: bool = False  # 专业研究模式: 多轮递进式文献检索 + 综述综合
    collection: Optional[str] = None  # 目标 Milvus 集合 (None = 使用配置默认)
    kb_ids: list[str] | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str = ""
    hits: List[Dict[str, Any]] = []
    context: str = ""
    usage: Optional[Dict[str, Any]] = None
    latency_s: float = 0.0
    error: Optional[str] = None
    # 智能体执行信号 (LangGraph 路径填充)
    needs_clarify: bool = False
    needs_reuse: bool = False
    no_answer: bool = False
    retry_count: int = 0
    correlation_id: str = ""
    # 专业研究模式执行概要 (rounds / evidence_docs / gaps ...); 普通模式为 None
    research: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 对话
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # None = 新建会话
    use_agentic: bool = True
    mode: Optional[str] = None
    top_k: Optional[int] = None
    stream: bool = False
    professional: bool = False  # 专业研究模式: 多轮递进式文献检索 + 综述综合
    collection: Optional[str] = None  # 目标 Milvus 集合 (None = 使用配置默认)
    kb_ids: list[str] | None = None
    conversation_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    client_user_message_id: Optional[str] = None
    client_assistant_message_id: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    answer: str = ""
    hits: List[Dict[str, Any]] = []
    context: str = ""
    usage: Optional[Dict[str, Any]] = None
    latency_s: float = 0.0
    session_id: str = ""
    error: Optional[str] = None
    # 智能体执行信号 (LangGraph 路径填充)
    needs_clarify: bool = False
    needs_reuse: bool = False
    no_answer: bool = False
    retry_count: int = 0
    correlation_id: str = ""
    # 专业研究模式执行概要 (rounds / evidence_docs / gaps ...); 普通模式为 None
    research: Optional[Dict[str, Any]] = None


class ChatAppendRequest(ChatRequest):
    """生产级对话入口：创建消息与 generation run，但不在 Web 请求中生成。"""


class ChatAppendResponse(BaseModel):
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    run_id: str
    status: str = "queued"


class RunStatusResponse(BaseModel):
    run_id: str
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    status: str
    error: str | None = None
    cancel_requested: bool = False


# ---------------------------------------------------------------------------
# 灌入
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    directory: str
    recreate: bool = False
    skip_existing: bool = True
    backend: Optional[str] = None  # mineru / uniparser; 仅 parse 模式


class ParseRequest(BaseModel):
    path: str  # PDF 文件路径或目录
    output_dir: Optional[str] = None
    backend: Optional[str] = None
    timeout: int = 1800


class LoadVecRequest(BaseModel):
    path: str  # 目录 / glob / 单个 .json
    recreate: bool = False
    purge_existing: bool = True
    skip_existing: bool = False


# ---------------------------------------------------------------------------
# 任务
# ---------------------------------------------------------------------------

class TaskResponse(BaseModel):
    id: str
    status: str  # pending / running / done / failed
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = 0.0


class IngestTaskItemResponse(BaseModel):
    id: str
    doc_id: str
    filename: str | None = None
    status: str = "pending"
    error: str | None = None
    chunk_count: int = 0


class IngestTaskResponse(BaseModel):
    id: str
    collection_name: str
    status: str
    progress: float = 0.0
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    cancel_requested: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = 0.0
    items: list[IngestTaskItemResponse] = []


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------

class StatsResponse(BaseModel):
    stats: Dict[str, Any] = {}


class DocSummaryResponse(BaseModel):
    doc_id: str
    doc_name: str = ""
    title: str = ""
    year: Optional[int] = None
    summary: str = ""
    found: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
    milvus: str = "unknown"
    llm: str = "unknown"
    embedding: str = "unknown"
    reranker: str = "unknown"
    reflection: str = "unknown"


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int


class SessionResponse(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# 知识库 (Collection) 管理
# ---------------------------------------------------------------------------

class CollectionInfo(BaseModel):
    name: str  # Milvus 集合名 (kb_ 前缀, 纯 ASCII)
    display_name: str = ""  # 用户可见名 (可含中文); 缺省回退为去前缀的 name
    row_count: int = 0
    doc_count: int = 0  # 本地工作目录中已收纳的文档数 (含尚未灌入的)
    owner_id: str | None = None
    org_id: str | None = None
    visibility: Visibility = "private"
    mine: bool = False
    can_manage: bool = False


class CollectionsListResponse(BaseModel):
    collections: List[CollectionInfo] = []


class CreateCollectionRequest(BaseModel):
    name: str
    visibility: Visibility = "private"


class DeleteCollectionResponse(BaseModel):
    deleted: bool
    name: str


# ---------------------------------------------------------------------------
# 专家技能 (Skill) 管理
# ---------------------------------------------------------------------------

class SkillSpec(BaseModel):
    """新建/编辑一个 skill 的提交体 (前端表单填写)。"""
    id: str
    name: str
    description: str = ""
    priority: Optional[int] = 50
    triggers: List[str] = []
    prefer_first_paths: List[str] = []
    sufficiency: Dict[str, Any] = {}
    tuning: Dict[str, Any] = {}
    guards: List[str] = []
    plan: str = ""
    policy: str = ""
    synthesis_system: str = ""
    synthesis_thinking: str = ""
    synthesis_user: str = ""


class SkillSummary(BaseModel):
    """skill 列表/编辑项 (含可编辑的提示词正文)。"""
    id: str
    name: str
    description: str = ""
    priority: int = 0
    triggers: List[str] = []
    prefer_first_paths: List[str] = []
    sufficiency: Dict[str, Any] = {}
    tuning: Dict[str, Any] = {}
    guards: List[str] = []
    plan: str = ""
    policy: str = ""
    synthesis_system: str = ""
    synthesis_thinking: str = ""
    synthesis_user: str = ""
    editable: bool = False
    owner_id: str | None = None
    org_id: str | None = None
    visibility: Visibility = "private"
    mine: bool = False
    can_manage: bool = False


class SkillListResponse(BaseModel):
    enabled: bool = False
    router_mode: str = "off"
    upload_dir: str = ""
    skills: List[SkillSummary] = []


class SkillSaveResponse(BaseModel):
    saved: bool = True
    id: str
    skill: SkillSummary


class SkillDeleteResponse(BaseModel):
    deleted: bool
    id: str


# ---------------------------------------------------------------------------
# 日志查看
# ---------------------------------------------------------------------------

class LogSessionSummary(BaseModel):
    """session 列表中的一条摘要。"""
    session_id: str
    query: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    line_count: int = 0


class LogLineEntry(BaseModel):
    """单条日志行。"""
    ts: float = 0.0
    timestamp: str = ""
    level: str = ""
    logger: str = ""
    message: str = ""


class LogSessionDetail(BaseModel):
    """指定 session 的完整日志。"""
    session_id: str
    query: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    line_count: int = 0
    lines: List[LogLineEntry] = []


class LogSessionListResponse(BaseModel):
    sessions: List[LogSessionSummary] = []


# ---------------------------------------------------------------------------
# 对象存储 / 分享 / 复制
# ---------------------------------------------------------------------------


class PdfUrlRequest(BaseModel):
    doc_id: str
    collection: str | None = None
    expires_in: int = 900


class PdfUrlResponse(BaseModel):
    url: str
    doc_id: str
    collection: str
    expires_in: int


class ConversationShareRequest(BaseModel):
    conversation_id: str


class ConversationShareResponse(BaseModel):
    token: str
    url: str


class SharedConversationRequest(BaseModel):
    token: str


class ConversationCopyRequest(BaseModel):
    conversation_id: str | None = None
    token: str | None = None


class ResourceCopyRequest(BaseModel):
    id: str


class ResourceCopyResponse(BaseModel):
    id: str
    name: str | None = None


# ---------------------------------------------------------------------------
# POST 化补充请求模型
#
# API 约定（强制，见 ARCHITECTURE §10.0）：业务接口统一 POST + 动词式路径，
# 参数走 JSON body（或 multipart 上传）。下列模型承载原先用 path-param / query
# 传递的参数，使其改由 body 传入。
# ---------------------------------------------------------------------------


class ListCollectionsRequest(BaseModel):
    prefix: str = "kb_"


class CollectionNameRequest(BaseModel):
    name: str


class CollectionVisibilityRequest(BaseModel):
    name: str
    visibility: Visibility


class CollectionDocumentRequest(BaseModel):
    name: str
    doc_id: str


class TaskIdRequest(BaseModel):
    task_id: str


class StreamTaskRequest(BaseModel):
    task_id: str
    after_seq: int = 0


class SkillIdRequest(BaseModel):
    skill_id: str


class SkillVisibilityRequest(BaseModel):
    skill_id: str
    visibility: Visibility


class ConversationIdRequest(BaseModel):
    conversation_id: str


class ConversationVisibilityRequest(BaseModel):
    conversation_id: str
    visibility: Visibility


class RunIdRequest(BaseModel):
    run_id: str


class StreamRunRequest(BaseModel):
    run_id: str
    after_seq: int = 0


class DocIdRequest(BaseModel):
    doc_id: str


class SessionIdRequest(BaseModel):
    session_id: str


class AdminListRequest(BaseModel):
    limit: int = 200


class AdminCollectionDocsRequest(BaseModel):
    name: str


class AdminConversationRequest(BaseModel):
    conversation_id: str


class LogSessionGetRequest(BaseModel):
    session_id: str
    tail: int | None = None


class LogSessionStreamRequest(BaseModel):
    session_id: str
