"""Pydantic 请求/响应模型。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    row_count: int = 0  # Milvus 中实际入库的数据块数 (flush 后真实值)
    doc_count: int = 0  # Milvus 中实际入库的 distinct doc_id 数 (不含解析/灌入失败的)


class CollectionsListResponse(BaseModel):
    collections: List[CollectionInfo] = []


class CreateCollectionRequest(BaseModel):
    name: str


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
