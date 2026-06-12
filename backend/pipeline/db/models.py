"""数据库数据结构（pydantic）。

用 pydantic 定义实体（而非 ORM）：psycopg 取回 dict 行后 model_validate 即可，
入库时 model_dump 取字段。可见性 visibility: private(仅 owner) | org(组织内可读) | public(平台可读)。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

Visibility = Literal["private", "org", "public"]
Role = Literal["user", "assistant"]
MessageStatus = Literal["pending", "streaming", "done", "failed", "stopped"]
RunStatus = Literal["queued", "running", "done", "failed", "stopped"]
IngestTaskStatus = Literal["queued", "running", "done", "failed", "cancelled"]
IngestItemStatus = Literal["pending", "running", "ready", "failed", "cancelled", "skipped"]


class Conversation(BaseModel):
    id: str
    owner_id: str
    org_id: str | None = None
    visibility: Visibility = "private"
    title: str = ""
    # 当前主线叶子；沿 parent_id 回溯到根再反转 = 主线
    active_leaf_message_id: str | None = None
    # pipeline 多轮上下文 / 日志会话标识
    session_id: str | None = None
    # copy-on-continue / share continue 的来源，仅作审计展示，不参与鉴权
    forked_from: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Message(BaseModel):
    id: str
    conversation_id: str
    parent_id: str | None = None  # 消息树父指针；分叉点多子
    role: Role
    content: str = ""
    # assistant 元数据
    hits: list[dict[str, Any]] | None = None
    context: str | None = None
    research: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    latency_s: float | None = None
    # 生成参数快照（便于基于该分支重生成）
    params: dict[str, Any] | None = None
    status: MessageStatus = "done"
    error: str | None = None
    created_at: datetime | None = None


class KbCollection(BaseModel):
    name: str  # 业务 KB id/slug，不再代表 Milvus 物理 collection 名
    display_name: str = ""
    owner_id: str
    org_id: str | None = None
    visibility: Visibility = "private"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Document(BaseModel):
    id: str
    collection_name: str  # 业务 KB id/slug
    owner_id: str
    doc_id: str  # pipeline doc_id（原文件名 stem）
    title: str | None = None
    filename: str | None = None
    year: int | None = None
    pdf_object_key: str | None = None
    artifact_prefix: str | None = None
    source_document_id: str | None = None
    status: Literal["parsing", "ready", "failed"] = "parsing"
    task_id: str | None = None
    chunk_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class UserSkill(BaseModel):
    owner_id: str
    id: str
    org_id: str | None = None
    visibility: Visibility = "private"
    name: str = ""
    description: str | None = None
    source_owner_id: str | None = None
    source_skill_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ConversationShare(BaseModel):
    token: str
    conversation_id: str
    owner_id: str
    created_at: datetime | None = None
    revoked_at: datetime | None = None


class GenerationRun(BaseModel):
    id: str
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    owner_id: str
    org_id: str | None = None
    status: RunStatus = "queued"
    params: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False
    redis_stream: str | None = None
    artifact_prefix: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class MessageEvent(BaseModel):
    id: int | None = None
    run_id: str
    seq: int
    type: str
    payload: dict[str, Any]
    created_at: datetime | None = None


class IngestTask(BaseModel):
    id: str
    owner_id: str
    org_id: str | None = None
    collection_name: str  # 业务 KB id/slug
    kind: str = "upload"
    status: IngestTaskStatus = "queued"
    progress: float = 0.0
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    cancel_requested: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None
    params: dict[str, Any] | None = None
    redis_stream: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class IngestTaskItem(BaseModel):
    id: str
    task_id: str
    collection_name: str  # 业务 KB id/slug
    owner_id: str
    doc_id: str
    filename: str | None = None
    pdf_path: str | None = None
    doc_dir: str | None = None
    pdf_object_key: str | None = None
    artifact_prefix: str | None = None
    status: IngestItemStatus = "pending"
    error: str | None = None
    chunk_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class IngestTaskEvent(BaseModel):
    id: int | None = None
    task_id: str
    seq: int
    type: str
    payload: dict[str, Any]
    created_at: datetime | None = None


class AuditLog(BaseModel):
    id: int | None = None
    actor_id: str
    actor_role: str
    actor_org_id: str | None = None
    target_owner_id: str | None = None
    resource_type: str
    resource_id: str
    action: str
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
