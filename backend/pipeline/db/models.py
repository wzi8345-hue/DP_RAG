"""数据库数据结构（pydantic）。

用 pydantic 定义实体（而非 ORM）：psycopg 取回 dict 行后 model_validate 即可，
入库时 model_dump 取字段。可见性 visibility: private(仅 owner) | org(组织内可读)。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

Visibility = Literal["private", "org"]
Role = Literal["user", "assistant"]
MessageStatus = Literal["pending", "streaming", "done", "failed", "stopped"]


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
    name: str  # Milvus 集合名 kb_xxx
    display_name: str = ""
    owner_id: str
    org_id: str | None = None
    visibility: Visibility = "private"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Document(BaseModel):
    id: str
    collection_name: str
    owner_id: str
    doc_id: str  # pipeline doc_id（原文件名 stem）
    title: str | None = None
    filename: str | None = None
    year: int | None = None
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
    created_at: datetime | None = None
    updated_at: datetime | None = None
