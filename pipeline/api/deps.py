"""依赖注入: FastAPI Depends 工厂。"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import Header, HTTPException

if TYPE_CHECKING:
    from ..pipeline import Pipeline
    from .sessions import SessionStore
    from .tasks import TaskStore


# ---------------------------------------------------------------------------
# 单例访问
# ---------------------------------------------------------------------------

_pipeline_instance: "Pipeline | None" = None
_session_store: "SessionStore | None" = None
_task_store: "TaskStore | None" = None
_api_keys: list[str] | None = None


def init_dependencies(
    pipeline: "Pipeline",
    session_store: "SessionStore",
    task_store: "TaskStore",
    api_keys: list[str] | None = None,
) -> None:
    global _pipeline_instance, _session_store, _task_store, _api_keys
    _pipeline_instance = pipeline
    _session_store = session_store
    _task_store = task_store
    _api_keys = api_keys


def get_pipeline() -> "Pipeline":
    if _pipeline_instance is None:
        raise RuntimeError("Pipeline 未初始化")
    return _pipeline_instance


def get_session_store() -> "SessionStore":
    if _session_store is None:
        raise RuntimeError("SessionStore 未初始化")
    return _session_store


def get_task_store() -> "TaskStore":
    if _task_store is None:
        raise RuntimeError("TaskStore 未初始化")
    return _task_store


# ---------------------------------------------------------------------------
# API Key 认证
# ---------------------------------------------------------------------------

async def verify_api_key(authorization: str = Header(default="")) -> str:
    """Bearer token 认证; 未配置 api_keys 时跳过。"""
    if not _api_keys:
        return ""
    token = authorization.removeprefix("Bearer ").strip()
    if token in _api_keys:
        return token
    raise HTTPException(status_code=401, detail="Invalid API key")
