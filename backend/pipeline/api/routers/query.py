"""查询: POST /api/v1/query"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ..authz import require_read
from ..deps import AuthContext, get_pipeline, require_auth
from ..models import QueryRequest, QueryResponse
from ..session_logger import clear_session_log_context, set_session_log_context
from ...db import repo

router = APIRouter()


def _apply_skill_scope(auth: AuthContext) -> None:
    if not repo.available():
        return
    pipe = get_pipeline()
    readable_ids = sorted({s.id for s in repo.list_skill_metadata(auth)})
    prof = (pipe.config.retrieval.get("langgraph", {}) or {}).setdefault("professional", {})
    skills_cfg = prof.setdefault("skills", {})
    if skills_cfg.get("allowed_ids") == readable_ids:
        return
    skills_cfg["allowed_ids"] = readable_ids
    try:
        pipe._get_query_flow().reload_skills()
    except Exception:
        pass


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    auth: AuthContext = Depends(require_auth),
) -> QueryResponse:
    """单次查询: retrieve → generate, 返回完整结果。"""
    pipe = get_pipeline()
    if req.collection and repo.available():
        meta = repo.get_collection(req.collection)
        if meta is None:
            raise HTTPException(status_code=404, detail="知识库不存在或不可读")
        require_read(auth, meta)
    if req.professional:
        _apply_skill_scope(auth)
    # 单次 query 用 correlation_id 的前缀作为日志 session 标识
    import uuid
    log_session = uuid.uuid4().hex[:8]
    set_session_log_context(log_session, req.query)
    try:
        result = await asyncio.to_thread(
            pipe.query,
            query=req.query,
            mode=req.mode,
            top_k=req.top_k,
            stream=False,
            use_agentic=req.use_agentic,
            professional=req.professional,
            collection=req.collection,
        )
    finally:
        clear_session_log_context()
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        hits=result.hits,
        context=result.context,
        usage=result.usage,
        latency_s=result.latency_s,
        error=result.error,
        needs_clarify=result.needs_clarify,
        needs_reuse=result.needs_reuse,
        no_answer=result.no_answer,
        retry_count=result.retry_count,
        correlation_id=result.correlation_id,
        research=result.research,
    )
