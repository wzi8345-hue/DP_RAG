"""查询: POST /api/v1/query"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from ..deps import get_pipeline, require_auth
from ..models import QueryRequest, QueryResponse
from ..session_logger import clear_session_log_context, set_session_log_context

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    _auth: str = Depends(require_auth),
) -> QueryResponse:
    """单次查询: retrieve → generate, 返回完整结果。"""
    pipe = get_pipeline()
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
