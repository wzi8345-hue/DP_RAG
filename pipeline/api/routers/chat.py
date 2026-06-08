"""对话: POST /api/v1/chat, POST /api/v1/chat/stream (SSE)"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..deps import get_pipeline, get_session_store, verify_api_key
from ..models import ChatRequest, ChatResponse
from ..session_logger import clear_session_log_context, set_session_log_context
from ...flows.query import ChatSession

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    _auth: str = Depends(verify_api_key),
) -> ChatResponse:
    """多轮对话: 检索 + 生成, 返回完整结果。"""
    pipe = get_pipeline()
    store = get_session_store()

    # 获取或创建会话
    session_id = req.session_id
    if session_id:
        session = store.get(session_id)
        if session is None:
            session_id = store.create()
            session = store.get(session_id)
    else:
        session_id = store.create()
        session = store.get(session_id)

    # 设置日志上下文: 后续 pipeline 日志自动归入该 session
    set_session_log_context(session_id, req.query)

    result, updated_session = await asyncio.to_thread(
        pipe.chat,
        query=req.query,
        session=session,
        mode=req.mode,
        top_k=req.top_k,
        stream=False,
        use_agentic=req.use_agentic,
        professional=req.professional,
        collection=req.collection,
    )

    store.update(session_id, updated_session)
    store.append_messages(session_id, req.query, result.answer)

    clear_session_log_context()

    return ChatResponse(
        query=result.query,
        answer=result.answer,
        hits=result.hits,
        context=result.context,
        usage=result.usage,
        latency_s=result.latency_s,
        session_id=session_id,
        error=result.error,
        needs_clarify=result.needs_clarify,
        needs_reuse=result.needs_reuse,
        no_answer=result.no_answer,
        retry_count=result.retry_count,
        correlation_id=result.correlation_id,
        research=result.research,
    )


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    _auth: str = Depends(verify_api_key),
) -> StreamingResponse:
    """SSE 流式对话: 检索完成后逐块推送 LLM 输出。

    请求体同 /chat, 忽略 req.stream 字段。

    SSE 事件格式:
        data: {"type": "status", "stage": "retrieving"}

        data: {"type": "status", "stage": "generating"}

        data: {"type": "text", "content": "..."}

        data: {"type": "done", "answer": "完整回复", "hits": [...], ...}

        data: {"type": "error", "message": "..."}
    """
    pipe = get_pipeline()
    store = get_session_store()

    session_id = req.session_id
    if session_id:
        session = store.get(session_id)
        if session is None:
            session_id = store.create()
            session = store.get(session_id)
    else:
        session_id = store.create()
        session = store.get(session_id)

    # 设置日志上下文: 后续 pipeline 日志自动归入该 session。
    # 注意: 本端点必须是 async —— set 在请求任务上下文中生效, Starlette 迭代同步生成器时
    # 每步从该上下文复制到线程池, session_id 才可见; 若改成同步 def, set 会落在一次性
    # 线程池上下文里随函数返回被丢弃, 流式期间 emit() 读不到 session_id → 日志全部不收集。
    set_session_log_context(session_id, req.query)

    def event_generator():
        final_session_id = session_id
        try:
            for event in pipe._get_query_flow().stream_chat_events(
                query=req.query,
                session=session,
                use_agentic=req.use_agentic,
                mode=req.mode,
                top_k=req.top_k,
                professional=req.professional,
                collection=req.collection,
            ):
                # 把 session_id 注入 done 事件, 前端据此在流式模式下也能维持多轮
                if event.get("type") == "done":
                    event["session_id"] = final_session_id
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                # 流式完成后更新会话
                if event["type"] == "done":
                    answer = event.get("answer", "")
                    session_meta = event.get("session_meta", {})
                    session.add_turn(
                        query=req.query, answer=answer, meta=session_meta,
                    )
                    store.update(final_session_id, session)
                    store.append_messages(final_session_id, req.query, answer)

        except Exception as e:
            logger.exception("[chat/stream] 生成器异常")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            clear_session_log_context()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx 不缓冲
            "X-Session-Id": session_id,
        },
    )


@router.post("/sessions", response_model=dict)
def create_session(
    _auth: str = Depends(verify_api_key),
) -> dict:
    """创建新的对话会话, 返回 session_id。"""
    store = get_session_store()
    sid = store.create()
    return {"session_id": sid}


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    _auth: str = Depends(verify_api_key),
) -> dict:
    """销毁对话会话。"""
    store = get_session_store()
    ok = store.delete(session_id)
    return {"deleted": ok}
