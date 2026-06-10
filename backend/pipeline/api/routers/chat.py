"""对话: POST /api/v1/chat, POST /api/v1/chat/stream (SSE)"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..authz import require_read
from ..deps import AuthContext, get_pipeline, get_session_store, require_auth
from ..models import ChatAppendRequest, ChatAppendResponse, ChatRequest, ChatResponse, RunStatusResponse
from ..session_logger import clear_session_log_context, set_session_log_context
from ...clients import redis as redis_runtime
from ...db import repo
from ...db.models import Message

logger = logging.getLogger(__name__)

router = APIRouter()


def _ensure_collection_readable(collection: str | None, auth: AuthContext) -> None:
    if not collection or not repo.available():
        return
    meta = repo.get_collection(collection)
    if meta is None:
        raise HTTPException(status_code=404, detail="知识库不存在或不可读")
    require_read(auth, meta)


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


def _conversation_for_turn(req: ChatRequest, auth: AuthContext) -> tuple[str | None, str | None]:
    """Return (conversation_id, parent_message_id) for the next user message."""
    if not repo.available():
        return req.conversation_id, req.parent_message_id
    conv_id = req.conversation_id or f"c_{uuid.uuid4().hex[:16]}"
    conv = repo.get_conversation(conv_id)
    parent_id = req.parent_message_id
    title = req.query.strip().replace("\n", " ")[:48]
    if conv is None:
        repo.upsert_conversation(conversation_id=conv_id, auth=auth, title=title)
        return conv_id, parent_id
    if conv.owner_id == auth.user_id:
        return conv.id, parent_id
    require_read(auth, conv)
    copied = repo.copy_conversation_mainline_to_owner(conv.id, auth)
    return copied.id, copied.active_leaf_message_id


def _persist_user_message(
    *,
    conversation_id: str | None,
    user_message_id: str | None,
    parent_id: str | None,
    query: str,
    params: dict,
) -> str | None:
    if not repo.available() or not conversation_id:
        return user_message_id
    mid = user_message_id or f"m_{uuid.uuid4().hex[:16]}"
    repo.upsert_message(
        Message(
            id=mid,
            conversation_id=conversation_id,
            parent_id=parent_id,
            role="user",
            content=query,
            params=params,
            status="done",
        )
    )
    return mid


def _persist_assistant_message(
    *,
    conversation_id: str | None,
    assistant_message_id: str | None,
    user_message_id: str | None,
    result: dict,
    status: str = "done",
    error: str | None = None,
    session_id: str | None = None,
) -> str | None:
    if not repo.available() or not conversation_id or not user_message_id:
        return assistant_message_id
    mid = assistant_message_id or f"m_{uuid.uuid4().hex[:16]}"
    repo.upsert_message(
        Message(
            id=mid,
            conversation_id=conversation_id,
            parent_id=user_message_id,
            role="assistant",
            content=result.get("answer", ""),
            hits=result.get("hits") or [],
            context=result.get("context"),
            research=result.get("research"),
            usage=result.get("usage"),
            latency_s=result.get("latency_s"),
            status=status,
            error=error,
        )
    )
    # Keep active leaf and pipeline session id aligned with the latest assistant node.
    conv = repo.get_conversation(conversation_id)
    if conv:
        repo.upsert_conversation(
            conversation_id=conversation_id,
            auth=AuthContext(user_id=conv.owner_id, org_id=conv.org_id),
            title=conv.title,
            active_leaf_message_id=mid,
            session_id=session_id,
            forked_from=conv.forked_from,
        )
    return mid


def _require_run_owner(run_id: str, auth: AuthContext):
    run = repo.get_generation_run(run_id) if repo.available() else None
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id != auth.user_id:
        raise HTTPException(status_code=403, detail="No permission for run")
    return run


def _event_payload(event) -> dict:
    payload = dict(event.payload or {})
    payload.setdefault("type", event.type)
    payload["seq"] = event.seq
    payload["run_id"] = event.run_id
    return payload


def _append_and_publish_event(run_id: str, event_type: str, payload: dict) -> None:
    ev = repo.append_message_event(run_id, event_type, payload)
    live_payload = _event_payload(ev)
    if redis_runtime.configured():
        redis_runtime.get_redis_runtime().publish_event(run_id, live_payload)


@router.post("/chat/append", response_model=ChatAppendResponse)
async def append_chat_run(
    req: ChatAppendRequest,
    auth: AuthContext = Depends(require_auth),
) -> ChatAppendResponse:
    """生产级对话入口: 只创建消息/run 并入队, 不在 Web 请求中执行生成。"""
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    if not redis_runtime.configured():
        raise HTTPException(status_code=503, detail="REDIS_URL 未配置")

    _ensure_collection_readable(req.collection, auth)
    conversation_id, parent_id = _conversation_for_turn(req, auth)
    if not conversation_id:
        raise HTTPException(status_code=500, detail="无法创建 conversation")

    user_message_id = _persist_user_message(
        conversation_id=conversation_id,
        user_message_id=req.client_user_message_id,
        parent_id=parent_id,
        query=req.query,
        params=req.model_dump(exclude={"query"}),
    )
    if not user_message_id:
        raise HTTPException(status_code=500, detail="无法创建 user message")

    assistant_message_id = req.client_assistant_message_id or f"m_{uuid.uuid4().hex[:16]}"
    repo.upsert_message(
        Message(
            id=assistant_message_id,
            conversation_id=conversation_id,
            parent_id=user_message_id,
            role="assistant",
            content="",
            params=req.model_dump(exclude={"query"}),
            status="streaming",
        )
    )
    repo.set_conversation_active_leaf(conversation_id, assistant_message_id)

    run_id = f"run_{uuid.uuid4().hex[:16]}"
    stream_key = redis_runtime.get_redis_runtime().stream_key(run_id)
    repo.create_generation_run(
        run_id=run_id,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        owner_id=auth.user_id,
        org_id=auth.org_id,
        params=req.model_dump(),
        redis_stream=stream_key,
        artifact_prefix=f"runs/{auth.user_id}/{run_id}",
    )
    _append_and_publish_event(
        run_id,
        "status",
        {"type": "status", "stage": "queued", "run_id": run_id},
    )
    redis_runtime.get_redis_runtime().enqueue_run(run_id)
    return ChatAppendResponse(
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        run_id=run_id,
        status="queued",
    )


@router.get("/runs/{run_id}/status", response_model=RunStatusResponse)
def get_run_status(
    run_id: str,
    auth: AuthContext = Depends(require_auth),
) -> RunStatusResponse:
    run = _require_run_owner(run_id, auth)
    return RunStatusResponse(
        run_id=run.id,
        conversation_id=run.conversation_id,
        user_message_id=run.user_message_id,
        assistant_message_id=run.assistant_message_id,
        status=run.status,
        error=run.error,
        cancel_requested=run.cancel_requested,
    )


@router.post("/runs/{run_id}/stop", response_model=RunStatusResponse)
def stop_run(
    run_id: str,
    auth: AuthContext = Depends(require_auth),
) -> RunStatusResponse:
    _require_run_owner(run_id, auth)
    run = repo.request_generation_run_stop(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status == "stopped":
        _append_and_publish_event(
            run_id,
            "error",
            {"type": "error", "message": "stopped", "run_id": run_id},
        )
    return RunStatusResponse(
        run_id=run.id,
        conversation_id=run.conversation_id,
        user_message_id=run.user_message_id,
        assistant_message_id=run.assistant_message_id,
        status=run.status,
        error=run.error,
        cancel_requested=run.cancel_requested,
    )


@router.get("/runs/{run_id}/stream")
def stream_run(
    run_id: str,
    after_seq: int = 0,
    auth: AuthContext = Depends(require_auth),
) -> StreamingResponse:
    """SSE: replay durable message_events, then tail Redis stream for live events."""
    _require_run_owner(run_id, auth)

    def event_generator():
        last_seq = max(0, int(after_seq or 0))
        last_redis_id = "$"
        terminal_types = {"done", "error"}
        try:
            while True:
                replayed = repo.list_message_events(run_id, after_seq=last_seq, limit=500)
                for ev in replayed:
                    payload = _event_payload(ev)
                    last_seq = max(last_seq, ev.seq)
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    if payload.get("type") in terminal_types:
                        return

                run = repo.get_generation_run(run_id)
                if run and run.status in {"done", "failed", "stopped"}:
                    # If terminal event was not observed for any reason, close after replay.
                    return

                if redis_runtime.configured():
                    for redis_id, payload in redis_runtime.get_redis_runtime().read_events(
                        run_id,
                        last_id=last_redis_id,
                        block_ms=1000,
                        count=100,
                    ):
                        last_redis_id = redis_id
                        seq = int(payload.get("seq") or 0)
                        if seq <= last_seq:
                            continue
                        last_seq = seq
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        if payload.get("type") in terminal_types:
                            return
                else:
                    time.sleep(1)
        except Exception as e:
            logger.exception("[runs/stream] 生成器异常")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    auth: AuthContext = Depends(require_auth),
) -> ChatResponse:
    """多轮对话: 检索 + 生成, 返回完整结果。"""
    raise HTTPException(
        status_code=410,
        detail="Legacy /chat is disabled; use POST /chat/append and GET /runs/{run_id}/stream",
    )
    pipe = get_pipeline()
    store = get_session_store()
    _ensure_collection_readable(req.collection, auth)
    if req.professional:
        _apply_skill_scope(auth)
    conversation_id, parent_id = _conversation_for_turn(req, auth)
    user_message_id = _persist_user_message(
        conversation_id=conversation_id,
        user_message_id=req.client_user_message_id,
        parent_id=parent_id,
        query=req.query,
        params=req.model_dump(exclude={"query"}),
    )

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

    response = ChatResponse(
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
    _persist_assistant_message(
        conversation_id=conversation_id,
        assistant_message_id=req.client_assistant_message_id,
        user_message_id=user_message_id,
        result=response.model_dump(),
        status="failed" if result.error else "done",
        error=result.error,
        session_id=session_id,
    )
    return response


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    auth: AuthContext = Depends(require_auth),
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
    raise HTTPException(
        status_code=410,
        detail="Legacy /chat/stream is disabled; use POST /chat/append and GET /runs/{run_id}/stream",
    )
    pipe = get_pipeline()
    store = get_session_store()
    _ensure_collection_readable(req.collection, auth)
    if req.professional:
        _apply_skill_scope(auth)
    conversation_id, parent_id = _conversation_for_turn(req, auth)
    user_message_id = _persist_user_message(
        conversation_id=conversation_id,
        user_message_id=req.client_user_message_id,
        parent_id=parent_id,
        query=req.query,
        params=req.model_dump(exclude={"query"}),
    )

    # 切换目标集合 (collection=None/空 → 回退原始默认库 literature_chunks)。
    # 必须经 pipeline 统一切换: stream_chat_events 自身无法感知"原始默认库",
    # 选默认库时前端传 null, 若不在此处回退会沿用上次检索污染的集合。
    pipe._maybe_switch_collection(req.collection)

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
                    if conversation_id:
                        event["conversation_id"] = conversation_id
                    assistant_id = _persist_assistant_message(
                        conversation_id=conversation_id,
                        assistant_message_id=req.client_assistant_message_id,
                        user_message_id=user_message_id,
                        result=event,
                        status="done",
                        session_id=final_session_id,
                    )
                    if assistant_id:
                        event["message_id"] = assistant_id
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
            _persist_assistant_message(
                conversation_id=conversation_id,
                assistant_message_id=req.client_assistant_message_id,
                user_message_id=user_message_id,
                result={"answer": "", "hits": []},
                status="failed",
                error=str(e),
                session_id=session_id,
            )
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
    _auth: str = Depends(require_auth),
) -> dict:
    """创建新的对话会话, 返回 session_id。"""
    store = get_session_store()
    sid = store.create()
    return {"session_id": sid}


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    _auth: str = Depends(require_auth),
) -> dict:
    """销毁对话会话。"""
    store = get_session_store()
    ok = store.delete(session_id)
    return {"deleted": ok}
