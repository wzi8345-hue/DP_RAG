"""运维: GET /api/v1/stats, GET /api/v1/health, 日志查看 API"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...db import repo
from ..authz import require_read
from ..deps import AuthContext, require_auth
from ..models import (
    AdminCollectionDocsRequest,
    AdminConversationRequest,
    AdminListRequest,
    DocIdRequest,
    DocSummaryResponse,
    HealthResponse,
    LogLineEntry,
    LogSessionDetail,
    LogSessionGetRequest,
    LogSessionListResponse,
    LogSessionStreamRequest,
    LogSessionSummary,
    StatsResponse,
)
from ..session_logger import get_session_log_handler

logger = logging.getLogger(__name__)

router = APIRouter()

# 懒加载 + 缓存一个只读 MilvusClient, 用于按 doc_id 取文献简介
_summary_client = None
_summary_collection = ""


def _require_admin_console(auth: AuthContext) -> None:
    if auth.role not in ("admin", "root"):
        raise HTTPException(status_code=403, detail="Admin role required")


def _message_payload(m) -> dict:
    return {
        "id": m.id,
        "parentId": m.parent_id,
        "role": m.role,
        "content": m.content,
        "hits": m.hits or [],
        "context": m.context,
        "research": m.research,
        "latency": m.latency_s,
        "usage": m.usage,
        "status": m.status,
        "error": m.error,
        "createdAt": int(m.created_at.timestamp() * 1000) if m.created_at else 0,
    }


def _conversation_payload(conv, *, include_messages: bool = False) -> dict:
    payload = {
        "id": conv.id,
        "title": conv.title,
        "sessionId": conv.session_id,
        "visibility": conv.visibility,
        "activeLeafId": conv.active_leaf_message_id,
        "updatedAt": int(conv.updated_at.timestamp() * 1000) if conv.updated_at else 0,
        "ownerId": conv.owner_id,
        "orgId": conv.org_id,
        "mine": False,
        "forkedFrom": conv.forked_from,
    }
    if include_messages:
        messages = repo.list_messages(conv.id)
        payload["messages"] = {m.id: _message_payload(m) for m in messages}
        payload["rootIds"] = [m.id for m in messages if m.parent_id is None]
    return payload


def _get_summary_client():
    """复用一个 MilvusClient (按需创建), 返回 (client, collection)。"""
    global _summary_client, _summary_collection
    if _summary_client is None:
        from pymilvus import MilvusClient

        from ...clients.milvus import resolve_milvus_connection
        from ..deps import get_pipeline
        cfg = get_pipeline().config.milvus
        uri, token, db = resolve_milvus_connection(cfg)
        _summary_client = MilvusClient(uri=uri, token=token or "", db_name=db or "")
        _summary_collection = cfg.get("collection", "literature_chunks")
    return _summary_client, _summary_collection


@router.get("/stats", response_model=StatsResponse)
def stats(auth: AuthContext = Depends(require_auth)) -> StatsResponse:
    """查看 Milvus 集合统计。"""
    _require_admin_console(auth)
    from ..deps import get_pipeline
    pipe = get_pipeline()
    raw = pipe.stats()
    # pipe.stats() 返回 StepResult.data, 形如 {"stats": {...}}; 解一层嵌套,
    # 让响应直接是 {total, scanned, doc_count, per_doc}。
    inner = raw.get("stats", raw) if isinstance(raw, dict) else raw
    return StatsResponse(stats=inner)


def _esc(v: str) -> str:
    return str(v).replace('"', '\\"')


@router.post("/doc_summary", response_model=DocSummaryResponse)
def doc_summary(req: DocIdRequest, auth: AuthContext = Depends(require_auth)) -> DocSummaryResponse:
    """按 doc_id 返回该文献的简介 (summary 摘要块), 供前端角标点击展示。

    优先取 type=summary 块; 缺失时回退到 title / 首个 text 块。
    """
    doc_id = req.doc_id
    if repo.available():
        doc = repo.find_document_by_doc_id(doc_id, auth)
        if doc is None:
            return DocSummaryResponse(doc_id=doc_id, found=False)
        collection = repo.get_collection(doc.collection_name)
        if collection is None:
            return DocSummaryResponse(doc_id=doc_id, found=False)
        require_read(auth, collection)

    fields = [
        "chunk_id", "doc_id", "doc_name", "type",
        "section", "page_start", "publication_year", "content",
    ]
    try:
        client, coll = _get_summary_client()
    except Exception as e:
        logger.warning(f"[doc_summary] Milvus 客户端初始化失败: {e}")
        return DocSummaryResponse(doc_id=doc_id, found=False)

    def _query(type_filter: str):
        flt = f'doc_id == "{_esc(doc_id)}"'
        if type_filter:
            flt += f' and type == "{type_filter}"'
        try:
            return client.query(
                collection_name=coll, filter=flt,
                output_fields=fields, limit=1,
            )
        except Exception as e:
            logger.warning(f"[doc_summary] query 失败 ({type_filter}): {e}")
            return []

    rows = _query("summary") or _query("text")
    title_rows = _query("title")
    title = ""
    if title_rows:
        title = (title_rows[0].get("content") or "").strip()

    if not rows:
        return DocSummaryResponse(
            doc_id=doc_id,
            doc_name=(title_rows[0].get("doc_name") if title_rows else "") or doc_id,
            title=title,
            found=bool(title_rows),
        )

    r = rows[0]
    year = r.get("publication_year") or 0
    return DocSummaryResponse(
        doc_id=doc_id,
        doc_name=r.get("doc_name") or doc_id,
        title=title or (r.get("section") or ""),
        year=int(year) if year else None,
        summary=(r.get("content") or "").strip(),
        found=True,
    )


def _config_status(cfg: dict, *, requires_key: bool = True) -> str:
    """根据配置判断某依赖是否就绪 (不发起实际网络调用)。

    - 缺 api_base → not_configured
    - requires_key 且缺 api_key → no_api_key
    - 否则 configured
    """
    if not cfg or not cfg.get("api_base"):
        return "not_configured"
    if requires_key and not cfg.get("api_key"):
        return "no_api_key"
    return "configured"


def _probe_postgres() -> str:
    from ... import db

    if not db.configured():
        return "not_configured"
    try:
        with db.cursor() as cur:
            cur.execute("SELECT 1")
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def _probe_redis() -> str:
    from ...clients import redis as redis_runtime

    if not redis_runtime.configured():
        return "not_configured"
    try:
        redis_runtime.get_redis_runtime().client.ping()
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def _probe_object_store() -> str:
    from ...clients import object_store

    if not object_store.configured():
        return "not_configured"
    try:
        object_store.get_object_store().ensure_bucket()
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查: 检测各依赖服务的连通性与配置完整性。

    Milvus 做实际连通性探测; embedding/reranker/reflection LLM 仅做配置完整性
    检查 (避免每次健康检查都打满后端 GPU)。
    """
    milvus_status = "unknown"
    llm_status = "unknown"
    embedding_status = "unknown"
    reranker_status = "unknown"
    reflection_status = "unknown"
    postgres_status = _probe_postgres()
    redis_status = _probe_redis()
    object_store_status = _probe_object_store()

    try:
        from ..deps import get_pipeline
        pipe = get_pipeline()

        # Milvus 连通性 (实际探测)
        try:
            pipe.stats()
            milvus_status = "ok"
        except Exception as e:
            milvus_status = f"error: {e}"
            logger.warning(f"[health] Milvus 不可达: {e}")

        # 生成 LLM (配置完整性)
        try:
            llm_status = _config_status(pipe.config.generation)
        except Exception:
            llm_status = "error"

        # Embedding (配置完整性; 部分本地服务允许空 key)
        try:
            embedding_status = _config_status(
                pipe.config.embedding, requires_key=False,
            )
        except Exception:
            embedding_status = "error"

        # Reranker / Reflection LLM (仅 LangGraph 启用时才相关)
        try:
            lg_cfg = pipe.config.retrieval.get("langgraph", {}) or {}
            rerank_cfg = lg_cfg.get("reranker", {}) or {}
            if not rerank_cfg.get("enabled", False):
                reranker_status = "disabled"
            else:
                reranker_status = _config_status(rerank_cfg, requires_key=False)

            ref_cfg = lg_cfg.get("reflection", {}) or {}
            if not ref_cfg.get("enabled", True):
                reflection_status = "disabled"
            else:
                # reflection 缺省复用 generation 配置, 故缺 api_base 视为继承
                if not ref_cfg.get("api_base"):
                    reflection_status = "inherits_generation"
                else:
                    reflection_status = _config_status(
                        ref_cfg, requires_key=False,
                    )
        except Exception:
            reranker_status = reranker_status if reranker_status != "unknown" else "error"
            reflection_status = reflection_status if reflection_status != "unknown" else "error"

    except Exception:
        pass  # Pipeline 未初始化

    healthy = milvus_status == "ok" and postgres_status in ("ok", "not_configured") and redis_status in ("ok", "not_configured")
    return HealthResponse(
        status="ok" if healthy else "degraded",
        milvus=milvus_status,
        llm=llm_status,
        embedding=embedding_status,
        reranker=reranker_status,
        reflection=reflection_status,
        postgres=postgres_status,
        redis=redis_status,
        object_store=object_store_status,
    )


# ---------------------------------------------------------------------------
# 组织/平台管理资源 API
# ---------------------------------------------------------------------------


@router.post("/admin/me")
def admin_me(auth: AuthContext = Depends(require_auth)) -> dict:
    return {
        "user_id": auth.user_id,
        "org_id": auth.org_id,
        "role": auth.role,
        "organizations": auth.organizations,
        "organization_roles": auth.organization_roles,
        "is_admin": auth.role in ("admin", "root"),
        "is_root": auth.is_root,
    }


@router.post("/admin/resources/collections")
def admin_collections(auth: AuthContext = Depends(require_auth)) -> dict:
    _require_admin_console(auth)
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    collections = []
    for meta in repo.list_admin_collections(auth):
        docs = repo.list_documents_as(meta.name)
        collections.append(
            {
                "name": meta.name,
                "display_name": meta.display_name,
                "owner_id": meta.owner_id,
                "org_id": meta.org_id,
                "visibility": meta.visibility,
                "doc_count": len([d for d in docs if d.status == "ready"]),
                "row_count": sum(d.chunk_count for d in docs),
                "mine": meta.owner_id == auth.user_id,
                "can_manage": True,
            }
        )
    repo.append_audit_log(
        auth=auth,
        resource_type="kb_collection",
        resource_id="*",
        action="admin_list",
        metadata={"count": len(collections)},
    )
    return {"collections": collections}


@router.post("/admin/resources/collection-documents")
def admin_collection_documents(req: AdminCollectionDocsRequest, auth: AuthContext = Depends(require_auth)) -> dict:
    name = req.name
    _require_admin_console(auth)
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    meta = repo.get_collection(name)
    if meta is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if not (auth.is_root or meta.owner_id == auth.user_id or (auth.role == "admin" and meta.org_id == auth.org_id)):
        raise HTTPException(status_code=403, detail="No admin permission for collection")
    docs = repo.list_documents_as(name)
    repo.append_audit_log(
        auth=auth,
        resource_type="kb_collection",
        resource_id=name,
        action="admin_list_documents",
        target_owner_id=meta.owner_id,
        metadata={"count": len(docs)},
    )
    return {"documents": [d.model_dump(mode="json") for d in docs]}


@router.post("/admin/resources/conversations")
def admin_conversations(auth: AuthContext = Depends(require_auth)) -> dict:
    _require_admin_console(auth)
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    conversations = [_conversation_payload(c) for c in repo.list_admin_conversations(auth)]
    repo.append_audit_log(
        auth=auth,
        resource_type="conversation",
        resource_id="*",
        action="admin_list",
        metadata={"count": len(conversations)},
    )
    return {"conversations": conversations}


@router.post("/admin/resources/conversation")
def admin_conversation(req: AdminConversationRequest, auth: AuthContext = Depends(require_auth)) -> dict:
    conversation_id = req.conversation_id
    _require_admin_console(auth)
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    conv = repo.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    if not (auth.is_root or conv.owner_id == auth.user_id or (auth.role == "admin" and conv.org_id == auth.org_id)):
        raise HTTPException(status_code=403, detail="No admin permission for conversation")
    repo.append_audit_log(
        auth=auth,
        resource_type="conversation",
        resource_id=conversation_id,
        action="admin_read",
        target_owner_id=conv.owner_id,
    )
    return {"conversation": _conversation_payload(conv, include_messages=True)}


@router.post("/admin/resources/skills")
def admin_skills(auth: AuthContext = Depends(require_auth)) -> dict:
    _require_admin_console(auth)
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    skills = [s.model_dump(mode="json") | {"can_manage": True} for s in repo.list_admin_skill_metadata(auth)]
    repo.append_audit_log(
        auth=auth,
        resource_type="skill",
        resource_id="*",
        action="admin_list",
        metadata={"count": len(skills)},
    )
    return {"skills": skills}


@router.post("/admin/resources/ingest-tasks")
def admin_ingest_tasks(
    req: AdminListRequest | None = None,
    auth: AuthContext = Depends(require_auth),
) -> dict:
    limit = req.limit if req else 200
    _require_admin_console(auth)
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    tasks = [t.model_dump(mode="json") for t in repo.list_admin_ingest_tasks(auth, limit=max(1, min(limit, 500)))]
    repo.append_audit_log(
        auth=auth,
        resource_type="ingest_task",
        resource_id="*",
        action="admin_list",
        metadata={"count": len(tasks)},
    )
    return {"tasks": tasks}


@router.post("/admin/resources/generation-runs")
def admin_generation_runs(
    req: AdminListRequest | None = None,
    auth: AuthContext = Depends(require_auth),
) -> dict:
    limit = req.limit if req else 200
    _require_admin_console(auth)
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    runs = [r.model_dump(mode="json") for r in repo.list_admin_generation_runs(auth, limit=max(1, min(limit, 500)))]
    repo.append_audit_log(
        auth=auth,
        resource_type="generation_run",
        resource_id="*",
        action="admin_list",
        metadata={"count": len(runs)},
    )
    return {"runs": runs}


@router.post("/admin/audit-logs")
def admin_audit_logs(
    req: AdminListRequest | None = None,
    auth: AuthContext = Depends(require_auth),
) -> dict:
    limit = req.limit if req else 200
    _require_admin_console(auth)
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    logs = repo.list_audit_logs(auth, limit=max(1, min(limit, 500)))
    return {"logs": [log.model_dump(mode="json") for log in logs]}


# ---------------------------------------------------------------------------
# 日志查看 API
# ---------------------------------------------------------------------------


@router.post("/logs/sessions/list", response_model=LogSessionListResponse)
def list_log_sessions(auth: AuthContext = Depends(require_auth)) -> LogSessionListResponse:
    """返回有日志的 session 列表, 按最后更新时间倒序。"""
    _require_admin_console(auth)
    handler = get_session_log_handler()
    sessions = handler.list_sessions()
    return LogSessionListResponse(
        sessions=[LogSessionSummary(**s) for s in sessions]
    )


@router.post("/logs/sessions/get", response_model=LogSessionDetail)
def get_log_session(
    req: LogSessionGetRequest,
    auth: AuthContext = Depends(require_auth),
) -> LogSessionDetail:
    """返回指定 session 的检索流程日志。"""
    session_id = req.session_id
    tail = req.tail
    _require_admin_console(auth)
    handler = get_session_log_handler()
    detail = handler.get_session(session_id, tail=tail)
    if detail is None:
        return LogSessionDetail(session_id=session_id)
    return LogSessionDetail(
        session_id=detail["session_id"],
        query=detail["query"],
        created_at=detail["created_at"],
        updated_at=detail["updated_at"],
        line_count=detail["line_count"],
        lines=[
            LogLineEntry(
                ts=line["ts"],
                timestamp=line["timestamp"],
                level=line["level"],
                logger=line["logger"],
                message=line["message"],
            )
            for line in detail["lines"]
        ],
    )


@router.post("/logs/sessions/stream")
async def stream_log_session(
    req: LogSessionStreamRequest,
    auth: AuthContext = Depends(require_auth),
) -> StreamingResponse:
    """SSE 实时推送指定 session 的新日志行。"""
    session_id = req.session_id
    _require_admin_console(auth)

    async def event_generator():
        import asyncio

        handler = get_session_log_handler()
        queue = handler.subscribe_session(session_id)
        last_keepalive = time.time()
        try:
            while True:
                # 批量消费队列中已有的日志行
                while queue:
                    line = queue.popleft()
                    data = {
                        "ts": line.ts,
                        "timestamp": line.timestamp,
                        "level": line.level,
                        "logger": line.logger_name,
                        "message": line.message,
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                # 每 15 秒发送 SSE 注释行作为心跳, 防止代理超时断开
                now = time.time()
                if now - last_keepalive >= 15:
                    yield ": keepalive\n\n"
                    last_keepalive = now
                # 无新日志时短暂等待, 避免忙等 (async, 不阻塞事件循环)
                await asyncio.sleep(0.3)
        except GeneratorExit:
            pass
        finally:
            handler.unsubscribe_session(session_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
