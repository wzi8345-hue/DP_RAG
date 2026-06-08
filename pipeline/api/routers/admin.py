"""运维: GET /api/v1/stats, GET /api/v1/health, 日志查看 API"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from ..models import (
    DocSummaryResponse,
    HealthResponse,
    LogLineEntry,
    LogSessionDetail,
    LogSessionListResponse,
    LogSessionSummary,
    StatsResponse,
)
from ..session_logger import get_session_log_handler

logger = logging.getLogger(__name__)

router = APIRouter()

# 懒加载 + 缓存一个只读 MilvusClient, 用于按 doc_id 取文献简介
_summary_client = None
_summary_collection = ""


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
def stats() -> StatsResponse:
    """查看 Milvus 集合统计。"""
    from ..deps import get_pipeline
    pipe = get_pipeline()
    raw = pipe.stats()
    # pipe.stats() 返回 StepResult.data, 形如 {"stats": {...}}; 解一层嵌套,
    # 让响应直接是 {total, scanned, doc_count, per_doc}。
    inner = raw.get("stats", raw) if isinstance(raw, dict) else raw
    return StatsResponse(stats=inner)


def _esc(v: str) -> str:
    return str(v).replace('"', '\\"')


@router.get("/doc_summary", response_model=DocSummaryResponse)
def doc_summary(doc_id: str) -> DocSummaryResponse:
    """按 doc_id 返回该文献的简介 (summary 摘要块), 供前端角标点击展示。

    优先取 type=summary 块; 缺失时回退到 title / 首个 text 块。
    """
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

    return HealthResponse(
        status="ok" if milvus_status == "ok" else "degraded",
        milvus=milvus_status,
        llm=llm_status,
        embedding=embedding_status,
        reranker=reranker_status,
        reflection=reflection_status,
    )


# ---------------------------------------------------------------------------
# 日志查看 API
# ---------------------------------------------------------------------------


@router.get("/logs/sessions", response_model=LogSessionListResponse)
def list_log_sessions() -> LogSessionListResponse:
    """返回有日志的 session 列表, 按最后更新时间倒序。"""
    handler = get_session_log_handler()
    sessions = handler.list_sessions()
    return LogSessionListResponse(
        sessions=[LogSessionSummary(**s) for s in sessions]
    )


@router.get("/logs/sessions/{session_id}", response_model=LogSessionDetail)
def get_log_session(
    session_id: str,
    tail: Optional[int] = Query(None, description="仅返回最后 N 行"),
) -> LogSessionDetail:
    """返回指定 session 的检索流程日志。"""
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


@router.get("/logs/sessions/{session_id}/stream")
async def stream_log_session(session_id: str) -> StreamingResponse:
    """SSE 实时推送指定 session 的新日志行。"""

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
