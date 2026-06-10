"""FastAPI 应用: lifespan 初始化 + 路由注册。"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .deps import init_dependencies
from .session_logger import get_session_log_handler
from .sessions import SessionStore
from .tasks import TaskStore

logger = logging.getLogger(__name__)


def _startup_secret_check(pipeline) -> None:
    """启动自检: 关键密钥为空时打一条醒目的汇总告警。

    背景: 后端若没 `source .env.local` 就启动, ${VAR} 占位符会退化成空串,
    导致 MinerU 解析 401 (chunk=0)、embedding 401/404 (无法入库) 等"静默失败"。
    这里在启动时集中提示, 避免上传后才从一堆日志里反推根因。仅告警, 不阻断启动。
    """
    try:
        cfg = pipeline.config
        backend = (cfg.parsing or {}).get("backend", "mineru")
        missing: list[str] = []
        if backend == "mineru" and not (cfg.mineru or {}).get("authorization"):
            missing.append("MINERU_AUTHORIZATION (PDF 解析将 401 → chunk=0)")
        if backend == "uniparser" and not (cfg.uniparser or {}).get("api_key"):
            missing.append("UNIPARSER_API_KEY (PDF 解析将失败)")
        if not (cfg.embedding or {}).get("api_key"):
            missing.append("embedding.api_key/VLLM_API_KEY (向量化可能 401 → 无法入库)")
        if not (cfg.generation or {}).get("api_key"):
            missing.append("generation.api_key/VLLM_API_KEY (生成/摘要可能 401)")
        if missing:
            logger.error(
                "[startup] 检测到关键密钥为空, 上传/检索很可能静默失败: %s; "
                "若用本地服务请确认已 `set -a; source .env.local; set +a` 再启动。",
                "; ".join(missing),
            )
    except Exception as e:
        logger.warning(f"[startup] 密钥自检异常 (忽略): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """启动时创建 Pipeline 单例, 关闭时清理资源。"""
    from ..pipeline import Pipeline

    # 注册 SessionLogHandler: 按 session_id 收集 pipeline 检索流程日志
    session_log_handler = get_session_log_handler()
    session_log_handler.setLevel(logging.INFO)
    logging.root.addHandler(session_log_handler)
    # 关键: 确保 pipeline.* 的 INFO 日志真正被创建并向 root 传播。
    # 否则当以 `uvicorn pipeline.api.app:app` 直接启动时 (未经 run_api.py 的
    # basicConfig), root 默认 WARNING 级别会在 isEnabledFor 处把 INFO 记录丢弃,
    # SessionLogHandler 永远收不到 → 检索日志全空。
    logging.getLogger("pipeline").setLevel(logging.INFO)

    config_path = os.environ.get("CONFIG_PATH")
    pipeline = Pipeline(config_path=config_path or None)
    _startup_secret_check(pipeline)

    # 多轮对话保留轮数: 与算法侧 generation.max_history_turns 一致
    gen_cfg = getattr(pipeline, "config", None) and pipeline.config.generation or {}
    max_turns = int(gen_cfg.get("max_history_turns", 5))
    persist_dir = os.environ.get("SESSION_DIR", "logs/sessions")

    session_store = SessionStore(
        ttl=int(os.environ.get("SESSION_TTL", "1800")),
        default_max_turns=max_turns,
        persist_dir=persist_dir,
    )
    task_store = TaskStore(
        max_workers=int(os.environ.get("TASK_MAX_WORKERS", "2")),
        persist_dir=os.environ.get("TASK_DIR", "logs/tasks"),
    )

    # API Keys: 逗号分隔, 空则不鉴权
    raw_keys = os.environ.get("API_KEYS", "")
    api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()] or None

    init_dependencies(pipeline, session_store, task_store, api_keys)
    app.state.pipeline = pipeline
    app.state.session_store = session_store
    app.state.task_store = task_store

    logger.info("[api] Pipeline 初始化完成")
    yield

    session_store.shutdown()
    task_store.shutdown()
    # 关闭查询专用线程池 (取消未开始的排队查询)
    from .concurrency import shutdown_query_executor
    shutdown_query_executor()
    # 移除 SessionLogHandler, 避免重复注册
    logging.root.removeHandler(session_log_handler)
    logger.info("[api] 资源已释放")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DP-RAG Pipeline API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS: 前端跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    from .routers import admin, chat, collections, files, ingest, query, skills
    app.include_router(query.router, prefix="/api/v1", tags=["查询"])
    app.include_router(chat.router, prefix="/api/v1", tags=["对话"])
    app.include_router(ingest.router, prefix="/api/v1", tags=["灌入"])
    app.include_router(files.router, prefix="/api/v1", tags=["文件"])
    app.include_router(collections.router, prefix="/api/v1", tags=["知识库"])
    app.include_router(skills.router, prefix="/api/v1", tags=["专家技能"])
    app.include_router(admin.router, prefix="/api/v1", tags=["运维"])

    return app


app = create_app()
