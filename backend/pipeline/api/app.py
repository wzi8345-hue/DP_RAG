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

    # 数据库: 检查/初始化表结构 (幂等)。未配置 DATABASE_URL 时跳过 (过渡期允许无库运行);
    # 配置了但连接失败则记录错误并继续, 不阻塞 RAG 主功能。
    from .. import db
    if db.configured():
        try:
            db.init_db()
        except Exception as e:
            logger.error("[db] 初始化失败 (继续启动): %s", e)
    else:
        logger.warning("[db] 未配置 DATABASE_URL, 跳过表结构初始化")

    config_path = os.environ.get("CONFIG_PATH")
    pipeline = Pipeline(config_path=config_path or None)

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
    try:
        db.close_pool()
    except Exception:
        pass
    # 移除 SessionLogHandler, 避免重复注册
    logging.root.removeHandler(session_log_handler)
    logger.info("[api] 资源已释放")


def create_app() -> FastAPI:
    # 反代前缀 (如 funmg.dp.tech/sci-loop-api 未剥离前缀时, 设 API_ROOT_PATH=/sci-loop-api)
    root_path = os.environ.get("API_ROOT_PATH", "").rstrip("/")
    app = FastAPI(
        title="DP-RAG Pipeline API",
        version="2.0.0",
        lifespan=lifespan,
        root_path=root_path or "",
    )

    # CORS: 前端跨域。带凭证时不能用 "*"; 显式来源才允许 credentials。
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
    allow_credentials = origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=allow_credentials,
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
