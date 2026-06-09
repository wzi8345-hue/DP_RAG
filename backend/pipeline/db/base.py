"""psycopg 3 连接池 + 表结构初始化。

不使用 SQLAlchemy/ORM：直接用 psycopg 以便灵活使用 Postgres 高级查询。
连接默认 dict_row（行 → dict），便于喂给 pydantic 模型 model_validate。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg import Connection, Cursor

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

_pool = None  # psycopg_pool.ConnectionPool（惰性创建）


def configured() -> bool:
    """是否配置了数据库连接。"""
    return bool(DATABASE_URL)


def get_pool():
    """返回（惰性创建的）连接池。"""
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL 未配置")
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        _pool = ConnectionPool(
            DATABASE_URL,
            min_size=int(os.environ.get("DB_POOL_MIN", "1")),
            max_size=int(os.environ.get("DB_POOL_MAX", "10")),
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


@contextmanager
def connection() -> Iterator[Connection]:
    """获取一个连接（上下文退出时自动 commit/rollback）。"""
    with get_pool().connection() as conn:
        yield conn


@contextmanager
def cursor() -> Iterator[Cursor]:
    """获取一个游标（自动提交）。"""
    with connection() as conn, conn.cursor() as cur:
        yield cur


def init_db() -> None:
    """检查/初始化表结构（幂等，逐条执行 DDL）。"""
    from .schema import SCHEMA_STATEMENTS

    with connection() as conn:
        with conn.cursor() as cur:
            for stmt in SCHEMA_STATEMENTS:
                cur.execute(stmt)
        conn.commit()
    logger.info("[db] 表结构检查/初始化完成（%d 条 DDL）", len(SCHEMA_STATEMENTS))


def close_pool() -> None:
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        finally:
            _pool = None
