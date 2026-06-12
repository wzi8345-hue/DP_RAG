"""启动前依赖自检 + 幂等初始化。

后端 API / worker 启动时立即检查关键资源是否可达，并执行幂等初始化：

- Postgres：连通性 (``SELECT 1``) + 建表 (``init_db``)
- Redis：``PING``
- 对象存储 (RustFS / S3)：连通性 + 建 bucket (``ensure_bucket``)
- Milvus：实际探测 (``pipeline.stats()``，需传入 pipeline)

带重试：``docker compose`` 的 ``depends_on`` 只保证 postgres/redis ``healthy``，
milvus/rustfs 仅 ``service_started``，因此需要重试等待其真正就绪。

某个「必需」资源在重试后仍不可达则抛出 ``RuntimeError``——调用方 (API lifespan /
worker) 据此 fail-fast 退出，由容器编排重启并在日志中暴露明确原因，避免「进程活着但
连不上依赖」的假健康状态。

环境变量：
- PREFLIGHT_DISABLED         1/true 跳过全部自检 (仅本地调试)
- PREFLIGHT_ATTEMPTS         每个资源最大重试次数 (默认 30)
- PREFLIGHT_DELAY_S          重试间隔秒 (默认 2)

放在 ``pipeline`` 顶层 (而非 ``pipeline.api``) 以免 worker 引入整个 FastAPI 栈。
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def disabled() -> bool:
    return os.environ.get("PREFLIGHT_DISABLED", "0").strip().lower() in ("1", "true", "yes")


def _attempts() -> int:
    try:
        return max(1, int(os.environ.get("PREFLIGHT_ATTEMPTS", "30")))
    except ValueError:
        return 30


def _delay() -> float:
    try:
        return max(0.0, float(os.environ.get("PREFLIGHT_DELAY_S", "2")))
    except ValueError:
        return 2.0


def _retry(name: str, fn: Callable[[], None], *, attempts: int, delay: float) -> None:
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            fn()
            logger.info("[preflight] %s 就绪", name)
            return
        except Exception as e:  # noqa: BLE001 - 连通性探测对任意异常都需重试
            last = e
            logger.warning("[preflight] %s 未就绪 (%d/%d): %s", name, i, attempts, e)
            if i < attempts:
                time.sleep(delay)
    raise RuntimeError(f"{name} 在 {attempts} 次重试后仍不可用: {last}") from last


# --- 单项检查 -------------------------------------------------------------

def check_postgres(*, init: bool = True) -> None:
    from . import db

    with db.cursor() as cur:
        cur.execute("SELECT 1")
    if init:
        db.init_db()


def check_redis() -> None:
    from .clients import redis as redis_runtime

    redis_runtime.get_redis_runtime().client.ping()


def check_object_store(*, init: bool = True) -> None:
    from .clients import object_store

    client = object_store.get_object_store()
    if init:
        client.ensure_bucket()
    else:
        client.object_exists("__preflight_probe__")


def check_milvus(pipeline) -> None:
    pipeline.stats()


def _resolve_required(require: bool | None, configured: bool, name: str) -> bool:
    """显式 require=True 但未配置 → 立即报错；require=None → 配置了才检查。"""
    if require is None:
        return configured
    if require and not configured:
        raise RuntimeError(f"{name} 被标记为必需，但未配置对应环境变量")
    return require


def run_dependency_checks(
    *,
    pipeline=None,
    require_db: bool | None = None,
    require_redis: bool | None = None,
    require_object_store: bool | None = None,
) -> None:
    """启动前自检全部已配置/必需资源并执行初始化。

    传入 ``pipeline`` 时额外探测 Milvus（需要 pipeline 实例）。
    """
    if disabled():
        logger.warning("[preflight] PREFLIGHT_DISABLED 已启用，跳过依赖自检")
        return

    from . import db
    from .clients import object_store
    from .clients import redis as redis_runtime

    attempts, delay = _attempts(), _delay()

    need_db = _resolve_required(require_db, db.configured(), "DATABASE_URL")
    need_redis = _resolve_required(require_redis, redis_runtime.configured(), "REDIS_URL")
    need_object_store = _resolve_required(require_object_store, object_store.configured(), "对象存储")

    if need_db:
        _retry("Postgres", lambda: check_postgres(init=True), attempts=attempts, delay=delay)
    else:
        logger.warning("[preflight] 未配置 DATABASE_URL，跳过 Postgres 检查")

    if need_redis:
        _retry("Redis", check_redis, attempts=attempts, delay=delay)
    else:
        logger.warning("[preflight] 未配置 REDIS_URL，跳过 Redis 检查")

    if need_object_store:
        _retry("对象存储", lambda: check_object_store(init=True), attempts=attempts, delay=delay)
    else:
        logger.warning("[preflight] 未配置对象存储，跳过 RustFS/S3 检查")

    if pipeline is not None:
        _retry("Milvus", lambda: check_milvus(pipeline), attempts=attempts, delay=delay)

    logger.info("[preflight] 依赖自检完成")
