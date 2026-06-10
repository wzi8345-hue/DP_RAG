"""请求上下文中间件: 为每个 HTTP 请求注入全局 request_id + 记录出入口日志。

设计为**纯 ASGI 中间件** (而非 Starlette BaseHTTPMiddleware): 后者会包裹/缓冲
响应体, 破坏 ``/chat/stream`` 的 SSE 流式输出。纯 ASGI 只在 response.start 阶段
读状态码、注入响应头, 不触碰 body 流。

request_id 存入 contextvar, 供后续结构化日志阶段串联全链路 (当前阶段先用于
出入口日志行 + 响应头 X-Request-ID); 若入站已带 X-Request-ID 则透传复用。
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from typing import Optional

logger = logging.getLogger("pipeline.api.request")

_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> Optional[str]:
    """返回当前请求的 request_id (无上下文时为 None)。"""
    return _request_id_var.get()


class RequestContextMiddleware:
    """纯 ASGI 中间件: 注入 request_id, 记录请求出入口与耗时。"""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        # 入站若带 X-Request-ID 则复用 (便于跨服务串联), 否则新生成。
        incoming = None
        for k, v in scope.get("headers", []):
            if k == b"x-request-id":
                incoming = v.decode("latin-1").strip() or None
                break
        rid = incoming or f"req_{uuid.uuid4().hex[:12]}"
        token = _request_id_var.set(rid)

        method = scope.get("method", "")
        path = scope.get("path", "")
        start = time.monotonic()
        status_holder = {"code": 0}

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                status_holder["code"] = message.get("status", 0)
                headers = message.setdefault("headers", [])
                headers.append((b"x-request-id", rid.encode("latin-1")))
            await send(message)

        logger.info(f"[req] {rid} -> {method} {path}")
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            dur_ms = (time.monotonic() - start) * 1000
            logger.exception(
                f"[req] {rid} !! {method} {path} 未处理异常 ({dur_ms:.0f}ms)"
            )
            raise
        finally:
            dur_ms = (time.monotonic() - start) * 1000
            logger.info(
                f"[req] {rid} <- {status_holder['code']} {method} {path} "
                f"({dur_ms:.0f}ms)"
            )
            _request_id_var.reset(token)
