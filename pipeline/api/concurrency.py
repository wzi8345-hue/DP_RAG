"""查询/对话请求的并发隔离 + 单请求总超时。

背景: ``/query`` 与 ``/chat`` 原本走 ``asyncio.to_thread`` —— 即事件循环的**默认**
线程池, 与框架内其它 ``to_thread`` 调用共享。一条慢查询 (langgraph 反思重试 +
fallback 回退 + reranker + 多轮 research, 每段都有自己的 LLM 调用) 能挂很久;
一波并发慢查询会把默认池打满, 拖垮整个 API。

这里给查询类请求**单独一个有界线程池** + **单请求总死线**:
- 有界池把查询负载与其余请求隔离, 一波慢查询最多占满本池而不波及框架其它路径。
- 死线保证慢查询到点返回明确错误, 不无限期挂起。

注意: Python 线程无法强杀, 超时只是让调用方尽快拿到结果; 底层线程会继续运行,
直到下游 client 各自超时 (LLM/embedding 默认 120s) 后结束并释放槽位。因此 timeout
应 >= 一条正常链路的最长耗时, 池大小也别太小, 避免超时线程长期占槽。
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 单请求总超时 (秒); 默认略大于下游 LLM 单次 120s, 给一次重试留点余量。
_QUERY_TIMEOUT_S = float(os.environ.get("QUERY_TIMEOUT_S", "150"))
# 查询并发上限 (= 专用线程池大小); 超过的请求排队等待空闲线程。
_QUERY_MAX_CONCURRENCY = max(1, int(os.environ.get("QUERY_MAX_CONCURRENCY", "8")))

_executor: Optional[ThreadPoolExecutor] = None


def get_query_executor() -> ThreadPoolExecutor:
    """惰性创建查询专用有界线程池 (进程内单例)。"""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_QUERY_MAX_CONCURRENCY,
            thread_name_prefix="query",
        )
    return _executor


def query_timeout_s() -> float:
    return _QUERY_TIMEOUT_S


async def run_query_with_timeout(
    fn: Callable[..., T], *args: Any, timeout: Optional[float] = None, **kwargs: Any,
) -> T:
    """在查询专用线程池里执行 ``fn``, 超过 ``timeout`` 抛 ``asyncio.TimeoutError``。"""
    loop = asyncio.get_running_loop()
    executor = get_query_executor()
    fut = loop.run_in_executor(executor, lambda: fn(*args, **kwargs))
    return await asyncio.wait_for(
        fut, timeout=timeout if timeout is not None else _QUERY_TIMEOUT_S
    )


def shutdown_query_executor() -> None:
    """关闭查询线程池 (取消未开始的排队任务, 不等在途线程)。"""
    global _executor
    if _executor is not None:
        try:
            _executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # 老版本 Python 无 cancel_futures 参数
            _executor.shutdown(wait=False)
        _executor = None
