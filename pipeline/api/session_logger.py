"""Session-scoped log handler: 按 session_id 收集 pipeline 检索流程日志。

设计要点:
- 自定义 logging.Handler, 每条日志按 session_id 归入内存 ring buffer
- 通过 contextvars.ContextVar 在请求入口设置当前 session_id, handler 自动归类
  (使用 contextvars 而非 threading.local, 因为 asyncio.to_thread 会自动传播
   contextvars 到线程池, 而 threading.local 不会)
- 每个 session 最多保留 MAX_LINES_PER_SESSION 行, 超出自动淘汰旧行
- 最多保留 MAX_SESSIONS 个 session 的日志, LRU 淘汰最不活跃的
- 支持 SSE 订阅: 新日志行实时推送给已连接的 EventSource 客户端
- 零侵入: 不需要修改 pipeline 代码中的 logger 调用
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

MAX_LINES_PER_SESSION = 2000
MAX_SESSIONS = 100

# ---------------------------------------------------------------------------
# ContextVar: 当前请求关联的 session_id
# 使用 contextvars 而非 threading.local:
#   asyncio.to_thread / run_in_executor 会自动 copy context,
#   使 ContextVar 在线程池中仍可读取; threading.local 则不行。
# ---------------------------------------------------------------------------

_session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_log_id", default=None
)
_session_query_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_log_query", default=""
)


def set_session_log_context(session_id: str, query: str = "") -> None:
    """在请求入口调用, 设置当前上下文的 session_id + 首条 query 摘要。"""
    _session_id_var.set(session_id)
    _session_query_var.set(query)


def clear_session_log_context() -> None:
    """请求结束时调用, 清除上下文变量。"""
    _session_id_var.set(None)
    _session_query_var.set("")


def get_current_session_id() -> Optional[str]:
    return _session_id_var.get()


def get_current_session_query() -> str:
    return _session_query_var.get()


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class LogLine:
    """单条日志记录。"""

    timestamp: str  # 格式化后的时间, 如 "15:42:51"
    level: str  # INFO / WARNING / ERROR
    logger_name: str  # pipeline.retrieval.langgraph_agent
    message: str  # 日志正文
    ts: float = 0.0  # 原始时间戳, 供排序 / SSE 推送判断


@dataclass
class SessionLog:
    """单个 session 的日志集合。"""

    session_id: str
    query: str = ""  # 首条查询摘要
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    lines: List[LogLine] = field(default_factory=list)
    # SSE 订阅者队列: 新日志行会被放入每个订阅者的 deque
    _subscribers: List[Deque[LogLine]] = field(default_factory=list)
    _sub_lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, line: LogLine) -> None:
        self.lines.append(line)
        self.updated_at = time.time()
        # 超出上限时淘汰最旧的行
        if len(self.lines) > MAX_LINES_PER_SESSION:
            self.lines = self.lines[-MAX_LINES_PER_SESSION:]
        # 推送给所有 SSE 订阅者
        with self._sub_lock:
            dead: List[Deque[LogLine]] = []
            for q in self._subscribers:
                q.append(line)
                # 订阅者队列过长说明消费太慢, 标记清理
                if len(q) > 500:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    def subscribe(self) -> Deque[LogLine]:
        """注册一个 SSE 订阅者, 返回该订阅者的队列 (新行会被 append 进去)。"""
        q: Deque[LogLine] = deque()
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: Deque[LogLine]) -> None:
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# SessionLogHandler
# ---------------------------------------------------------------------------


class SessionLogHandler(logging.Handler):
    """将标准 logging 输出按 session_id 归类的 handler。

    使用方式:
        handler = SessionLogHandler()
        logging.root.addHandler(handler)

    在请求入口:
        set_session_log_context(session_id, query)
        # ... pipeline 处理 ...
        clear_session_log_context()
    """

    def __init__(
        self,
        max_sessions: int = MAX_SESSIONS,
        max_lines: int = MAX_LINES_PER_SESSION,
    ) -> None:
        super().__init__()
        self.max_sessions = max_sessions
        self.max_lines = max_lines
        self._sessions: OrderedDict[str, SessionLog] = OrderedDict()
        self._lock = threading.Lock()
        # 日志格式: 只取时间+级别+logger名, 正文由 handler 自己拼接
        self.formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            session_id = get_current_session_id()
            if not session_id:
                return  # 不在请求上下文中, 不收集

            # 只收集 pipeline 相关的日志, 跳过 uvicorn / httpx 等噪音
            if not record.name.startswith("pipeline"):
                return

            msg = self.format(record)

            # 从格式化后的消息中提取字段
            # 格式: "15:42:51 INFO [pipeline.xxx] 消息正文"
            parts = msg.split(" ", 3)
            if len(parts) >= 4:
                timestamp = parts[0]
                level = parts[1]
                logger_name = parts[2].strip("[]")
                message = parts[3]
            else:
                timestamp = ""
                level = record.levelname
                logger_name = record.name
                message = msg

            line = LogLine(
                timestamp=timestamp,
                level=level,
                logger_name=logger_name,
                message=message,
                ts=record.created,
            )

            with self._lock:
                slog = self._sessions.get(session_id)
                if slog is None:
                    query = get_current_session_query()
                    slog = SessionLog(session_id=session_id, query=query)
                    self._sessions[session_id] = slog
                    # 超出上限时淘汰最不活跃的 session
                    if len(self._sessions) > self.max_sessions:
                        # 移除 updated_at 最早的
                        oldest_key = next(iter(self._sessions))
                        self._sessions.pop(oldest_key)
                else:
                    # 移到末尾 (LRU touch)
                    self._sessions.move_to_end(session_id)

                slog.append(line)
        except Exception:
            # handler 内部异常绝对不能影响业务逻辑
            self.handleError(record)

    # ── 查询接口 ─────────────────────────────────────────────────────

    def list_sessions(self) -> List[dict]:
        """返回有日志的 session 列表, 按最后更新时间倒序。"""
        with self._lock:
            sessions = sorted(
                self._sessions.values(),
                key=lambda s: -s.updated_at,
            )
            return [
                {
                    "session_id": s.session_id,
                    "query": s.query[:100],
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                    "line_count": len(s.lines),
                }
                for s in sessions
            ]

    def get_session(
        self,
        session_id: str,
        tail: Optional[int] = None,
    ) -> Optional[dict]:
        """返回指定 session 的日志详情。"""
        with self._lock:
            slog = self._sessions.get(session_id)
            if slog is None:
                return None
            lines = slog.lines
            if tail:
                lines = lines[-tail:]
            return {
                "session_id": slog.session_id,
                "query": slog.query[:100],
                "created_at": slog.created_at,
                "updated_at": slog.updated_at,
                "line_count": len(slog.lines),
                "lines": [
                    {
                        "ts": line.ts,
                        "timestamp": line.timestamp,
                        "level": line.level,
                        "logger": line.logger_name,
                        "message": line.message,
                    }
                    for line in lines
                ],
            }

    def subscribe_session(self, session_id: str) -> Deque[LogLine]:
        """注册 SSE 订阅, 返回订阅者队列 (后续新行会被 append 进去)。"""
        with self._lock:
            slog = self._sessions.get(session_id)
            if slog is None:
                # session 还不存在, 先创建一个空的
                slog = SessionLog(session_id=session_id)
                self._sessions[session_id] = slog
        return slog.subscribe()

    def unsubscribe_session(self, session_id: str, q: Deque[LogLine]) -> None:
        with self._lock:
            slog = self._sessions.get(session_id)
            if slog:
                slog.unsubscribe(q)


# ---------------------------------------------------------------------------
# 全局单例 (在 app.py lifespan 中创建并注册)
# ---------------------------------------------------------------------------

_global_handler: Optional[SessionLogHandler] = None


def get_session_log_handler() -> SessionLogHandler:
    global _global_handler
    if _global_handler is None:
        _global_handler = SessionLogHandler()
    return _global_handler
