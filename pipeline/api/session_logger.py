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
import json
import logging
import os
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
# 落盘后最多保留多少个 session 文件 (按修改时间淘汰最旧的)。
MAX_PERSISTED_FILES = int(os.environ.get("RETRIEVAL_LOG_MAX_FILES", "500"))
# emit() 内进度落盘最小间隔 (秒), 避免每行都重写整文件。
_PERSIST_THROTTLE_S = 1.0

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
    """请求结束时调用, 清除上下文变量, 并把该 session 日志最终落盘一次。"""
    sid = _session_id_var.get()
    if sid and _global_handler is not None:
        try:
            _global_handler.flush_session(sid)
        except Exception:
            pass
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
        persist_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_sessions = max_sessions
        self.max_lines = max_lines
        self._sessions: OrderedDict[str, SessionLog] = OrderedDict()
        self._lock = threading.Lock()
        # 检索日志落盘目录: 纯内存的检索 trace 重启即丢 (LogViewer 变空),
        # 落盘后重启可回读最近若干 session, 排障/复盘不再断档。
        self._persist_dir = persist_dir
        # 每个 session 上次落盘时间, 用于节流 emit 内的高频写。
        self._last_persist: dict[str, float] = {}
        # 日志格式: 只取时间+级别+logger名, 正文由 handler 自己拼接
        self.formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        if persist_dir:
            try:
                os.makedirs(persist_dir, exist_ok=True)
                self._load_persisted()
            except Exception as e:
                logger.warning(f"[session-log] 初始化落盘目录失败 (仅内存): {e}")

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
                    # 超出上限时淘汰最不活跃的 session (仅内存; 磁盘文件另由
                    # _prune_persisted 按文件数淘汰, 故重启仍可回读历史)。
                    if len(self._sessions) > self.max_sessions:
                        # 移除 updated_at 最早的
                        oldest_key = next(iter(self._sessions))
                        self._sessions.pop(oldest_key)
                        self._last_persist.pop(oldest_key, None)
                    if self._persist_dir:
                        self._prune_persisted()
                else:
                    # 移到末尾 (LRU touch)
                    self._sessions.move_to_end(session_id)

                slog.append(line)

                # 节流落盘: 首行或距上次 >_PERSIST_THROTTLE_S 才重写文件,
                # 请求结束时 clear_session_log_context() 会再 flush 一次最终态。
                if self._persist_dir:
                    now = line.ts or time.time()
                    last = self._last_persist.get(session_id, 0.0)
                    if now - last >= _PERSIST_THROTTLE_S:
                        self._last_persist[session_id] = now
                        self._write_session_file(slog)
        except Exception:
            # handler 内部异常绝对不能影响业务逻辑
            self.handleError(record)

    # ── 落盘 / 回读 ───────────────────────────────────────────────────

    def _session_file_path(self, session_id: str) -> Optional[str]:
        if not self._persist_dir:
            return None
        # session_id 是 hex/8-16 位, 直接用作文件名 (无路径分隔风险)。
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
        if not safe:
            return None
        return os.path.join(self._persist_dir, f"{safe}.json")

    def _write_session_file(self, slog: "SessionLog") -> None:
        """原子写一个 session 的全部日志到磁盘 (调用方持有 self._lock)。"""
        path = self._session_file_path(slog.session_id)
        if not path:
            return
        payload = {
            "session_id": slog.session_id,
            "query": slog.query,
            "created_at": slog.created_at,
            "updated_at": slog.updated_at,
            "lines": [
                {
                    "ts": ln.ts,
                    "timestamp": ln.timestamp,
                    "level": ln.level,
                    "logger_name": ln.logger_name,
                    "message": ln.message,
                }
                for ln in slog.lines
            ],
        }
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning(f"[session-log] 落盘失败 {slog.session_id}: {e}")

    def flush_session(self, session_id: str) -> None:
        """把指定 session 的当前日志最终落盘 (请求结束时调用)。"""
        if not self._persist_dir:
            return
        with self._lock:
            slog = self._sessions.get(session_id)
            if slog is not None:
                self._last_persist[session_id] = time.time()
                self._write_session_file(slog)

    def _prune_persisted(self) -> None:
        """按文件数上限淘汰最旧的落盘文件 (调用方持有 self._lock)。"""
        if not self._persist_dir:
            return
        try:
            files = [
                os.path.join(self._persist_dir, fn)
                for fn in os.listdir(self._persist_dir)
                if fn.endswith(".json")
            ]
            if len(files) <= MAX_PERSISTED_FILES:
                return
            files.sort(key=lambda p: os.path.getmtime(p))
            for p in files[: len(files) - MAX_PERSISTED_FILES]:
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[session-log] 清理落盘文件失败: {e}")

    def _load_persisted(self) -> None:
        """启动时回读最近的 session 日志 (按修改时间取最新 max_sessions 个)。"""
        if not self._persist_dir or not os.path.isdir(self._persist_dir):
            return
        try:
            files = [
                os.path.join(self._persist_dir, fn)
                for fn in os.listdir(self._persist_dir)
                if fn.endswith(".json")
            ]
        except Exception:
            return
        # 最新的在后 (与 OrderedDict LRU 一致: 末尾最活跃)
        files.sort(key=lambda p: os.path.getmtime(p))
        loaded = 0
        for path in files[-self.max_sessions:]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                slog = SessionLog(
                    session_id=data.get("session_id", ""),
                    query=data.get("query", ""),
                    created_at=data.get("created_at", time.time()),
                    updated_at=data.get("updated_at", time.time()),
                )
                for ln in data.get("lines", []):
                    slog.lines.append(LogLine(
                        timestamp=ln.get("timestamp", ""),
                        level=ln.get("level", "INFO"),
                        logger_name=ln.get("logger_name", ""),
                        message=ln.get("message", ""),
                        ts=ln.get("ts", 0.0),
                    ))
                if slog.session_id:
                    self._sessions[slog.session_id] = slog
                    loaded += 1
            except Exception as e:
                logger.warning(f"[session-log] 回读 {path} 失败, 跳过: {e}")
        if loaded:
            logger.info(f"[session-log] 回读 {loaded} 个历史 session 检索日志")

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
        _global_handler = SessionLogHandler(
            persist_dir=os.environ.get("RETRIEVAL_LOG_DIR", "logs/retrieval"),
        )
    return _global_handler
