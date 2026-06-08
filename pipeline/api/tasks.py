"""异步任务管理: 后台线程池 + 进程内状态。"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_TASK_TTL = 86400          # 24 小时后清理
_CLEANUP_INTERVAL = 600    # 10 分钟清理一次
_MAX_WORKERS = 2           # 同时最多跑 2 个灌入任务


@dataclass
class TaskStatus:
    id: str
    status: str = "pending"  # pending / running / done / failed
    progress: float = 0.0
    current: int = 0
    total: int = 0
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class TaskStore:
    """线程安全的任务存储, 带后台线程池和 TTL 清理。"""

    def __init__(self, max_workers: int = _MAX_WORKERS) -> None:
        self._tasks: Dict[str, TaskStatus] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True,
        )
        self._cleanup_thread.start()

    def submit(
        self,
        fn: Callable,
        *args: Any,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """提交任务到线程池, 返回 task_id。"""
        import uuid
        tid = task_id or uuid.uuid4().hex[:16]
        status = TaskStatus(id=tid, status="pending")
        with self._lock:
            self._tasks[tid] = status

        def _run():
            with self._lock:
                self._tasks[tid].status = "running"
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._tasks[tid].status = "done"
                    self._tasks[tid].result = result
                    self._tasks[tid].progress = 1.0
            except Exception as e:
                logger.exception(f"[task {tid}] 执行失败")
                with self._lock:
                    self._tasks[tid].status = "failed"
                    self._tasks[tid].error = str(e)

        self._executor.submit(_run)
        return tid

    def get(self, task_id: str) -> Optional[TaskStatus]:
        with self._lock:
            return self._tasks.get(task_id)

    def update_progress(
        self, task_id: str, current: int, total: int, doc_id: str = "",
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.current = current
                task.total = total
                task.progress = current / max(total, 1)

    def shutdown(self) -> None:
        self._stop_event.set()
        self._executor.shutdown(wait=False)

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(_CLEANUP_INTERVAL):
            self._cleanup()

    def _cleanup(self) -> None:
        now = time.time()
        expired = []
        with self._lock:
            for tid, task in self._tasks.items():
                if task.status in ("done", "failed") and now - task.created_at > _TASK_TTL:
                    expired.append(tid)
            for tid in expired:
                del self._tasks[tid]
        if expired:
            logger.info(f"[tasks] 清理 {len(expired)} 个过期任务")
