"""异步任务管理: 后台线程池 + 进程内状态 + JSON 落盘。

落盘动机: 任务状态原本只在进程内存里, 后端重启 (含 launchd 守护拉起) 会丢光,
正在跑的灌入任务被掐断、状态全无, 前端轮询直接 404, 体验是"任务凭空消失"。
现在每次状态变化写盘 (原子替换), 重启时回读; 重启前处于 pending/running 的任务
标记为 interrupted (终态), 让前端拿到明确结论 (数据已落盘, 重新上传会自动续灌)。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_TASK_TTL = 86400          # 24 小时后清理
_CLEANUP_INTERVAL = 600    # 10 分钟清理一次
_MAX_WORKERS = 2           # 同时最多跑 2 个灌入任务
_PROGRESS_PERSIST_INTERVAL = 2.0  # 进度落盘最小间隔 (秒), 避免高频小写


@dataclass
class TaskStatus:
    id: str
    status: str = "pending"  # pending / running / done / failed / interrupted
    progress: float = 0.0
    current: int = 0
    total: int = 0
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class TaskStore:
    """线程安全的任务存储, 带后台线程池、TTL 清理与 JSON 落盘。"""

    def __init__(
        self,
        max_workers: int = _MAX_WORKERS,
        persist_dir: Optional[str] = None,
    ) -> None:
        self._tasks: Dict[str, TaskStatus] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._stop_event = threading.Event()
        self._persist_dir = persist_dir
        # 每个任务上次进度落盘的时间, 用于节流高频 progress 写盘。
        self._last_progress_persist: Dict[str, float] = {}
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._load_persisted()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True,
        )
        self._cleanup_thread.start()

    # ── 落盘 / 回读 ──────────────────────────────────────────────────────

    def _task_path(self, tid: str) -> Optional[str]:
        if not self._persist_dir:
            return None
        return os.path.join(self._persist_dir, f"{tid}.json")

    def _persist(self, tid: str) -> None:
        """原子写一个任务的状态到磁盘 (无 persist_dir 时空操作)。调用方持锁。"""
        path = self._task_path(tid)
        if not path:
            return
        task = self._tasks.get(tid)
        if task is None:
            return
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(asdict(task), f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning(f"[tasks] 持久化失败 {tid}: {e}")

    def _delete_persisted(self, tid: str) -> None:
        path = self._task_path(tid)
        if not path:
            return
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"[tasks] 删除持久化文件失败 {tid}: {e}")

    def _load_persisted(self) -> None:
        """启动时回读已落盘任务; 非终态 (pending/running) 标记为 interrupted。"""
        if not self._persist_dir or not os.path.isdir(self._persist_dir):
            return
        loaded = 0
        interrupted = 0
        for fn in os.listdir(self._persist_dir):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(self._persist_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                task = TaskStatus(**{
                    k: data.get(k) for k in TaskStatus.__dataclass_fields__
                    if k in data
                })
            except Exception as e:
                logger.warning(f"[tasks] 回读 {fn} 失败, 跳过: {e}")
                continue
            # 重启前还在跑/排队的任务: 线程已不在, 标记为中断终态。
            if task.status in ("pending", "running"):
                task.status = "interrupted"
                task.error = (
                    "服务重启导致任务中断。已解析/入库的文献不会丢失, "
                    "重新上传同一批文件会自动跳过已入库项、只续灌未完成的。"
                )
                interrupted += 1
            self._tasks[task.id] = task
            loaded += 1
        # 重写被改成 interrupted 的任务, 让磁盘与内存一致。
        if interrupted:
            with self._lock:
                for tid, t in self._tasks.items():
                    if t.status == "interrupted":
                        self._persist(tid)
        if loaded:
            logger.info(
                f"[tasks] 回读 {loaded} 个历史任务 (其中 {interrupted} 个标记为中断)"
            )

    # ── 任务生命周期 ─────────────────────────────────────────────────────

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
            self._persist(tid)

        def _run():
            with self._lock:
                self._tasks[tid].status = "running"
                self._persist(tid)
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._tasks[tid].status = "done"
                    self._tasks[tid].result = result
                    self._tasks[tid].progress = 1.0
                    self._persist(tid)
            except Exception as e:
                logger.exception(f"[task {tid}] 执行失败")
                with self._lock:
                    self._tasks[tid].status = "failed"
                    self._tasks[tid].error = str(e)
                    self._persist(tid)

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
            if not task:
                return
            task.current = current
            task.total = total
            task.progress = current / max(total, 1)
            # 进度落盘节流: 完成 (current>=total) 时强制写, 其余按时间间隔。
            now = time.time()
            last = self._last_progress_persist.get(task_id, 0.0)
            if current >= total or now - last >= _PROGRESS_PERSIST_INTERVAL:
                self._last_progress_persist[task_id] = now
                self._persist(task_id)

    def shutdown(self) -> None:
        self._stop_event.set()
        # 取消尚未开始的排队任务; 在途线程无法强杀, 其状态靠下次启动回读时
        # 标记为 interrupted (它们当前是 running, 磁盘上也是 running)。
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(_CLEANUP_INTERVAL):
            self._cleanup()

    def _cleanup(self) -> None:
        now = time.time()
        expired = []
        with self._lock:
            for tid, task in self._tasks.items():
                if task.status in ("done", "failed", "interrupted") and \
                        now - task.created_at > _TASK_TTL:
                    expired.append(tid)
            for tid in expired:
                del self._tasks[tid]
                self._last_progress_persist.pop(tid, None)
                self._delete_persisted(tid)
        if expired:
            logger.info(f"[tasks] 清理 {len(expired)} 个过期任务")
