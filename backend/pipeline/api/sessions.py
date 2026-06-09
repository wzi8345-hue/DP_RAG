"""进程内会话管理: ChatSession 存储 + TTL 清理 + 历史持久化。"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ..flows.query import ChatSession, ChatTurn

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 1800       # 30 分钟无活动自动清理
_CLEANUP_INTERVAL = 300   # 每 5 分钟清理一次


class SessionStore:
    """线程安全的会话存储, 带后台 TTL 清理 + JSON 历史落盘。

    - 内存中 ChatSession 仅保留最近 max_turns 轮 (供多轮检索/生成上下文)。
    - 完整历史 (所有轮次, 仅 user/assistant) 以 OpenAI messages 格式持久化到
      ``persist_dir/<session_id>.json``, 不受内存截断影响。
    """

    def __init__(
        self,
        ttl: int = _DEFAULT_TTL,
        default_max_turns: int = 5,
        persist_dir: Optional[str] = None,
    ) -> None:
        self._store: Dict[str, _SessionEntry] = {}
        self._lock = threading.Lock()
        self._ttl = ttl
        self._default_max_turns = default_max_turns
        self._persist_dir = persist_dir
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True,
        )
        self._cleanup_thread.start()

    def create(self, max_turns: Optional[int] = None) -> str:
        """创建新会话, 返回 session_id。"""
        import uuid
        sid = uuid.uuid4().hex[:16]
        turns = self._default_max_turns if max_turns is None else max_turns
        session = ChatSession(max_turns=turns)
        with self._lock:
            self._store[sid] = _SessionEntry(
                session=session, last_access=time.time(),
            )
        return sid

    # ── 历史持久化 (OpenAI messages 格式) ────────────────────────────────

    def _session_path(self, session_id: str) -> Optional[str]:
        if not self._persist_dir:
            return None
        return os.path.join(self._persist_dir, f"{session_id}.json")

    def append_messages(
        self, session_id: str, user_text: str, assistant_text: str,
    ) -> None:
        """把一轮 (user + assistant) 追加到该 session 的完整历史文件。

        文件格式:
            {"session_id", "created_at", "updated_at",
             "messages": [{"role": "user"|"assistant", "content": ...}, ...]}
        """
        if not self._persist_dir:
            return
        path = self._session_path(session_id)
        if not path:
            return
        now = time.time()
        with self._lock:
            entry = self._store.get(session_id)
            full = entry.all_messages if entry else []
            full.append({"role": "user", "content": user_text})
            full.append({"role": "assistant", "content": assistant_text})
            created = entry.created_at if entry else now
            payload = {
                "session_id": session_id,
                "created_at": created,
                "updated_at": now,
                "messages": full,
            }
            try:
                tmp = f"{path}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)
            except Exception as e:
                logger.warning(f"[sessions] 持久化失败 {session_id}: {e}")

    def get(self, session_id: str) -> Optional[ChatSession]:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            entry.last_access = time.time()
            return entry.session

    def update(self, session_id: str, session: ChatSession) -> None:
        with self._lock:
            if session_id in self._store:
                self._store[session_id].session = session
                self._store[session_id].last_access = time.time()
            else:
                self._store[session_id] = _SessionEntry(
                    session=session, last_access=time.time(),
                )

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._store.pop(session_id, None) is not None

    def session_to_dict(self, session: ChatSession) -> Dict[str, Any]:
        return {
            "turns": [asdict(t) for t in session.turns],
            "max_turns": session.max_turns,
        }

    def dict_to_session(self, data: Dict[str, Any]) -> ChatSession:
        turns = [ChatTurn(**t) for t in data.get("turns", [])]
        return ChatSession(turns=turns, max_turns=data.get("max_turns", 5))

    def shutdown(self) -> None:
        self._stop_event.set()

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(_CLEANUP_INTERVAL):
            self._cleanup()

    def _cleanup(self) -> None:
        now = time.time()
        expired = []
        with self._lock:
            for sid, entry in self._store.items():
                if now - entry.last_access > self._ttl:
                    expired.append(sid)
            for sid in expired:
                del self._store[sid]
        if expired:
            logger.info(f"[sessions] 清理 {len(expired)} 个过期会话")


class _SessionEntry:
    __slots__ = ("session", "last_access", "created_at", "all_messages")

    def __init__(self, session: ChatSession, last_access: float) -> None:
        self.session = session
        self.last_access = last_access
        self.created_at = last_access
        # 完整历史 (不截断), OpenAI messages 格式
        self.all_messages: List[Dict[str, Any]] = []
