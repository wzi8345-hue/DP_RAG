"""Redis runtime helpers for generation run queue and live event streams."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RedisSettings:
    url: str
    queue_name: str = "dprag:generation:runs"
    stream_prefix: str = "dprag:generation:stream"
    stream_maxlen: int = 2000
    stream_ttl_s: int = 86400
    ingest_queue_name: str = "dprag:ingest:tasks"
    ingest_stream_prefix: str = "dprag:ingest:stream"
    ingest_stream_maxlen: int = 2000
    ingest_stream_ttl_s: int = 86400


def load_settings() -> RedisSettings | None:
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        return None
    return RedisSettings(
        url=url,
        queue_name=os.environ.get("REDIS_RUN_QUEUE", "dprag:generation:runs").strip()
        or "dprag:generation:runs",
        stream_prefix=os.environ.get("REDIS_RUN_STREAM_PREFIX", "dprag:generation:stream").strip()
        or "dprag:generation:stream",
        stream_maxlen=int(os.environ.get("REDIS_RUN_STREAM_MAXLEN", "2000")),
        stream_ttl_s=int(os.environ.get("REDIS_RUN_STREAM_TTL", "86400")),
        ingest_queue_name=os.environ.get("REDIS_INGEST_QUEUE", "dprag:ingest:tasks").strip()
        or "dprag:ingest:tasks",
        ingest_stream_prefix=os.environ.get("REDIS_INGEST_STREAM_PREFIX", "dprag:ingest:stream").strip()
        or "dprag:ingest:stream",
        ingest_stream_maxlen=int(os.environ.get("REDIS_INGEST_STREAM_MAXLEN", "2000")),
        ingest_stream_ttl_s=int(os.environ.get("REDIS_INGEST_STREAM_TTL", "86400")),
    )


class RedisRuntime:
    def __init__(self, settings: RedisSettings | None = None) -> None:
        settings = settings or load_settings()
        if settings is None:
            raise RuntimeError("REDIS_URL is not configured")
        self.settings = settings
        from redis import Redis

        # 注意: 不设 socket_timeout (保持 None), 否则会与阻塞命令 (BLPOP timeout=5) 冲突,
        # 导致 "Timeout reading from socket"。仅设连接超时 + keepalive + 周期健康检查 + 自动重试,
        # 以便快速发现断连并在 redis 抖动/重启后自愈。
        self.client = Redis.from_url(
            settings.url,
            decode_responses=True,
            socket_connect_timeout=int(os.environ.get("REDIS_SOCKET_CONNECT_TIMEOUT", "5")),
            socket_keepalive=True,
            health_check_interval=int(os.environ.get("REDIS_HEALTH_CHECK_INTERVAL", "30")),
            retry_on_timeout=True,
        )

    def stream_key(self, run_id: str) -> str:
        return f"{self.settings.stream_prefix}:{run_id}"

    def ingest_stream_key(self, task_id: str) -> str:
        return f"{self.settings.ingest_stream_prefix}:{task_id}"

    def enqueue_run(self, run_id: str) -> None:
        self.client.rpush(self.settings.queue_name, run_id)

    def dequeue_run(self, *, timeout_s: int = 5) -> str | None:
        item = self.client.blpop(self.settings.queue_name, timeout=timeout_s)
        if not item:
            return None
        _, run_id = item
        return str(run_id)

    def enqueue_ingest_task(self, task_id: str) -> None:
        self.client.rpush(self.settings.ingest_queue_name, task_id)

    def dequeue_ingest_task(self, *, timeout_s: int = 5) -> str | None:
        item = self.client.blpop(self.settings.ingest_queue_name, timeout=timeout_s)
        if not item:
            return None
        _, task_id = item
        return str(task_id)

    def publish_event(self, run_id: str, event: dict[str, Any]) -> str:
        key = self.stream_key(run_id)
        entry_id = self.client.xadd(
            key,
            {"event": json.dumps(event, ensure_ascii=False)},
            maxlen=self.settings.stream_maxlen,
            approximate=True,
        )
        self.client.expire(key, self.settings.stream_ttl_s)
        return str(entry_id)

    def publish_ingest_event(self, task_id: str, event: dict[str, Any]) -> str:
        key = self.ingest_stream_key(task_id)
        entry_id = self.client.xadd(
            key,
            {"event": json.dumps(event, ensure_ascii=False)},
            maxlen=self.settings.ingest_stream_maxlen,
            approximate=True,
        )
        self.client.expire(key, self.settings.ingest_stream_ttl_s)
        return str(entry_id)

    def read_events(
        self,
        run_id: str,
        *,
        last_id: str = "$",
        block_ms: int = 1000,
        count: int = 100,
    ) -> list[tuple[str, dict[str, Any]]]:
        rows = self.client.xread(
            {self.stream_key(run_id): last_id},
            block=block_ms,
            count=count,
        )
        out: list[tuple[str, dict[str, Any]]] = []
        for _, entries in rows:
            for entry_id, fields in entries:
                raw = fields.get("event") if isinstance(fields, dict) else None
                if not raw:
                    continue
                try:
                    out.append((str(entry_id), json.loads(raw)))
                except json.JSONDecodeError:
                    continue
        return out

    def read_ingest_events(
        self,
        task_id: str,
        *,
        last_id: str = "$",
        block_ms: int = 1000,
        count: int = 100,
    ) -> list[tuple[str, dict[str, Any]]]:
        rows = self.client.xread(
            {self.ingest_stream_key(task_id): last_id},
            block=block_ms,
            count=count,
        )
        out: list[tuple[str, dict[str, Any]]] = []
        for _, entries in rows:
            for entry_id, fields in entries:
                raw = fields.get("event") if isinstance(fields, dict) else None
                if not raw:
                    continue
                try:
                    out.append((str(entry_id), json.loads(raw)))
                except json.JSONDecodeError:
                    continue
        return out


_runtime: RedisRuntime | None = None


def configured() -> bool:
    return load_settings() is not None


def get_redis_runtime() -> RedisRuntime:
    global _runtime
    if _runtime is None:
        _runtime = RedisRuntime()
    return _runtime
