"""Generation worker service: consumes run jobs from Redis and produces message events."""

from __future__ import annotations

import logging
import os
import signal
import time
from typing import Any

from redis.exceptions import RedisError

from .. import preflight
from ..auth import AuthContext
from ..clients import redis as redis_runtime
from ..db import repo
from ..flows.query import ChatSession
from ..pipeline import Pipeline

logger = logging.getLogger(__name__)

# redis 抖动 / 短暂不可达时的循环重试间隔 (秒)
_REDIS_RETRY_DELAY_S = 2.0


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _publish_event(run_id: str, event_type: str, payload: dict[str, Any]) -> None:
    ev = repo.append_message_event(run_id, event_type, payload)
    live = dict(ev.payload)
    live.setdefault("type", ev.type)
    live["seq"] = ev.seq
    live["run_id"] = ev.run_id
    redis_runtime.get_redis_runtime().publish_event(run_id, live)


def _session_from_parent(parent_message_id: str | None, *, max_turns: int) -> ChatSession:
    session = ChatSession(max_turns=max_turns)
    chain = repo.get_message_chain_to(parent_message_id)
    pending_user: str | None = None
    for msg in chain:
        if msg.role == "user":
            pending_user = msg.content
        elif msg.role == "assistant" and pending_user is not None:
            session.add_turn(pending_user, msg.content)
            pending_user = None
    return session


def _apply_skill_scope(pipe: Pipeline, auth: AuthContext) -> None:
    readable_ids = sorted({s.id for s in repo.list_skill_metadata(auth)})
    prof = (pipe.config.retrieval.get("langgraph", {}) or {}).setdefault("professional", {})
    skills_cfg = prof.setdefault("skills", {})
    if skills_cfg.get("allowed_ids") == readable_ids:
        return
    skills_cfg["allowed_ids"] = readable_ids
    try:
        pipe._get_query_flow().reload_skills()
    except Exception:
        pass


class GenerationWorker:
    def __init__(self) -> None:
        self.pipeline = Pipeline(config_path=os.environ.get("CONFIG_PATH") or None)
        # 启动即自检并初始化依赖 (postgres 必需+建表 / redis 必需 / 对象存储+milvus)
        preflight.run_dependency_checks(
            pipeline=self.pipeline, require_db=True, require_redis=True,
        )
        self.redis = redis_runtime.get_redis_runtime()
        self.stopping = False

    def stop(self, *_args) -> None:
        self.stopping = True

    def run_forever(self) -> None:
        logger.info("[generation-worker] started")
        while not self.stopping:
            try:
                run_id = self.redis.dequeue_run(timeout_s=5)
            except RedisError as e:
                # redis 抖动 / 重启不应让 worker 进程崩溃; 退避后重连重试
                logger.warning("[generation-worker] redis 暂时不可用, %.0fs 后重试: %s", _REDIS_RETRY_DELAY_S, e)
                time.sleep(_REDIS_RETRY_DELAY_S)
                continue
            if not run_id:
                continue
            try:
                self.process_run(run_id)
            except Exception:
                logger.exception("[generation-worker] run failed unexpectedly: %s", run_id)
        logger.info("[generation-worker] stopped")

    def process_run(self, run_id: str) -> None:
        run = repo.mark_generation_run_running(run_id)
        if run is None:
            logger.info("[generation-worker] skip run not queued: %s", run_id)
            return
        _publish_event(run_id, "status", {"type": "status", "stage": "running"})

        user_msg = repo.get_message(run.user_message_id)
        assistant_msg = repo.get_message(run.assistant_message_id)
        if user_msg is None or assistant_msg is None:
            raise RuntimeError("run messages not found")

        params = run.params or {}
        kb_ids = [str(x) for x in (params.get("kb_ids") or []) if str(x)]
        auth = AuthContext(user_id=run.owner_id, org_id=run.org_id)
        if bool(params.get("professional")):
            _apply_skill_scope(self.pipeline, auth)

        gen_cfg = getattr(self.pipeline, "config", None) and self.pipeline.config.generation or {}
        max_turns = int(gen_cfg.get("max_history_turns", 5))
        session = _session_from_parent(user_msg.parent_id, max_turns=max_turns)

        answer_parts: list[str] = []
        last_flush = time.time()
        try:
            for event in self.pipeline._get_query_flow().stream_chat_events(
                query=user_msg.content,
                session=session,
                # Agentic/professional 子图仍需完整接入 kb_id base filter；在单物理
                # collection 迁移期间，默认走 simple/hybrid，避免任何未过滤检索越权。
                use_agentic=False,
                mode=params.get("mode"),
                top_k=params.get("top_k"),
                kb_ids=kb_ids,
                professional=bool(params.get("professional", False)),
                collection=params.get("collection"),
            ):
                if repo.generation_run_should_stop(run_id):
                    repo.update_generation_run_status(run_id, "stopped", error="stopped")
                    repo.update_assistant_message(
                        run.assistant_message_id,
                        content="".join(answer_parts),
                        status="stopped",
                        error="stopped",
                    )
                    _publish_event(run_id, "error", {"type": "error", "message": "stopped"})
                    return

                event_type = str(event.get("type") or "message")
                if event_type == "text":
                    answer_parts.append(str(event.get("content") or ""))
                    now = time.time()
                    if now - last_flush >= 0.5:
                        repo.update_assistant_message(
                            run.assistant_message_id,
                            content="".join(answer_parts),
                            status="streaming",
                        )
                        last_flush = now

                if event_type == "done":
                    answer = str(event.get("answer") or "".join(answer_parts))
                    repo.update_assistant_message(
                        run.assistant_message_id,
                        content=answer,
                        hits=event.get("hits") or [],
                        context=event.get("context"),
                        research=event.get("research"),
                        usage=event.get("usage"),
                        latency_s=event.get("latency_s"),
                        status="done",
                    )
                    repo.set_conversation_active_leaf(
                        run.conversation_id,
                        run.assistant_message_id,
                        session_id=str(event.get("session_id") or "") or None,
                    )
                    repo.update_generation_run_status(run_id, "done")

                _publish_event(run_id, event_type, dict(event))

                if event_type == "done":
                    return
        except Exception as e:
            logger.exception("[generation-worker] process run error: %s", run_id)
            repo.update_generation_run_status(run_id, "failed", error=str(e))
            repo.update_assistant_message(
                run.assistant_message_id,
                content="".join(answer_parts),
                status="failed",
                error=str(e),
            )
            _publish_event(run_id, "error", {"type": "error", "message": str(e)})


def main() -> None:
    _setup_logging()
    worker = GenerationWorker()
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.run_forever()


if __name__ == "__main__":
    main()
