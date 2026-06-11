"""Ingest worker service: consumes upload/ingest tasks from Redis."""

from __future__ import annotations

import logging
import os
import signal
import threading
from typing import Any

from ..clients import object_store
from ..clients import redis as redis_runtime
from ..db import configured as db_configured
from ..db import init_db
from ..db import repo
from ..pipeline import Pipeline

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _publish_event(task_id: str, event_type: str, payload: dict[str, Any]) -> None:
    ev = repo.append_ingest_task_event(task_id, event_type, payload)
    live = dict(ev.payload)
    live.setdefault("type", ev.type)
    live["seq"] = ev.seq
    live["task_id"] = ev.task_id
    redis_runtime.get_redis_runtime().publish_ingest_event(task_id, live)


def _upload_doc_artifacts(collection: str, doc_id: str, doc_dir: str | None) -> str | None:
    if not object_store.configured() or not doc_dir or not os.path.isdir(doc_dir):
        return None
    client = object_store.get_object_store()
    prefix = object_store.document_prefix(collection, doc_id)
    for root, _, files in os.walk(doc_dir):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, doc_dir).replace(os.sep, "/")
            key = f"{prefix}/{rel}"
            content_type = (
                "application/pdf"
                if name.lower().endswith(".pdf")
                else "application/json"
                if name.lower().endswith(".json")
                else None
            )
            try:
                client.upload_file(path, key, content_type=content_type)
            except Exception as e:
                logger.warning("[ingest-worker] artifact upload failed %s -> %s: %s", path, key, e)
    return prefix


def _result_ok(result) -> bool:
    return bool(result.steps) and all(s.success for s in result.steps)


def _first_failure(result) -> str | None:
    for step in result.steps:
        if not step.success:
            return f"{step.step}: {step.error or '未知错误'}"
    if not result.steps:
        return "未执行任何步骤"
    return None


class IngestWorker:
    def __init__(self) -> None:
        if not db_configured():
            raise RuntimeError("DATABASE_URL 未配置")
        if not redis_runtime.configured():
            raise RuntimeError("REDIS_URL 未配置")
        init_db()
        self.redis = redis_runtime.get_redis_runtime()
        self.pipeline = Pipeline(config_path=os.environ.get("CONFIG_PATH") or None)
        self.stopping = False

    def stop(self, *_args) -> None:
        self.stopping = True

    def run_forever(self) -> None:
        logger.info("[ingest-worker] started")
        while not self.stopping:
            task_id = self.redis.dequeue_ingest_task(timeout_s=5)
            if not task_id:
                continue
            try:
                self.process_task(task_id)
            except Exception:
                logger.exception("[ingest-worker] task failed unexpectedly: %s", task_id)
        logger.info("[ingest-worker] stopped")

    def process_task(self, task_id: str) -> None:
        task = repo.mark_ingest_task_running(task_id)
        if task is None:
            logger.info("[ingest-worker] skip task not queued: %s", task_id)
            return
        _publish_event(task_id, "status", {"type": "status", "status": "running"})

        params = task.params or {}
        backend = params.get("backend") or None
        items = repo.list_ingest_task_items(task_id)
        if not items:
            repo.update_ingest_task_status(task_id, "done", result={"total": 0})
            _publish_event(task_id, "done", {"type": "done", "result": {"total": 0}})
            return

        for item in items:
            if item.status == "skipped":
                continue
            if repo.ingest_task_should_cancel(task_id):
                repo.cancel_pending_ingest_items(task_id)
                task = repo.update_ingest_task_counts(task_id)
                repo.update_ingest_task_status(task_id, "cancelled", error="cancelled")
                _publish_event(
                    task_id,
                    "cancelled",
                    {"type": "cancelled", "status": "cancelled", "progress": task.progress if task else 0},
                )
                return

            repo.update_ingest_task_item(item.id, status="running")
            _publish_event(
                task_id,
                "item",
                {"type": "item", "item_id": item.id, "doc_id": item.doc_id, "status": "running"},
            )
            try:
                result = self.pipeline.ingest_files(
                    [item.pdf_path],
                    collection=task.collection_name,
                    output_dir=item.doc_dir,
                    backend=backend,
                )
                ok = _result_ok(result)
                artifact_prefix = _upload_doc_artifacts(task.collection_name, item.doc_id, item.doc_dir)
                chunk_count = int(result.total_chunks or 0) if ok else 0
                status = "ready" if ok else "failed"
                error = None if ok else _first_failure(result)
                repo.update_ingest_task_item(
                    item.id,
                    status=status,
                    error=error,
                    chunk_count=chunk_count,
                    artifact_prefix=artifact_prefix or item.artifact_prefix,
                )
                repo.upsert_document(
                    collection_name=task.collection_name,
                    doc_id=item.doc_id,
                    owner_id=item.owner_id,
                    title=item.doc_id,
                    filename=item.filename,
                    pdf_object_key=item.pdf_object_key,
                    artifact_prefix=artifact_prefix or item.artifact_prefix,
                    status=status,
                    task_id=task_id,
                    chunk_count=chunk_count,
                )
                _publish_event(
                    task_id,
                    "item",
                    {
                        "type": "item",
                        "item_id": item.id,
                        "doc_id": item.doc_id,
                        "status": status,
                        "error": error,
                        "chunk_count": chunk_count,
                    },
                )
            except Exception as e:
                logger.exception("[ingest-worker] item failed: %s/%s", task_id, item.doc_id)
                repo.update_ingest_task_item(item.id, status="failed", error=str(e))
                repo.upsert_document(
                    collection_name=task.collection_name,
                    doc_id=item.doc_id,
                    owner_id=item.owner_id,
                    title=item.doc_id,
                    filename=item.filename,
                    pdf_object_key=item.pdf_object_key,
                    artifact_prefix=item.artifact_prefix,
                    status="failed",
                    task_id=task_id,
                    chunk_count=0,
                )
                _publish_event(
                    task_id,
                    "item",
                    {
                        "type": "item",
                        "item_id": item.id,
                        "doc_id": item.doc_id,
                        "status": "failed",
                        "error": str(e),
                    },
                )
            finally:
                task = repo.update_ingest_task_counts(task_id)
                if task:
                    _publish_event(
                        task_id,
                        "progress",
                        {
                            "type": "progress",
                            "progress": task.progress,
                            "completed_items": task.completed_items,
                            "failed_items": task.failed_items,
                            "skipped_items": task.skipped_items,
                            "total_items": task.total_items,
                        },
                    )

        try:
            self.pipeline.flush_collection(task.collection_name)
        except Exception as e:
            logger.warning("[ingest-worker] flush failed %s: %s", task.collection_name, e)

        final = repo.update_ingest_task_counts(task_id)
        result = {
            "total": final.total_items if final else 0,
            "success": (final.completed_items - final.failed_items - final.skipped_items) if final else 0,
            "failed": final.failed_items if final else 0,
            "skipped": final.skipped_items if final else 0,
        }
        status = "failed" if final and final.failed_items and final.completed_items == final.total_items else "done"
        repo.update_ingest_task_status(task_id, status, result=result)
        _publish_event(task_id, status, {"type": status, "status": status, "result": result})


def main() -> None:
    _setup_logging()
    concurrency = max(1, int(os.environ.get("INGEST_WORKER_CONCURRENCY", "1")))
    workers = [IngestWorker() for _ in range(concurrency)]

    def stop_all(*_args) -> None:
        for w in workers:
            w.stop()

    signal.signal(signal.SIGTERM, stop_all)
    signal.signal(signal.SIGINT, stop_all)
    if concurrency == 1:
        workers[0].run_forever()
        return
    threads = [
        threading.Thread(target=w.run_forever, name=f"ingest-worker-{i}", daemon=False)
        for i, w in enumerate(workers, 1)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
