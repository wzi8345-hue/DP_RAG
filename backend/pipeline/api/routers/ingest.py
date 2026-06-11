"""灌入: rebuild / append / parse / load-vec / upload (异步)"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..authz import require_write
from ..deps import AuthContext, get_pipeline, get_task_store, require_auth
from ..models import (
    IngestRequest,
    IngestTaskItemResponse,
    IngestTaskResponse,
    ParseRequest,
    LoadVecRequest,
    TaskResponse,
)
from ...clients import object_store
from ...clients import redis as redis_runtime
from ...db import repo
from ...flows.ingest import IngestResult
from .collections import (
    _kb_meta_path,
    _write_kb_meta,
    kb_workspace_dir,
    make_collection_slug,
    sanitized_doc_stem,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# 单请求上传硬上限: 防止一次 multipart 塞太多文件撑爆内存/文件描述符/body 解析。
_UPLOAD_MAX_FILES = int(os.environ.get("UPLOAD_MAX_FILES", "200"))
_UPLOAD_MAX_TOTAL_MB = int(os.environ.get("UPLOAD_MAX_TOTAL_MB", "500"))


def _ingest_task_payload(task_id: str) -> IngestTaskResponse:
    task = repo.get_ingest_task(task_id) if repo.available() else None
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    items = repo.list_ingest_task_items(task_id)
    return IngestTaskResponse(
        id=task.id,
        collection_name=task.collection_name,
        status=task.status,
        progress=round(task.progress or 0, 4),
        total_items=task.total_items,
        completed_items=task.completed_items,
        failed_items=task.failed_items,
        skipped_items=task.skipped_items,
        cancel_requested=task.cancel_requested,
        result=task.result,
        error=task.error,
        created_at=task.created_at.timestamp() if task.created_at else 0.0,
        items=[
            IngestTaskItemResponse(
                id=i.id,
                doc_id=i.doc_id,
                filename=i.filename,
                status=i.status,
                error=i.error,
                chunk_count=i.chunk_count,
            )
            for i in items
        ],
    )


def _ingest_event_payload(ev) -> dict:
    payload = dict(ev.payload or {})
    payload.setdefault("type", ev.type)
    payload["seq"] = ev.seq
    payload["task_id"] = ev.task_id
    return payload


def _append_and_publish_ingest_event(task_id: str, event_type: str, payload: dict) -> None:
    ev = repo.append_ingest_task_event(task_id, event_type, payload)
    live = _ingest_event_payload(ev)
    if redis_runtime.configured():
        redis_runtime.get_redis_runtime().publish_ingest_event(task_id, live)


def _ingest_rebuild(task_id: str, directory: str) -> Dict[str, Any]:
    pipe = get_pipeline()
    task_store = get_task_store()

    def on_progress(current, total, doc_id, status):
        task_store.update_progress(task_id, current, total, doc_id)

    results = pipe._get_ingest_flow().vectorize_from_directory(
        directory, recreate=True, skip_existing=False,
        progress_callback=on_progress,
    )
    return _summarize_ingest(results)


def _ingest_append(task_id: str, directory: str, skip_existing: bool) -> Dict[str, Any]:
    pipe = get_pipeline()
    task_store = get_task_store()

    def on_progress(current, total, doc_id, status):
        task_store.update_progress(task_id, current, total, doc_id)

    results = pipe._get_ingest_flow().vectorize_from_directory(
        directory, recreate=False, skip_existing=skip_existing,
        progress_callback=on_progress,
    )
    return _summarize_ingest(results)


def _ingest_parse(task_id: str, path: str, output_dir: str | None, backend: str | None, timeout: int) -> Dict[str, Any]:
    pipe = get_pipeline()
    results = pipe.parse([path], output_dir=output_dir, parse_timeout=timeout, backend=backend)
    return _summarize_ingest([results])


def _ingest_load_vec(task_id: str, path: str, recreate: bool, purge: bool, skip: bool) -> Dict[str, Any]:
    pipe = get_pipeline()
    results = pipe.load_vec(path, recreate=recreate, purge_existing=purge, skip_existing=skip)
    total = sum(int(r.get("count", 0) or 0) for r in results)
    return {"files_loaded": len(results), "total_chunks": total}


def _summarize_ingest(results: List[IngestResult]) -> Dict[str, Any]:
    success = sum(1 for r in results if r.steps and all(s.success for s in r.steps))
    failed = len(results) - success
    return {
        "total": len(results),
        "success": success,
        "failed": failed,
        "details": [
            {
                "doc_id": r.doc_id,
                "total_chunks": r.total_chunks,
                "steps": [{"step": s.step, "success": s.success, "elapsed": round(s.elapsed, 2)} for s in r.steps],
            }
            for r in results
        ],
    }


def _upload_doc_artifacts(collection: str, doc_id: str, doc_dir: str) -> str | None:
    """Upload local parse/vector artifacts for one document; return object prefix."""
    if not object_store.configured() or not os.path.isdir(doc_dir):
        return None
    client = object_store.get_object_store()
    prefix = object_store.document_prefix(collection, doc_id)
    for root, _, files in os.walk(doc_dir):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, doc_dir).replace(os.sep, "/")
            key = f"{prefix}/{rel}"
            content_type = "application/pdf" if name.lower().endswith(".pdf") else "application/json" if name.lower().endswith(".json") else None
            try:
                client.upload_file(path, key, content_type=content_type)
            except Exception as e:
                logger.warning("[object-store] 上传解析产物失败 %s -> %s: %s", path, key, e)
    return prefix


@router.post("/ingest/rebuild", response_model=TaskResponse)
def ingest_rebuild(
    req: IngestRequest,
    _auth: str = Depends(require_auth),
) -> TaskResponse:
    """全量重灌 (异步): 清空集合 → 逐篇灌入。"""
    import uuid, time as _time
    task_store = get_task_store()
    tid = uuid.uuid4().hex[:16]
    task_store.submit(_ingest_rebuild, tid, req.directory, task_id=tid)
    return TaskResponse(id=tid, status="queued", created_at=_time.time())


@router.post("/ingest/append", response_model=TaskResponse)
def ingest_append(
    req: IngestRequest,
    _auth: str = Depends(require_auth),
) -> TaskResponse:
    """增量追加 (异步): 不清空集合, 同名 doc_id 覆盖。"""
    task_store = get_task_store()
    import uuid, time as _time
    tid = uuid.uuid4().hex[:16]
    task_store.submit(
        _ingest_append, tid, req.directory, req.skip_existing, task_id=tid,
    )
    return TaskResponse(id=tid, status="pending", created_at=_time.time())


@router.post("/ingest/parse", response_model=TaskResponse)
def ingest_parse(
    req: ParseRequest,
    _auth: str = Depends(require_auth),
) -> TaskResponse:
    """仅解析 PDF (异步): 不做 chunk/embed/store。"""
    task_store = get_task_store()
    import uuid, time as _time
    tid = uuid.uuid4().hex[:16]
    task_store.submit(
        _ingest_parse, tid, req.path, req.output_dir, req.backend, req.timeout,
        task_id=tid,
    )
    return TaskResponse(id=tid, status="pending", created_at=_time.time())


@router.post("/ingest/load-vec", response_model=TaskResponse)
def ingest_load_vec(
    req: LoadVecRequest,
    _auth: str = Depends(require_auth),
) -> TaskResponse:
    """直接灌入已向量化文件 (异步): 跳过 parse/chunk/embed。"""
    task_store = get_task_store()
    import uuid, time as _time
    tid = uuid.uuid4().hex[:16]
    task_store.submit(
        _ingest_load_vec, tid, req.path, req.recreate, req.purge_existing, req.skip_existing,
        task_id=tid,
    )
    return TaskResponse(id=tid, status="pending", created_at=_time.time())


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(
    task_id: str,
    _auth: str = Depends(require_auth),
) -> TaskResponse:
    """查询异步任务状态。"""
    if repo.available():
        task = repo.get_ingest_task(task_id)
        if task is not None:
            return TaskResponse(
                id=task.id,
                status=task.status,
                progress=round(task.progress or 0, 4),
                result=task.result,
                error=task.error,
                created_at=task.created_at.timestamp() if task.created_at else 0.0,
            )
    task_store = get_task_store()
    status = task_store.get(task_id)
    if status is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskResponse(
        id=status.id,
        status=status.status,
        progress=round(status.progress, 4),
        result=status.result,
        error=status.error,
        created_at=status.created_at,
    )


@router.get("/ingest/tasks/{task_id}", response_model=IngestTaskResponse)
def get_ingest_task(
    task_id: str,
    auth: AuthContext = Depends(require_auth),
) -> IngestTaskResponse:
    task = repo.get_ingest_task(task_id) if repo.available() else None
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.owner_id != auth.user_id:
        raise HTTPException(status_code=403, detail="No permission for task")
    return _ingest_task_payload(task_id)


@router.post("/ingest/tasks/{task_id}/cancel", response_model=IngestTaskResponse)
def cancel_ingest_task(
    task_id: str,
    auth: AuthContext = Depends(require_auth),
) -> IngestTaskResponse:
    task = repo.get_ingest_task(task_id) if repo.available() else None
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.owner_id != auth.user_id:
        raise HTTPException(status_code=403, detail="No permission for task")
    repo.request_ingest_task_cancel(task_id)
    _append_and_publish_ingest_event(
        task_id,
        "status",
        {"type": "status", "status": "cancelling", "task_id": task_id},
    )
    return _ingest_task_payload(task_id)


@router.get("/ingest/tasks/{task_id}/stream")
def stream_ingest_task(
    task_id: str,
    after_seq: int = 0,
    auth: AuthContext = Depends(require_auth),
) -> StreamingResponse:
    task = repo.get_ingest_task(task_id) if repo.available() else None
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.owner_id != auth.user_id:
        raise HTTPException(status_code=403, detail="No permission for task")

    def event_generator():
        last_seq = max(0, int(after_seq or 0))
        last_redis_id = "$"
        terminal_types = {"done", "failed", "cancelled", "error"}
        try:
            while True:
                replayed = repo.list_ingest_task_events(task_id, after_seq=last_seq, limit=500)
                for ev in replayed:
                    payload = _ingest_event_payload(ev)
                    last_seq = max(last_seq, ev.seq)
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    if payload.get("type") in terminal_types:
                        return

                current = repo.get_ingest_task(task_id)
                if current and current.status in {"done", "failed", "cancelled"}:
                    return

                if redis_runtime.configured():
                    for redis_id, payload in redis_runtime.get_redis_runtime().read_ingest_events(
                        task_id,
                        last_id=last_redis_id,
                        block_ms=1000,
                        count=100,
                    ):
                        last_redis_id = redis_id
                        seq = int(payload.get("seq") or 0)
                        if seq <= last_seq:
                            continue
                        last_seq = seq
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        if payload.get("type") in terminal_types:
                            return
                else:
                    time.sleep(1)
        except Exception as e:
            logger.exception("[ingest/tasks/stream] 生成器异常")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# 上传 + 自动灌入 (一步完成)
# ---------------------------------------------------------------------------

def _ingest_upload(
    task_id: str,
    items: List[tuple],
    collection: str,
    backend: str | None,
    skipped_existing: List[str] | None = None,
) -> Dict[str, Any]:
    """后台任务: 将已上传的 PDF 灌入指定集合。

    items: [(pdf_path, doc_dir, doc_id, filename, owner_id, pdf_object_key, artifact_prefix)],
    与知识库工作目录绑定, 便于按库管理 / 重建 / 清理。
    skipped_existing: 因已入库 (同名 doc_id 已在集合中) 而被跳过的原始文件名,
    仅用于结果回显。
    """
    pipe = get_pipeline()
    task_store = get_task_store()
    skipped_existing = skipped_existing or []
    total = len(items)

    results = []
    for idx, (fp, doc_dir, doc_id, filename, owner_id, pdf_key, artifact_prefix) in enumerate(items, start=1):
        doc_status = "failed"
        chunk_count = 0
        try:
            r = pipe.ingest_files(
                [fp],
                collection=collection,
                output_dir=doc_dir,
                backend=backend or None,
            )
            results.append(r)
            chunk_count = int(r.total_chunks or 0)
            doc_status = "ready" if r.steps and all(s.success for s in r.steps) else "failed"
            uploaded_prefix = _upload_doc_artifacts(collection, doc_id, doc_dir)
            artifact_prefix = uploaded_prefix or artifact_prefix
            task_store.update_progress(task_id, idx, total, os.path.basename(doc_dir))
        except Exception as e:
            logger.warning(f"[ingest-upload] 灌入失败 {fp}: {e}")
            results.append(IngestResult(file_paths=[fp], steps=[]))
            task_store.update_progress(task_id, idx, total, os.path.basename(doc_dir))
        finally:
            if repo.available():
                try:
                    repo.upsert_document(
                        collection_name=collection,
                        doc_id=doc_id,
                        owner_id=owner_id,
                        title=doc_id,
                        filename=filename,
                        pdf_object_key=pdf_key,
                        artifact_prefix=artifact_prefix,
                        status=doc_status,
                        task_id=task_id,
                        chunk_count=chunk_count,
                    )
                except Exception as e:
                    logger.warning("[db] 回填 document 失败 %s/%s: %s", collection, doc_id, e)

    # flush 一次, 让列表 row_count 立即反映新灌入的数据
    try:
        pipe.flush_collection(collection)
    except Exception as e:
        logger.warning(f"[ingest-upload] flush 失败 {collection}: {e}")

    total_chunks = sum(r.total_chunks for r in results)
    success = sum(1 for r in results if r.steps and all(s.success for s in r.steps))
    failed = len(results) - success
    return {
        "collection": collection,
        "files": total,
        "success": success,
        "failed": failed,
        "skipped_existing": len(skipped_existing),
        "skipped_files": skipped_existing,
        "total_chunks": total_chunks,
        "details": [
            {
                "doc_id": r.doc_id,
                "total_chunks": r.total_chunks,
                "steps": [
                    {"step": s.step, "success": s.success, "elapsed": round(s.elapsed, 2)}
                    for s in r.steps
                ],
            }
            for r in results
        ],
    }


@router.post("/ingest/upload", response_model=TaskResponse)
async def ingest_upload(
    collection: str = Form(...),
    backend: str = Form(None),
    files: List[UploadFile] = File(...),
    auth: AuthContext = Depends(require_auth),
) -> TaskResponse:
    """上传 PDF + 自动灌入到指定知识库 (异步任务)。

    集合名会被自动清洗并加 kb_ 前缀。若知识库不存在会自动创建。
    每篇 PDF 连同其中间产物落在 ``<UPLOAD_DIR>/kb_<name>/<doc_stem>/`` 下,
    原始文件名 (去后缀) 作为文档目录名 / doc_id。

    断点续传: 同一知识库再次上传时, 按 PDF 文件名 (= doc_id) 跳过集合中
    已入库的文献, 只灌未入库的; 中断 (未入库) 的文献复用同一目录续灌。
    """
    import uuid
    import time as _time

    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    if not redis_runtime.configured():
        raise HTTPException(status_code=503, detail="REDIS_URL 未配置")

    if len(files) > _UPLOAD_MAX_FILES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"单次最多上传 {_UPLOAD_MAX_FILES} 个文件 (本次 {len(files)} 个); "
                "请分批上传。"
            ),
        )
    total_bytes = sum(int(getattr(f, "size", 0) or 0) for f in files)
    max_total_bytes = _UPLOAD_MAX_TOTAL_MB * 1024 * 1024
    if total_bytes > max_total_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"单次上传总大小 {total_bytes / 1024 / 1024:.0f}MB 超过上限 "
                f"{_UPLOAD_MAX_TOTAL_MB}MB; 请分批上传或减少单批文件数。"
            ),
        )

    # 与新建保持一致: 支持中文名 (解析为 ASCII slug); 已是 kb_ 名时幂等
    safe_collection = make_collection_slug(collection)
    kb_dir = kb_workspace_dir(safe_collection)
    os.makedirs(kb_dir, exist_ok=True)
    # 直接带 (中文) 集合名上传时, 若尚无元数据则补写显示名, 保证列表回显原名而非 slug。
    if not os.path.isfile(_kb_meta_path(kb_dir)):
        _write_kb_meta(kb_dir, (collection or "").strip(), safe_collection)
    if repo.available():
        meta = repo.get_collection(safe_collection)
        if meta is None:
            repo.upsert_collection(
                name=safe_collection,
                display_name=(collection or "").strip(),
                auth=auth,
                visibility="private",
            )
        else:
            require_write(auth, meta)

    # 查询该集合中已入库的 doc_id, 用于按文件名跳过 (best-effort, 失败则不跳过)
    pipe = get_pipeline()
    try:
        existing_doc_ids = pipe.list_doc_ids(safe_collection)
    except Exception as e:
        logger.warning(f"[ingest-upload] 查询已入库 doc_id 失败, 不跳过: {e}")
        existing_doc_ids = set()

    # 每篇 PDF 一个独立子目录, 保存原始 PDF; 已入库的 (同名 doc_id) 直接跳过
    items: List[tuple] = []
    skipped_existing: List[str] = []
    for f in files:
        # 用确定性 stem: 同一文件名恒映射到同一 doc_id, 不追加 _2 后缀
        doc_stem = sanitized_doc_stem(f.filename or "document.pdf")
        if doc_stem in existing_doc_ids:
            skipped_existing.append(f.filename or doc_stem)
            logger.info(
                f"[ingest-upload] 跳过已入库: {f.filename!r} (doc_id={doc_stem})"
            )
            continue
        doc_dir = os.path.join(kb_dir, doc_stem)
        os.makedirs(doc_dir, exist_ok=True)
        # 以原始文件名 (清洗后) 保存, run() 据此推导 doc_title/doc_name;
        # 中断未入库的文献复用同一目录, 覆盖残留产物后续灌。
        save_path = os.path.join(doc_dir, f"{doc_stem}.pdf")
        content = await f.read()
        with open(save_path, "wb") as out:
            out.write(content)
        pdf_key = None
        artifact_prefix = object_store.document_prefix(safe_collection, doc_stem)
        if object_store.configured():
            try:
                pdf_key = object_store.pdf_object_key(safe_collection, doc_stem)
                object_store.get_object_store().upload_bytes(
                    content,
                    pdf_key,
                    content_type=f.content_type or "application/pdf",
                )
            except Exception as e:
                logger.warning("[object-store] 上传原始 PDF 失败 %s: %s", save_path, e)
        if repo.available():
            repo.upsert_document(
                collection_name=safe_collection,
                doc_id=doc_stem,
                owner_id=auth.user_id,
                title=doc_stem,
                filename=f.filename,
                pdf_object_key=pdf_key,
                artifact_prefix=artifact_prefix,
                status="parsing",
                chunk_count=0,
            )
        items.append((save_path, doc_dir, doc_stem, f.filename or doc_stem, auth.user_id, pdf_key, artifact_prefix))
        logger.info(f"[ingest-upload] 已保存: {save_path} ({len(content)} bytes)")

    if skipped_existing:
        logger.info(
            f"[ingest-upload] 共跳过 {len(skipped_existing)} 篇已入库文献: "
            f"{skipped_existing}"
        )

    tid = uuid.uuid4().hex[:16]
    stream_key = redis_runtime.get_redis_runtime().ingest_stream_key(tid)
    repo.create_ingest_task(
        task_id=tid,
        auth=auth,
        collection_name=safe_collection,
        kind="upload",
        total_items=len(items),
        params={"backend": backend, "skipped_existing": skipped_existing},
        redis_stream=stream_key,
    )
    for save_path, doc_dir, doc_stem, filename, owner_id, pdf_key, artifact_prefix in items:
        repo.add_ingest_task_item(
            item_id=f"{tid}:{doc_stem}",
            task_id=tid,
            collection_name=safe_collection,
            owner_id=owner_id,
            doc_id=doc_stem,
            filename=filename,
            pdf_path=save_path,
            doc_dir=doc_dir,
            pdf_object_key=pdf_key,
            artifact_prefix=artifact_prefix,
        )
    for idx, skipped in enumerate(skipped_existing, 1):
        skipped_doc_id = sanitized_doc_stem(skipped)
        repo.add_ingest_task_item(
            item_id=f"{tid}:skip:{idx}:{skipped_doc_id}",
            task_id=tid,
            collection_name=safe_collection,
            owner_id=auth.user_id,
            doc_id=skipped_doc_id,
            filename=skipped,
            status="skipped",
            error="already ingested",
        )
    repo.update_ingest_task_counts(tid)
    _append_and_publish_ingest_event(
        tid,
        "status",
        {"type": "status", "status": "queued", "task_id": tid},
    )
    redis_runtime.get_redis_runtime().enqueue_ingest_task(tid)
    return TaskResponse(id=tid, status="pending", created_at=_time.time())
