"""灌入: rebuild / append / parse / load-vec / upload (异步)"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..deps import get_pipeline, get_task_store, verify_api_key
from ..models import IngestRequest, ParseRequest, LoadVecRequest, TaskResponse
from ...flows.ingest import IngestResult
from .collections import (
    _kb_meta_path,
    _write_kb_meta,
    kb_workspace_dir,
    make_collection_slug,
    safe_doc_stem,
)

logger = logging.getLogger(__name__)

router = APIRouter()


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


@router.post("/ingest/rebuild", response_model=TaskResponse)
def ingest_rebuild(
    req: IngestRequest,
    _auth: str = Depends(verify_api_key),
) -> TaskResponse:
    """全量重灌 (异步): 清空集合 → 逐篇灌入。"""
    import uuid, time as _time
    task_store = get_task_store()
    tid = uuid.uuid4().hex[:16]
    task_store.submit(_ingest_rebuild, tid, req.directory, task_id=tid)
    return TaskResponse(id=tid, status="pending", created_at=_time.time())


@router.post("/ingest/append", response_model=TaskResponse)
def ingest_append(
    req: IngestRequest,
    _auth: str = Depends(verify_api_key),
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
    _auth: str = Depends(verify_api_key),
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
    _auth: str = Depends(verify_api_key),
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
    _auth: str = Depends(verify_api_key),
) -> TaskResponse:
    """查询异步任务状态。"""
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


# ---------------------------------------------------------------------------
# 上传 + 自动灌入 (一步完成)
# ---------------------------------------------------------------------------

def _ingest_upload(
    task_id: str,
    items: List[tuple],
    collection: str,
    backend: str | None,
) -> Dict[str, Any]:
    """后台任务: 将已上传的 PDF 灌入指定集合。

    items: [(pdf_path, doc_dir)], 中间产物落在每篇文档自己的 doc_dir 下,
    与知识库工作目录绑定, 便于按库管理 / 重建 / 清理。
    """
    pipe = get_pipeline()
    task_store = get_task_store()
    total = len(items)

    results = []
    for idx, (fp, doc_dir) in enumerate(items, start=1):
        try:
            r = pipe.ingest_files(
                [fp],
                collection=collection,
                output_dir=doc_dir,
                backend=backend or None,
            )
            results.append(r)
            task_store.update_progress(task_id, idx, total, os.path.basename(doc_dir))
        except Exception as e:
            logger.warning(f"[ingest-upload] 灌入失败 {fp}: {e}")
            results.append(IngestResult(file_paths=[fp], steps=[]))
            task_store.update_progress(task_id, idx, total, os.path.basename(doc_dir))

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
    _auth: str = Depends(verify_api_key),
) -> TaskResponse:
    """上传 PDF + 自动灌入到指定知识库 (异步任务)。

    集合名会被自动清洗并加 kb_ 前缀。若知识库不存在会自动创建。
    每篇 PDF 连同其中间产物落在 ``<UPLOAD_DIR>/kb_<name>/<doc_stem>/`` 下,
    原始文件名 (去后缀) 作为文档目录名 / doc_id。
    """
    import uuid
    import time as _time

    # 与新建保持一致: 支持中文名 (解析为 ASCII slug); 已是 kb_ 名时幂等
    safe_collection = make_collection_slug(collection)
    kb_dir = kb_workspace_dir(safe_collection)
    os.makedirs(kb_dir, exist_ok=True)
    # 直接带 (中文) 集合名上传时, 若尚无元数据则补写显示名, 保证列表回显原名而非 slug。
    if not os.path.isfile(_kb_meta_path(kb_dir)):
        _write_kb_meta(kb_dir, (collection or "").strip(), safe_collection)

    # 每篇 PDF 一个独立子目录, 保存原始 PDF 为 source.pdf
    items: List[tuple] = []
    for f in files:
        doc_stem = safe_doc_stem(f.filename or "document.pdf", kb_dir)
        doc_dir = os.path.join(kb_dir, doc_stem)
        os.makedirs(doc_dir, exist_ok=True)
        # 以原始文件名 (清洗后) 保存, run() 据此推导 doc_title/doc_name
        save_path = os.path.join(doc_dir, f"{doc_stem}.pdf")
        content = await f.read()
        with open(save_path, "wb") as out:
            out.write(content)
        items.append((save_path, doc_dir))
        logger.info(f"[ingest-upload] 已保存: {save_path} ({len(content)} bytes)")

    task_store = get_task_store()
    tid = uuid.uuid4().hex[:16]
    task_store.submit(
        _ingest_upload, tid, items, safe_collection, backend,
        task_id=tid,
    )
    return TaskResponse(id=tid, status="pending", created_at=_time.time())
