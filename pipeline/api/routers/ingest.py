"""灌入: rebuild / append / parse / load-vec / upload (异步)"""

from __future__ import annotations

import logging
import os
import threading
import time as _time_mod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..deps import get_pipeline, get_task_store, verify_api_key
from ..models import IngestRequest, ParseRequest, LoadVecRequest, TaskResponse
from ...flows.ingest import IngestFlow, IngestResult
from .collections import (
    _kb_meta_path,
    _write_kb_meta,
    kb_workspace_dir,
    make_collection_slug,
    sanitized_doc_stem,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class _SlidingWindowRateLimiter:
    """线程安全滑动窗口限速器: 任意 ``period`` 秒内最多放行 ``max_calls`` 次。

    用于限制 UniParser 解析的启动频率 (每分钟不超过 N 次), 避免超配额。
    """

    def __init__(self, max_calls: int, period: float = 60.0) -> None:
        self.max_calls = max(1, int(max_calls))
        self.period = float(period)
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """阻塞直到允许再发起一次调用 (并登记本次时间)。"""
        while True:
            with self._lock:
                now = _time_mod.monotonic()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0])
            _time_mod.sleep(min(max(wait, 0.05), self.period))


def _make_ingest_flow(pipe: Any, collection: str, backend: Optional[str]) -> IngestFlow:
    """为单篇上传构建独立 IngestFlow (独立 Config), 供多线程并发互不干扰。

    不走 ``pipe.ingest_files`` —— 后者会改写共享的 ``config.milvus['collection']``
    并重置 ``pipe._ingest_flow``, 并发调用会互相踩。这里给每个任务一份独立 Config,
    只共享底层 (注册表缓存的) MilvusIngester / Embedder, 它们本身可并发使用。
    """
    from ...config import Config

    cfg = Config(pipe.config.to_dict())
    cfg.milvus["collection"] = collection
    if backend:
        cfg.parsing["backend"] = backend
    return IngestFlow(cfg)


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


def _file_ok(r: IngestResult) -> bool:
    """单篇文档是否全链路成功 (有步骤且每一步都成功)。"""
    return bool(r.steps) and all(s.success for s in r.steps)


def _first_failure(r: IngestResult) -> Optional[str]:
    """提取一篇文档第一个失败步骤的原因 (供前端直接展示真实错误)。"""
    for s in r.steps:
        if not s.success:
            return f"{s.step}: {s.error or '未知错误'}"
    if not r.steps:
        return "未执行任何步骤 (上传保存或解析提交阶段失败)"
    return None


def _step_dicts(r: IngestResult) -> List[Dict[str, Any]]:
    """把步骤摘要转成 dict, 失败步骤带上 error 原因。"""
    out: List[Dict[str, Any]] = []
    for s in r.steps:
        d: Dict[str, Any] = {"step": s.step, "success": s.success, "elapsed": round(s.elapsed, 2)}
        if not s.success and s.error:
            d["error"] = s.error
        out.append(d)
    return out


def _summarize_ingest(results: List[IngestResult]) -> Dict[str, Any]:
    ok_flags = [_file_ok(r) for r in results]
    success = sum(ok_flags)
    failed = len(results) - success
    # stored_chunks 只统计全链路成功的文档; total_chunks (来自 chunk 步) 失败时仍 > 0,
    # 但那些块并未入库, 单看它会误以为成功 — 所以两者都返回, 以 stored_chunks 为准。
    stored_chunks = sum(r.total_chunks for r, ok in zip(results, ok_flags) if ok)
    total_chunks = sum(r.total_chunks for r in results)
    failed_reasons = [
        {"doc_id": r.doc_id, "reason": _first_failure(r)}
        for r, ok in zip(results, ok_flags) if not ok
    ]
    return {
        "total": len(results),
        "success": success,
        "failed": failed,
        "stored_chunks": stored_chunks,
        "total_chunks": total_chunks,
        "failed_reasons": failed_reasons,
        "details": [
            {
                "doc_id": r.doc_id,
                "ok": ok,
                "total_chunks": r.total_chunks,
                "steps": _step_dicts(r),
            }
            for r, ok in zip(results, ok_flags)
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
    skipped_existing: List[str] | None = None,
) -> Dict[str, Any]:
    """后台任务: 将已上传的 PDF 灌入指定集合。

    items: [(pdf_path, doc_dir)], 中间产物落在每篇文档自己的 doc_dir 下,
    与知识库工作目录绑定, 便于按库管理 / 重建 / 清理。
    skipped_existing: 因已入库 (同名 doc_id 已在集合中) 而被跳过的原始文件名,
    仅用于结果回显。
    """
    pipe = get_pipeline()
    task_store = get_task_store()
    skipped_existing = skipped_existing or []
    total = len(items)

    # 解析后端决定并发策略: 仅 uniparser 走并发 + 限速; 其它后端保持串行 (不改既有行为)。
    effective_backend = (
        backend or pipe.config.parsing.get("backend") or "mineru"
    ).strip().lower()
    uni_cfg = pipe.config.uniparser or {}
    if effective_backend == "uniparser":
        concurrency = max(1, int(uni_cfg.get("upload_concurrency", 3)))
        rate_per_min = int(uni_cfg.get("upload_rate_per_minute", 8))
    else:
        concurrency = 1
        rate_per_min = 0
    concurrency = max(1, min(concurrency, total))
    limiter = (
        _SlidingWindowRateLimiter(rate_per_min, 60.0) if rate_per_min > 0 else None
    )
    logger.info(
        f"[ingest-upload] 开始灌入 {total} 篇 (collection={collection}, "
        f"backend={effective_backend}, 并发={concurrency}, "
        f"限速={rate_per_min or '无'}/min)"
    )

    results: List[IngestResult] = [None] * total  # type: ignore[list-item]
    progress_lock = threading.Lock()
    done_count = {"n": 0}

    def _work(idx: int, fp: str, doc_dir: str) -> None:
        # 限速针对"启动解析"这一刻: 阻塞直到本分钟配额允许, 再开始整条灌入链。
        if limiter is not None:
            limiter.acquire()
        try:
            flow = _make_ingest_flow(pipe, collection, backend)
            r = flow.run([fp], output_dir=doc_dir)
        except Exception as e:
            logger.warning(f"[ingest-upload] 灌入失败 {fp}: {e}")
            r = IngestResult(file_paths=[fp], steps=[])
        results[idx] = r
        with progress_lock:
            done_count["n"] += 1
            task_store.update_progress(
                task_id, done_count["n"], total, os.path.basename(doc_dir)
            )

    if concurrency <= 1:
        for idx, (fp, doc_dir) in enumerate(items):
            _work(idx, fp, doc_dir)
    else:
        with ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="ingest-upload"
        ) as ex:
            futures = [
                ex.submit(_work, idx, fp, doc_dir)
                for idx, (fp, doc_dir) in enumerate(items)
            ]
            for fut in futures:
                fut.result()  # 等待全部完成 (异常已在 _work 内捕获)

    # flush 一次, 让列表 row_count 立即反映新灌入的数据
    try:
        pipe.flush_collection(collection)
    except Exception as e:
        logger.warning(f"[ingest-upload] flush 失败 {collection}: {e}")

    ok_flags = [_file_ok(r) for r in results]
    success = sum(ok_flags)
    failed = len(results) - success
    # stored_chunks: 真正入库的块数 (只统计全链路成功的文档)。
    # total_chunks 仍是 chunk 步产出, 失败时 > 0 但未入库 — 单看它会误判成功。
    stored_chunks = sum(r.total_chunks for r, ok in zip(results, ok_flags) if ok)
    total_chunks = sum(r.total_chunks for r in results)
    failed_reasons = [
        {"doc_id": r.doc_id, "reason": _first_failure(r)}
        for r, ok in zip(results, ok_flags) if not ok
    ]
    if failed_reasons:
        logger.warning(
            f"[ingest-upload] {failed}/{total} 篇灌入失败 (collection={collection}): "
            f"{failed_reasons}"
        )
    return {
        "collection": collection,
        "files": total,
        "success": success,
        "failed": failed,
        "skipped_existing": len(skipped_existing),
        "skipped_files": skipped_existing,
        "stored_chunks": stored_chunks,
        "total_chunks": total_chunks,
        "failed_reasons": failed_reasons,
        "details": [
            {
                "doc_id": r.doc_id,
                "ok": ok,
                "total_chunks": r.total_chunks,
                "steps": _step_dicts(r),
            }
            for r, ok in zip(results, ok_flags)
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

    断点续传: 同一知识库再次上传时, 按 PDF 文件名 (= doc_id) 跳过集合中
    已入库的文献, 只灌未入库的; 中断 (未入库) 的文献复用同一目录续灌。
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
        items.append((save_path, doc_dir))
        logger.info(f"[ingest-upload] 已保存: {save_path} ({len(content)} bytes)")

    if skipped_existing:
        logger.info(
            f"[ingest-upload] 共跳过 {len(skipped_existing)} 篇已入库文献: "
            f"{skipped_existing}"
        )

    task_store = get_task_store()
    tid = uuid.uuid4().hex[:16]
    task_store.submit(
        _ingest_upload, tid, items, safe_collection, backend, skipped_existing,
        task_id=tid,
    )
    return TaskResponse(id=tid, status="pending", created_at=_time.time())
