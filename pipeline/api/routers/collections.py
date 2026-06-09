"""知识库 (Collection) 管理: 列表 / 新建 / 删除 / 重建。

每个知识库对应一个 Milvus 集合 (kb_ 前缀) + 一个本地工作目录
``<UPLOAD_DIR>/kb_<name>/``, 后者收纳原始 PDF 与所有中间产物
(解析结果 / 分块 / 向量化 json), 按文档分子目录:

    <UPLOAD_DIR>/kb_<name>/
      <doc_stem>/
        source.pdf
        knowledge_blocks.json
        knowledge_blocks_vec.json
        ... (mineru/uniparser 解析产物)

这样删除知识库可连带清理本地产物, 重建则复用已存解析产物 (跳过 PDF 解析)。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_pipeline, get_task_store, verify_api_key
from ..models import (
    CollectionInfo,
    CollectionsListResponse,
    CreateCollectionRequest,
    DeleteCollectionResponse,
    TaskResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_KB_PREFIX = "kb_"
_DEFAULT_COLLECTION = "literature_chunks"
# 集合名仅允许小写字母/数字/下划线, 最长 64 字符
_SANITIZE_RE = re.compile(r"[^a-z0-9_]")
# 文档目录名: 保留 unicode, 仅剔除路径分隔符与文件系统危险字符
_DOC_STEM_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

# 本地工作目录根 (与 ingest 上传保存路径一致)
UPLOAD_ROOT = os.environ.get(
    "UPLOAD_DIR",
    os.path.join(os.getcwd(), "uploads"),
)


def sanitize_collection_name(raw: str) -> str:
    """清洗用户输入的集合名, 加 kb_ 前缀。

    规则:
    - 转小写, 非 [a-z0-9_] 字符替换为下划线
    - 合并连续下划线, 去除首尾下划线
    - 自动加 kb_ 前缀 (若尚未有)
    - 截断到 64 字符
    """
    name = raw.strip().lower()
    name = _SANITIZE_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        raise HTTPException(status_code=400, detail="集合名不能为空")
    if not name.startswith(_KB_PREFIX):
        name = f"{_KB_PREFIX}{name}"
    if len(name) > 64:
        name = name[:64]
    return name


def make_collection_slug(raw: str) -> str:
    """由用户显示名生成合法的 Milvus 集合名 (kb_ 前缀, 纯 ASCII)。

    Milvus 集合名只能是 [A-Za-z0-9_] 且不能以数字开头, 无法直接用中文。
    - 纯 ASCII 名: 直接清洗为 kb_<base> (保持可读, 如 demo -> kb_demo)
    - 含非 ASCII (如中文): 用 kb_<base>_<hash> / kb_<hash> 保证唯一且非空,
      真实中文名另存到工作目录元数据, 列表/上传时回显。
    """
    s = (raw or "").strip()
    if not s:
        raise HTTPException(status_code=400, detail="知识库名不能为空")
    base = re.sub(r"_+", "_", _SANITIZE_RE.sub("_", s.lower())).strip("_")
    is_pure_ascii = all(ord(c) < 128 for c in s)
    if is_pure_ascii and base:
        name = base if base.startswith(_KB_PREFIX) else f"{_KB_PREFIX}{base}"
    else:
        digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
        base_noprefix = base[len(_KB_PREFIX):] if base.startswith(_KB_PREFIX) else base
        name = f"{_KB_PREFIX}{base_noprefix}_{digest}" if base_noprefix else f"{_KB_PREFIX}{digest}"
    return name[:64]


def kb_workspace_dir(kb_name: str) -> str:
    """返回某知识库的本地工作目录 (kb_name 须已带 kb_ 前缀)。"""
    return os.path.join(UPLOAD_ROOT, kb_name)


def _kb_meta_path(kb_dir: str) -> str:
    return os.path.join(kb_dir, ".kb_meta.json")


def _write_kb_meta(kb_dir: str, display_name: str, name: str) -> None:
    try:
        with open(_kb_meta_path(kb_dir), "w", encoding="utf-8") as f:
            json.dump(
                {"display_name": display_name, "collection": name, "created_at": time.time()},
                f,
                ensure_ascii=False,
            )
    except Exception as e:  # 元数据失败不影响主流程
        logger.warning(f"[collections] 写入 kb 元数据失败 {kb_dir}: {e}")


def _read_display_name(kb_dir: str, fallback: str) -> str:
    path = _kb_meta_path(kb_dir)
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("display_name") or fallback
        except Exception:
            pass
    return fallback


def sanitized_doc_stem(filename: str) -> str:
    """由原始文件名推导稳定的文档目录名 / doc_id (保留中文, 不做去重改名)。

    与 ``safe_doc_stem`` 的区别: 不追加 _2/_3 后缀, 因此同一 PDF 文件名总是
    映射到同一 doc_stem (= doc_id)。用于"同一知识库再次上传时按文件名去重":
    已入库的文献据此被跳过, 中断未入库的据此复用同一目录续灌。
    """
    stem = os.path.splitext(os.path.basename(filename or "document.pdf"))[0]
    stem = _DOC_STEM_RE.sub("_", stem).strip().strip(".")
    if not stem:
        stem = f"doc_{uuid.uuid4().hex[:8]}"
    if len(stem) > 100:
        stem = stem[:100]
    return stem


def safe_doc_stem(filename: str, kb_dir: str) -> str:
    """由原始文件名推导文档目录名 (保留中文), 与既有目录去重。"""
    stem = sanitized_doc_stem(filename)
    # 同名去重: 不同文件不互相覆盖
    candidate = stem
    i = 2
    while os.path.exists(os.path.join(kb_dir, candidate)):
        candidate = f"{stem}_{i}"
        i += 1
    return candidate


def _disk_doc_count(kb_dir: str) -> int:
    """统计本地工作目录中已收纳的文档数 (每个文档一个子目录)。"""
    if not os.path.isdir(kb_dir):
        return 0
    return sum(
        1
        for entry in os.scandir(kb_dir)
        if entry.is_dir() and not entry.name.startswith(".")
    )


def _is_kb_collection(name: str, *, default_collection: str = _DEFAULT_COLLECTION) -> bool:
    """判断 Milvus 集合名是否属于知识库 (默认库 + kb_ 前缀)。"""
    return name == default_collection or name.startswith(_KB_PREFIX)


def _list_disk_kbs(prefix: str) -> List[str]:
    """列出本地工作目录下的知识库目录 (仅 kb_*; 排除 uploads/skills 等)。"""
    if not os.path.isdir(UPLOAD_ROOT):
        return []
    return [
        entry.name
        for entry in os.scandir(UPLOAD_ROOT)
        if entry.is_dir()
        and not entry.name.startswith(".")
        and entry.name.startswith(_KB_PREFIX)
        and (not prefix or entry.name.startswith(prefix))
    ]


@router.get("/collections", response_model=CollectionsListResponse)
def list_collections(
    prefix: str = _KB_PREFIX,
    _auth: str = Depends(verify_api_key),
) -> CollectionsListResponse:
    """列出知识库集合: 默认库 literature_chunks + kb_* 集合 ∪ 本地 kb_* 工作目录。"""
    pipe = get_pipeline()
    default_coll = getattr(pipe, "_default_collection", None) or _DEFAULT_COLLECTION
    by_name: Dict[str, Dict[str, Any]] = {}

    # 1) Milvus: 只保留默认库与 kb_* (排除 skills 等无关集合)
    for c in pipe.list_collections(prefix=""):
        name = c["name"]
        if not _is_kb_collection(name, default_collection=default_coll):
            continue
        if prefix and name != default_coll and not name.startswith(prefix):
            continue
        by_name[name] = {"name": name, "row_count": c.get("row_count", 0)}

    # 2) 默认库必须始终可见 (即使用户只筛 kb_ 前缀)
    by_name.setdefault(default_coll, {"name": default_coll, "row_count": 0})

    # 3) 本地 kb_* 工作目录 (含尚未灌入的空知识库; uploads/skills 等不进列表)
    for name in _list_disk_kbs(prefix):
        by_name.setdefault(name, {"name": name, "row_count": 0})

    # 4) 补充每个库的本地文档数 + 显示名
    collections = []
    for name, info in sorted(by_name.items()):
        kb_dir = kb_workspace_dir(name)
        info["doc_count"] = _disk_doc_count(kb_dir)
        if name == default_coll:
            info["display_name"] = "默认知识库"
        else:
            fallback = name[len(_KB_PREFIX):] if name.startswith(_KB_PREFIX) else name
            info["display_name"] = _read_display_name(kb_dir, fallback)
        collections.append(CollectionInfo(**info))

    return CollectionsListResponse(collections=collections)


@router.post("/collections", response_model=CollectionInfo)
def create_collection(
    req: CreateCollectionRequest,
    _auth: str = Depends(verify_api_key),
) -> CollectionInfo:
    """新建一个空知识库: 创建本地工作目录, 立即出现在列表中。

    Milvus 集合会在首次上传 PDF 灌入时自动创建。
    支持中文名: 集合名用 ASCII slug, 中文显示名存入工作目录元数据。
    """
    display_name = (req.name or "").strip()
    name = make_collection_slug(req.name)
    kb_dir = kb_workspace_dir(name)
    os.makedirs(kb_dir, exist_ok=True)
    _write_kb_meta(kb_dir, display_name, name)
    logger.info(f"[collections] 已创建知识库 {name} (显示名: {display_name})")
    return CollectionInfo(name=name, display_name=display_name, row_count=0, doc_count=0)


@router.delete("/collections/{name}", response_model=DeleteCollectionResponse)
def delete_collection(
    name: str,
    _auth: str = Depends(verify_api_key),
) -> DeleteCollectionResponse:
    """删除一个知识库: 删 Milvus 集合 + 连带清理本地工作目录 (中间产物/原始 PDF)。"""
    if not name.startswith(_KB_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=f"仅允许删除 {_KB_PREFIX} 前缀的集合, "
                   f"默认集合 literature_chunks 不可通过此接口删除",
        )
    pipe = get_pipeline()
    deleted = pipe.drop_collection(name)

    # 连带清理本地工作目录 (中间产物 + 原始 PDF)
    kb_dir = kb_workspace_dir(name)
    if os.path.isdir(kb_dir):
        try:
            shutil.rmtree(kb_dir)
            deleted = True
            logger.info(f"[collections] 已清理本地工作目录: {kb_dir}")
        except Exception as e:
            logger.warning(f"[collections] 清理本地目录失败 {kb_dir}: {e}")

    return DeleteCollectionResponse(deleted=deleted, name=name)


def _rebuild_collection(task_id: str, collection: str, directory: str) -> Dict[str, Any]:
    """后台任务: 复用本地已落盘的向量 (knowledge_blocks_vec.json), 清空集合后
    逐字节重灌, 不重新 chunk / embed (向量不漂移, 不依赖 embedding 服务在线)。
    缺 vec.json 的文档自动回退到完整 chunk→embed→store。"""
    pipe = get_pipeline()
    task_store = get_task_store()

    def on_progress(current, total, doc_id, status):
        task_store.update_progress(task_id, current, total, doc_id)

    results = pipe.reingest_directory(
        directory,
        collection=collection,
        recreate=True,
        skip_existing=False,
        progress_callback=on_progress,
    )
    try:
        pipe.flush_collection(collection)
    except Exception as e:
        logger.warning(f"[rebuild] flush 失败 {collection}: {e}")
    success = sum(1 for r in results if r.steps and all(s.success for s in r.steps))
    return {
        "collection": collection,
        "docs": len(results),
        "success": success,
        "failed": len(results) - success,
        "total_chunks": sum(r.total_chunks for r in results),
    }


@router.post("/collections/{name}/rebuild", response_model=TaskResponse)
def rebuild_collection(
    name: str,
    _auth: str = Depends(verify_api_key),
) -> TaskResponse:
    """重建知识库 (异步): 复用本地已落盘的向量, 清空集合后逐字节重灌。

    默认复用 knowledge_blocks_vec.json, 不重新 chunk / embed, 因此重灌结果与首次
    入库一致, 且不依赖 embedding 服务在线; 缺 vec.json 的文档回退完整向量化。
    """
    if not name.startswith(_KB_PREFIX):
        raise HTTPException(status_code=400, detail=f"仅允许重建 {_KB_PREFIX} 前缀的集合")
    kb_dir = kb_workspace_dir(name)
    if not os.path.isdir(kb_dir) or _disk_doc_count(kb_dir) == 0:
        raise HTTPException(
            status_code=400,
            detail="该知识库没有本地解析产物可供重建, 请先上传 PDF 灌入",
        )
    task_store = get_task_store()
    tid = uuid.uuid4().hex[:16]
    task_store.submit(_rebuild_collection, tid, name, kb_dir, task_id=tid)
    return TaskResponse(id=tid, status="pending", created_at=time.time())
