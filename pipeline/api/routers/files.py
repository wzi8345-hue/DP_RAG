"""文件: 上传 / 原始 PDF 取回 / chunk 定位框查询"""

from __future__ import annotations

import glob
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..deps import get_pipeline, verify_api_key
from ..models import FileUploadResponse

router = APIRouter()

_UPLOAD_DIR = os.environ.get(
    "UPLOAD_DIR",
    os.path.join(os.getcwd(), "uploads"),
)


def _safe_segment(value: str) -> bool:
    """拒绝包含路径分隔符 / 上跳的片段, 防目录穿越。"""
    if not value:
        return False
    if os.sep in value or ".." in value:
        return False
    if os.altsep and os.altsep in value:
        return False
    return True


def _resolve_pdf_path(collection: str, doc_id: str) -> Optional[str]:
    """定位某文献的原始 PDF: ``<UPLOAD_DIR>/<collection>/<doc_id>/<doc_id>.pdf``。

    找不到同名 PDF 时回退到该目录下任意 *.pdf。返回 None 表示不存在或非法路径。
    """
    if not (_safe_segment(collection) and _safe_segment(doc_id)):
        return None
    doc_dir = os.path.join(_UPLOAD_DIR, collection, doc_id)
    candidate = os.path.join(doc_dir, f"{doc_id}.pdf")
    chosen: Optional[str] = candidate if os.path.isfile(candidate) else None
    if chosen is None:
        pdfs = sorted(glob.glob(os.path.join(doc_dir, "*.pdf")))
        chosen = pdfs[0] if pdfs else None
    if chosen is None:
        return None
    real = os.path.realpath(chosen)
    root = os.path.realpath(_UPLOAD_DIR)
    if real != root and not real.startswith(root + os.sep):
        return None
    return real


@router.post("/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile,
    _auth: str = Depends(verify_api_key),
) -> FileUploadResponse:
    """上传 PDF 文件, 返回 file_id (用于后续灌入)。"""
    os.makedirs(_UPLOAD_DIR, exist_ok=True)

    file_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename or ".pdf")[1]
    save_name = f"{file_id}{ext}"
    save_path = os.path.join(_UPLOAD_DIR, save_name)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    return FileUploadResponse(
        file_id=file_id,
        filename=file.filename or save_name,
        size_bytes=len(content),
    )


@router.get("/files/pdf")
def get_pdf(
    collection: str,
    doc_id: str,
    _auth: str = Depends(verify_api_key),
) -> FileResponse:
    """返回某文献的原始 PDF, 供前端 PDF 阅读器内嵌渲染。"""
    path = _resolve_pdf_path(collection, doc_id)
    if not path:
        raise HTTPException(status_code=404, detail="PDF 不存在")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"{doc_id}.pdf",
        headers={"Content-Disposition": f'inline; filename="{doc_id}.pdf"'},
    )


@router.get("/files/chunk_bbox")
def get_chunk_bbox(
    collection: str,
    doc_id: str,
    chunk_id: str,
    _auth: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """按 (doc_id, chunk_id) 取该 chunk 在 PDF 中的定位框 (页内归一化坐标)。

    旧集合无 bboxes 字段时返回空列表 (前端只跳页不画框)。
    """
    pipe = get_pipeline()
    try:
        return pipe.get_chunk_bboxes(collection, doc_id, chunk_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询定位框失败: {e}")
