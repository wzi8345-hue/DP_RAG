"""文件上传: POST /api/v1/files/upload"""

from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from ..authz import require_read
from ..deps import AuthContext, require_auth
from ..models import FileUploadResponse, PdfUrlRequest, PdfUrlResponse
from ...clients import object_store
from ...db import repo

router = APIRouter()

_UPLOAD_DIR = os.environ.get(
    "UPLOAD_DIR",
    os.path.join(os.getcwd(), "uploads"),
)


@router.post("/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile,
    _auth: AuthContext = Depends(require_auth),
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


@router.post("/documents/pdf-url", response_model=PdfUrlResponse)
def document_pdf_url(
    req: PdfUrlRequest,
    auth: AuthContext = Depends(require_auth),
) -> PdfUrlResponse:
    if not repo.available():
        raise HTTPException(status_code=503, detail="DATABASE_URL 未配置")
    if not object_store.configured():
        raise HTTPException(status_code=503, detail="对象存储未配置")

    doc = repo.get_document(req.collection, req.doc_id) if req.collection else repo.find_document_by_doc_id(req.doc_id, auth)
    if doc is None:
        raise HTTPException(status_code=404, detail="文献不存在")
    collection = repo.get_collection(doc.collection_name)
    if collection is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    require_read(auth, collection)
    if not doc.pdf_object_key:
        raise HTTPException(status_code=404, detail="该文献没有 PDF 对象")

    expires = max(60, min(req.expires_in, 3600))
    url = object_store.get_object_store().presign_get_url(doc.pdf_object_key, expires_in=expires)
    return PdfUrlResponse(
        url=url,
        doc_id=doc.doc_id,
        collection=doc.collection_name,
        expires_in=expires,
    )
