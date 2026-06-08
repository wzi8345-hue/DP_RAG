"""文件上传: POST /api/v1/files/upload"""

from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, UploadFile

from ..deps import get_pipeline, verify_api_key
from ..models import FileUploadResponse

router = APIRouter()

_UPLOAD_DIR = os.environ.get(
    "UPLOAD_DIR",
    os.path.join(os.getcwd(), "uploads"),
)


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
