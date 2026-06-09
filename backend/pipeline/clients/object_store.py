"""S3-compatible object storage client (RustFS/MinIO/Aliyun OSS compatible)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectStoreSettings:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    region: str = "us-east-1"
    public_endpoint: str | None = None


def load_object_store_settings() -> ObjectStoreSettings | None:
    """Load object-store settings from env; return None when not configured."""
    endpoint = os.environ.get("OBJECT_STORE_ENDPOINT", "").strip()
    access_key = os.environ.get("OBJECT_STORE_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("OBJECT_STORE_SECRET_KEY", "").strip()
    bucket = os.environ.get("OBJECT_STORE_BUCKET", "").strip()
    if not (endpoint and access_key and secret_key and bucket):
        return None
    return ObjectStoreSettings(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region=os.environ.get("OBJECT_STORE_REGION", "us-east-1").strip() or "us-east-1",
        public_endpoint=os.environ.get("OBJECT_STORE_PUBLIC_ENDPOINT", "").strip() or None,
    )


class ObjectStoreClient:
    """Small wrapper around boto3 with path-style S3 URLs for RustFS."""

    def __init__(self, settings: ObjectStoreSettings | None = None) -> None:
        settings = settings or load_object_store_settings()
        if settings is None:
            raise RuntimeError("Object store is not configured")
        self.settings = settings
        import boto3

        self._client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key,
            region_name=settings.region,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

    @property
    def bucket(self) -> str:
        return self.settings.bucket

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
            return
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchBucket"}:
                raise
        self._client.create_bucket(Bucket=self.bucket)
        logger.info("[object-store] created bucket %s", self.bucket)

    def upload_file(self, path: str | Path, key: str, *, content_type: str | None = None) -> str:
        self.ensure_bucket()
        extra_args = {"ContentType": content_type} if content_type else None
        self._client.upload_file(str(path), self.bucket, key, ExtraArgs=extra_args or {})
        return key

    def upload_bytes(self, data: bytes, key: str, *, content_type: str | None = None) -> str:
        self.ensure_bucket()
        kwargs = {"ContentType": content_type} if content_type else {}
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data, **kwargs)
        return key

    def upload_fileobj(self, fileobj: BinaryIO, key: str, *, content_type: str | None = None) -> str:
        self.ensure_bucket()
        extra_args = {"ContentType": content_type} if content_type else None
        self._client.upload_fileobj(fileobj, self.bucket, key, ExtraArgs=extra_args or {})
        return key

    def copy_object(self, source_key: str, target_key: str) -> str:
        self.ensure_bucket()
        self._client.copy_object(
            Bucket=self.bucket,
            CopySource={"Bucket": self.bucket, "Key": source_key},
            Key=target_key,
        )
        return target_key

    def object_exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def presign_get_url(self, key: str, *, expires_in: int = 900) -> str:
        url = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        if self.settings.public_endpoint:
            return url.replace(self.settings.endpoint.rstrip("/"), self.settings.public_endpoint.rstrip("/"), 1)
        return url


_client: ObjectStoreClient | None = None


def configured() -> bool:
    return load_object_store_settings() is not None


def get_object_store() -> ObjectStoreClient:
    global _client
    if _client is None:
        _client = ObjectStoreClient()
    return _client


def document_prefix(collection: str, doc_id: str) -> str:
    return f"{collection.strip('/')}/{doc_id.strip('/')}"


def pdf_object_key(collection: str, doc_id: str) -> str:
    return f"{document_prefix(collection, doc_id)}/source.pdf"
