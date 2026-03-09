"""Object-store cache for canonical source documents.

This module stores original document bytes in object storage and returns a
stable pointer for retrieval payloads. Ingestion never relies on remote source
availability after this write succeeds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ObjectRef:
    """Provider-neutral object pointer persisted in retrieval payloads."""

    provider: str
    bucket: str
    key: str
    uri: str


@dataclass(slots=True)
class PutResult:
    """Result of put-if-absent write."""

    created: bool
    object_ref: ObjectRef


class _S3LikeClient(Protocol):
    """Minimal S3 client surface used by ObjectStoreCache."""

    def head_object(self, *, Bucket: str, Key: str) -> dict: ...

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> dict: ...

    def get_object(self, *, Bucket: str, Key: str) -> dict: ...


class ObjectStoreCache:
    """Provider-neutral object-store facade (AWS S3 currently implemented).

    Cloud-provider differences are intentionally isolated to constructor
    initialization. Runtime methods use a neutral interface.
    """

    def __init__(
        self,
        provider: str,
        bucket: str,
        prefix: str = "documents",
        s3_region: str | None = None,
        s3_client: _S3LikeClient | None = None,
    ) -> None:
        if provider != "s3":
            raise ValueError(f"Unsupported object store provider: {provider}")
        if not bucket:
            raise ValueError("bucket is required")

        self.provider = provider
        self.bucket = bucket
        self.prefix = prefix.strip("/")

        if s3_client is not None:
            self._client = s3_client
        else:
            import boto3

            kwargs = {"region_name": s3_region} if s3_region else {}
            self._client = boto3.client("s3", **kwargs)

    @staticmethod
    def sha256_hex(data: bytes) -> str:
        """Compute stable SHA-256 hex hash for object identity."""
        return hashlib.sha256(data).hexdigest()

    def build_object_key(self, content_hash: str, source_ext: str | None = None) -> str:
        """Build deterministic object key from content hash and optional extension."""
        ext = (source_ext or "").strip().lstrip(".")
        filename = f"{content_hash}.{ext}" if ext else content_hash
        if self.prefix:
            return f"{self.prefix}/{filename}"
        return filename

    def build_object_ref(self, key: str) -> ObjectRef:
        """Create provider-neutral pointer for payload storage."""
        return ObjectRef(
            provider=self.provider,
            bucket=self.bucket,
            key=key,
            uri=f"s3://{self.bucket}/{key}",
        )

    def exists(self, key: str) -> bool:
        """Check object existence via HEAD."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def put_if_absent(
        self,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> PutResult:
        """Write object iff absent, returning whether it was newly created."""
        ref = self.build_object_ref(key)
        if self.exists(key):
            return PutResult(created=False, object_ref=ref)

        kwargs = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": data,
        }
        if content_type:
            kwargs["ContentType"] = content_type
        self._client.put_object(**kwargs)
        return PutResult(created=True, object_ref=ref)

    def get(self, key: str) -> bytes:
        """Read object bytes."""
        resp = self._client.get_object(Bucket=self.bucket, Key=key)
        body = resp.get("Body")
        if body is None:
            return b""
        return body.read()
