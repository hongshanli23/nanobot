from __future__ import annotations

import pytest

from nanobot.agent.tools.retrieval import ObjectStoreCache


class _Body:
    def __init__(self, b: bytes):
        self._b = b

    def read(self) -> bytes:
        return self._b


class _S3:
    def __init__(self):
        self.kv: dict[str, bytes] = {}

    def head_object(self, *, Bucket: str, Key: str):
        if Key not in self.kv:
            raise RuntimeError("missing")
        return {"ok": True}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType=None):
        self.kv[Key] = Body
        return {"ok": True}

    def get_object(self, *, Bucket: str, Key: str):
        return {"Body": _Body(self.kv[Key])}


def test_object_store_requires_bucket() -> None:
    with pytest.raises(ValueError):
        ObjectStoreCache(provider="s3", bucket="")


def test_object_store_provider_validation() -> None:
    with pytest.raises(ValueError):
        ObjectStoreCache(provider="gcs", bucket="b")


def test_build_object_ref_and_key() -> None:
    cache = ObjectStoreCache(provider="s3", bucket="b", prefix="documents", s3_client=_S3())
    key = cache.build_object_key("abc123", "pdf")
    ref = cache.build_object_ref(key)
    assert key == "documents/abc123.pdf"
    assert ref.uri == "s3://b/documents/abc123.pdf"


def test_put_if_absent_exists_get_roundtrip() -> None:
    s3 = _S3()
    cache = ObjectStoreCache(provider="s3", bucket="b", prefix="p", s3_client=s3)
    key = cache.build_object_key("h", "txt")

    r1 = cache.put_if_absent(key=key, data=b"hello")
    r2 = cache.put_if_absent(key=key, data=b"world")

    assert r1.created is True
    assert r2.created is False
    assert cache.exists(key) is True
    assert cache.get(key) == b"hello"
