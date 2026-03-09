from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.tools.retrieval import (
    DocumentChunker,
    IngestDocumentTool,
    RetrieveDocumentsTool,
    RetrievedChunk,
    RetrievedDocument,
)


def test_document_chunker_validation() -> None:
    with pytest.raises(ValueError):
        DocumentChunker(max_chars=0)
    with pytest.raises(ValueError):
        DocumentChunker(max_chars=100, overlap_chars=100)
    with pytest.raises(ValueError):
        DocumentChunker(max_chars=100, overlap_chars=-1)
    with pytest.raises(ValueError):
        DocumentChunker(max_chars=100, min_chunk_chars=-1)


def test_document_chunker_overlap_and_ranges() -> None:
    text = "x" * 1000
    chunker = DocumentChunker(max_chars=300, overlap_chars=50, min_chunk_chars=10)
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 3
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 300
    assert chunks[1].start_char == 250


class _FakeBody:
    def __init__(self, b: bytes) -> None:
        self._b = b

    def read(self) -> bytes:
        return self._b


class _FakeS3:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self.store:
            raise RuntimeError("not found")
        return {"Key": Key}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> dict:
        self.store[Key] = Body
        return {"ok": True}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        return {"Body": _FakeBody(self.store[Key])}


def test_object_store_cache_put_if_absent() -> None:
    from nanobot.agent.tools.retrieval import ObjectStoreCache

    s3 = _FakeS3()
    cache = ObjectStoreCache(
        provider="s3",
        bucket="b",
        prefix="docs",
        s3_client=s3,
    )
    key = cache.build_object_key("abc", "txt")
    first = cache.put_if_absent(key=key, data=b"hello", content_type="text/plain")
    second = cache.put_if_absent(key=key, data=b"world", content_type="text/plain")

    assert first.created is True
    assert second.created is False
    assert cache.get(key) == b"hello"


@pytest.mark.asyncio
async def test_ingest_document_tool_happy_path_local(tmp_path: Path) -> None:
    text = "hello world " * 200
    src = tmp_path / "doc.txt"
    src.write_text(text, encoding="utf-8")

    retrieval = MagicMock()
    retrieval.embed_texts = AsyncMock(return_value=[[0.1, 0.2]] + [[0.2, 0.3]] * 10)
    retrieval.upsert_document = AsyncMock(return_value=None)
    retrieval.upsert_chunks = AsyncMock(return_value=None)

    from nanobot.agent.tools.retrieval import ObjectStoreCache

    tool = IngestDocumentTool(
        retrieval=retrieval,
        object_store=ObjectStoreCache(provider="s3", bucket="b", s3_client=_FakeS3()),
        documents_collection="documents",
        chunker=DocumentChunker(max_chars=300, overlap_chars=50, min_chunk_chars=10),
    )

    out = await tool.execute(source=str(src), sourceType="local", forceReindex=False)
    payload = json.loads(out)

    assert payload["ok"] is True
    assert payload["sourceType"] == "local"
    assert payload["documentsCollection"] == "documents"
    assert payload["chunksCollection"] == payload["contentHash"]
    assert payload["pointId"] == payload["contentHash"]
    assert payload["chunkCount"] >= 1

    retrieval.upsert_document.assert_awaited_once()
    retrieval.upsert_chunks.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieve_documents_tool_hierarchical_search() -> None:
    retrieval = MagicMock()
    retrieval.embed_texts = AsyncMock(return_value=[[0.1, 0.2]])
    retrieval.search_documents = AsyncMock(
        return_value=[
            RetrievedDocument(
                content_hash="h1",
                score=0.9,
                source_uri="file:///a",
                source_type="local",
                object_uri="s3://b/docs/h1.txt",
                object_key="docs/h1.txt",
                chunks_collection="h1",
            )
        ]
    )
    retrieval.search_chunks = AsyncMock(
        return_value=[
            RetrievedChunk(
                content_hash="h1",
                score=0.8,
                text="chunk text",
                start_char=0,
                end_char=20,
                source_uri="file:///a",
                source_type="local",
            )
        ]
    )

    tool = RetrieveDocumentsTool(retrieval=retrieval, documents_collection="documents")
    out = await tool.execute(query="what is this", topK=3, minScore=0.2, sourceType="local")
    payload = json.loads(out)

    assert payload["query"] == "what is this"
    assert payload["documentsCollection"] == "documents"
    assert len(payload["documents"]) == 1
    assert len(payload["chunks"]) == 1
    assert payload["documents"][0]["contentHash"] == "h1"
    assert payload["chunks"][0]["chunksCollection"] == "h1"


@pytest.mark.asyncio
async def test_agentloop_registers_retrieval_tools_when_enabled(tmp_path: Path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import RetrievalToolsConfig

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    cfg = RetrievalToolsConfig()
    cfg.enabled = True
    cfg.perplexity.api_key = "test-key"
    cfg.object_store.bucket = "thinking-tokens-object-store"

    with patch("nanobot.agent.loop.build_retrieval_client_from_config") as mock_build, patch(
        "nanobot.agent.loop.ObjectStoreCache"
    ) as mock_store:
        mock_build.return_value = MagicMock()
        mock_store.return_value = MagicMock()

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            retrieval_config=cfg,
        )

    assert loop.tools.has("ingest_document")
    assert loop.tools.has("retrive_documents")
