"""Retrieval tools and backend clients.

Design choices recovered from prior implementation:
- `content_hash` is the sole document identity.
- Pointer-first storage: payload stores object URI/key, not full document body.
- Hierarchical retrieval:
  1) search top documents in shared `documents` collection,
  2) search chunks in per-document collection named by `content_hash`.
- Object store is mandatory for ingestion.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.retrieval.chunking import DocumentChunker
from nanobot.agent.tools.retrieval.object_store import ObjectStoreCache


@dataclass(slots=True)
class RetrievedDocument:
    """Top-level document retrieval result."""

    content_hash: str
    score: float
    source_uri: str
    source_type: str
    object_uri: str
    object_key: str
    chunks_collection: str


@dataclass(slots=True)
class RetrievedChunk:
    """Chunk retrieval result for a selected document."""

    content_hash: str
    score: float
    text: str
    start_char: int
    end_char: int
    source_uri: str
    source_type: str


class RetrievalBackend(Protocol):
    """Backend contract for ingestion/retrieval tools."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def upsert_document(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None: ...

    async def upsert_chunks(
        self,
        collection: str,
        points: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None: ...

    async def search_documents(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        min_score: float | None,
        source_type: str | None,
    ) -> list[RetrievedDocument]: ...

    async def search_chunks(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        min_score: float | None,
    ) -> list[RetrievedChunk]: ...


class PerplexityEmbedder:
    """Perplexity contextual embeddings client.

    Request format intentionally uses contextual shape:
      {"model": ..., "input": [[text1], [text2], ...]}
    """

    def __init__(
        self,
        api_key: str,
        model: str = "pplx-embed-context-v1-4b",
        timeout_s: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("Perplexity API key is required")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": [[t] for t in texts],
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                "https://api.perplexity.ai/embeddings",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])

        vectors = [row.get("embedding", []) for row in data]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding response count mismatch: expected {len(texts)}, got {len(vectors)}"
            )
        return vectors


class QdrantRetrievalClient:
    """Qdrant-backed retrieval backend."""

    def __init__(
        self,
        embedder: PerplexityEmbedder,
        qdrant_url: str,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.embedder = embedder
        self.qdrant_url = qdrant_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["api-key"] = self.api_key
        return h

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.embedder.embed_texts(texts)

    async def _upsert_points(self, collection: str, points: list[dict[str, Any]]) -> None:
        url = f"{self.qdrant_url}/collections/{collection}/points?wait=true"
        payload = {"points": points}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.put(url, headers=self._headers(), json=payload)
            resp.raise_for_status()

    async def upsert_document(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        await self._upsert_points(
            collection=collection,
            points=[{"id": point_id, "vector": vector, "payload": payload}],
        )

    async def upsert_chunks(
        self,
        collection: str,
        points: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        q_points = [
            {"id": pid, "vector": vec, "payload": payload}
            for pid, vec, payload in points
        ]
        if q_points:
            await self._upsert_points(collection=collection, points=q_points)

    async def _search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        score_threshold: float | None,
        filt: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        url = f"{self.qdrant_url}/collections/{collection}/points/search"
        payload: dict[str, Any] = {
            "vector": vector,
            "limit": top_k,
            "with_payload": True,
        }
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if filt is not None:
            payload["filter"] = filt

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(url, headers=self._headers(), json=payload)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            return resp.json().get("result", [])

    async def search_documents(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        min_score: float | None,
        source_type: str | None,
    ) -> list[RetrievedDocument]:
        q_filter = None
        if source_type:
            q_filter = {
                "must": [
                    {
                        "key": "sourceType",
                        "match": {"value": source_type},
                    }
                ]
            }

        rows = await self._search(
            collection=collection,
            vector=vector,
            top_k=top_k,
            score_threshold=min_score,
            filt=q_filter,
        )

        out: list[RetrievedDocument] = []
        for row in rows:
            payload = row.get("payload", {})
            out.append(
                RetrievedDocument(
                    content_hash=payload.get("contentHash", ""),
                    score=float(row.get("score", 0.0)),
                    source_uri=payload.get("sourceUri", ""),
                    source_type=payload.get("sourceType", ""),
                    object_uri=payload.get("objectUri", ""),
                    object_key=payload.get("objectKey", ""),
                    chunks_collection=payload.get("chunksCollection", ""),
                )
            )
        return out

    async def search_chunks(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        min_score: float | None,
    ) -> list[RetrievedChunk]:
        rows = await self._search(
            collection=collection,
            vector=vector,
            top_k=top_k,
            score_threshold=min_score,
        )
        out: list[RetrievedChunk] = []
        for row in rows:
            payload = row.get("payload", {})
            out.append(
                RetrievedChunk(
                    content_hash=payload.get("contentHash", ""),
                    score=float(row.get("score", 0.0)),
                    text=payload.get("text", ""),
                    start_char=int(payload.get("startChar", 0)),
                    end_char=int(payload.get("endChar", 0)),
                    source_uri=payload.get("sourceUri", ""),
                    source_type=payload.get("sourceType", ""),
                )
            )
        return out


class IngestDocumentTool(Tool):
    """Ingest a document into hierarchical retrieval index.

    Flow:
      1) load bytes from local path or URL
      2) compute `content_hash` from full bytes
      3) persist original bytes to object store (mandatory)
      4) chunk plain text and embed
      5) upsert doc vector to shared docs collection
      6) upsert chunk vectors to per-doc collection (`content_hash`)
    """

    name = "ingest_document"
    description = "Ingest one document into retrieval index with object-store persistence."
    parameters = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Local file path or URL"},
            "sourceType": {
                "type": "string",
                "enum": ["local", "url"],
                "description": "Source type",
            },
            "forceReindex": {
                "type": "boolean",
                "description": "Force overwrite/reindex existing document",
                "default": False,
            },
        },
        "required": ["source", "sourceType"],
    }

    def __init__(
        self,
        retrieval: RetrievalBackend,
        object_store: ObjectStoreCache,
        documents_collection: str = "documents",
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.object_store = object_store
        self.documents_collection = documents_collection
        self.chunker = chunker or DocumentChunker()

    async def execute(
        self,
        source: str,
        sourceType: str,
        forceReindex: bool = False,
        **kwargs: Any,
    ) -> str:
        try:
            data, content_type = await self._load_source(source=source, source_type=sourceType)
            if not data:
                return json.dumps({"error": "empty source content"}, ensure_ascii=False)

            content_hash = hashlib.sha256(data).hexdigest()
            ext = self._guess_extension(source, content_type)
            object_key = self.object_store.build_object_key(content_hash, ext)
            put_result = self.object_store.put_if_absent(
                key=object_key,
                data=data,
                content_type=content_type,
            )
            object_ref = put_result.object_ref

            text = data.decode("utf-8", errors="replace")
            chunks = self.chunker.chunk_text(text)
            if not chunks:
                return json.dumps(
                    {
                        "error": "no chunks produced",
                        "contentHash": content_hash,
                    },
                    ensure_ascii=False,
                )

            doc_embed_text = text[:4000]
            vectors = await self.retrieval.embed_texts([doc_embed_text] + [c.text for c in chunks])
            doc_vector = vectors[0]
            chunk_vectors = vectors[1:]

            chunks_collection = content_hash

            doc_payload = {
                "contentHash": content_hash,
                "sourceUri": source,
                "sourceType": sourceType,
                "objectProvider": object_ref.provider,
                "objectBucket": object_ref.bucket,
                "objectKey": object_ref.key,
                "objectUri": object_ref.uri,
                "chunksCollection": chunks_collection,
                "chunkCount": len(chunks),
                "charCount": len(text),
            }
            await self.retrieval.upsert_document(
                collection=self.documents_collection,
                point_id=content_hash,
                vector=doc_vector,
                payload=doc_payload,
            )

            chunk_points: list[tuple[str, list[float], dict[str, Any]]] = []
            for chunk, vec in zip(chunks, chunk_vectors, strict=False):
                chunk_id = f"{content_hash}:{chunk.index}"
                payload = {
                    "contentHash": content_hash,
                    "chunkIndex": chunk.index,
                    "startChar": chunk.start_char,
                    "endChar": chunk.end_char,
                    "text": chunk.text,
                    "sourceUri": source,
                    "sourceType": sourceType,
                }
                chunk_points.append((chunk_id, vec, payload))

            await self.retrieval.upsert_chunks(
                collection=chunks_collection,
                points=chunk_points,
            )

            return json.dumps(
                {
                    "ok": True,
                    "contentHash": content_hash,
                    "sourceUri": source,
                    "sourceType": sourceType,
                    "documentsCollection": self.documents_collection,
                    "chunksCollection": chunks_collection,
                    "chunkCount": len(chunks),
                    "charCount": len(text),
                    "embeddedChars": len(doc_embed_text),
                    "object": {
                        "provider": object_ref.provider,
                        "bucket": object_ref.bucket,
                        "key": object_ref.key,
                        "uri": object_ref.uri,
                        "created": put_result.created,
                    },
                    "forceReindex": forceReindex,
                    "pointId": content_hash,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _load_source(self, source: str, source_type: str) -> tuple[bytes, str | None]:
        if source_type == "local":
            path = Path(source).expanduser().resolve()
            data = path.read_bytes()
            ctype, _ = mimetypes.guess_type(str(path))
            return data, ctype

        if source_type == "url":
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                resp = await client.get(source)
                resp.raise_for_status()
                return resp.content, resp.headers.get("content-type")

        raise ValueError(f"unsupported sourceType: {source_type}")

    @staticmethod
    def _guess_extension(source: str, content_type: str | None) -> str | None:
        if content_type:
            ctype = content_type.split(";", 1)[0].strip().lower()
            ext = mimetypes.guess_extension(ctype)
            if ext:
                return ext.lstrip(".")

        parsed = urlparse(source)
        tail = Path(parsed.path if parsed.scheme else source).suffix
        if tail:
            return tail.lstrip(".")
        return None


class RetrieveDocumentsTool(Tool):
    """Retrieve relevant documents/chunks via hierarchical search."""

    name = "retrive_documents"  # intentional spelling preserved
    description = "Retrieve documents by query with optional score/source filtering."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "topK": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 5,
            },
            "minScore": {
                "type": "number",
                "description": "Minimum similarity score",
            },
            "sourceType": {
                "type": "string",
                "enum": ["local", "url"],
                "description": "Filter by source type",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        retrieval: RetrievalBackend,
        documents_collection: str = "documents",
    ) -> None:
        self.retrieval = retrieval
        self.documents_collection = documents_collection

    async def execute(
        self,
        query: str,
        topK: int = 5,
        minScore: float | None = None,
        sourceType: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            query_vec = (await self.retrieval.embed_texts([query]))[0]
            docs = await self.retrieval.search_documents(
                collection=self.documents_collection,
                vector=query_vec,
                top_k=topK,
                min_score=minScore,
                source_type=sourceType,
            )

            chunk_results: list[dict[str, Any]] = []
            for doc in docs:
                if not doc.chunks_collection:
                    continue
                chunks = await self.retrieval.search_chunks(
                    collection=doc.chunks_collection,
                    vector=query_vec,
                    top_k=topK,
                    min_score=minScore,
                )
                for chunk in chunks:
                    chunk_results.append(
                        {
                            "contentHash": chunk.content_hash,
                            "score": chunk.score,
                            "text": chunk.text,
                            "startChar": chunk.start_char,
                            "endChar": chunk.end_char,
                            "sourceUri": chunk.source_uri,
                            "sourceType": chunk.source_type,
                            "chunksCollection": doc.chunks_collection,
                        }
                    )

            doc_results = [
                {
                    "contentHash": d.content_hash,
                    "score": d.score,
                    "sourceUri": d.source_uri,
                    "sourceType": d.source_type,
                    "objectUri": d.object_uri,
                    "objectKey": d.object_key,
                    "chunksCollection": d.chunks_collection,
                }
                for d in docs
            ]

            return json.dumps(
                {
                    "query": query,
                    "documentsCollection": self.documents_collection,
                    "topK": topK,
                    "minScore": minScore,
                    "sourceType": sourceType,
                    "documents": doc_results,
                    "chunks": chunk_results,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)


def build_retrieval_client_from_config(cfg: Any) -> QdrantRetrievalClient:
    """Construct Qdrant retrieval client from config sub-tree.

    `cfg` is expected to be `tools.retrieval` config object.
    """
    api_key = cfg.perplexity.api_key or os.environ.get("PERPLEXITY_API_KEY", "")
    embedder = PerplexityEmbedder(
        api_key=api_key,
        model=cfg.perplexity.model,
    )
    return QdrantRetrievalClient(
        embedder=embedder,
        qdrant_url=cfg.qdrant.url,
        api_key=cfg.qdrant.api_key or None,
    )
