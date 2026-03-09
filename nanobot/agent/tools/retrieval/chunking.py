"""Plain-text document chunking utilities for retrieval ingestion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DocumentChunk:
    """A plain-text chunk with character-range metadata."""

    index: int
    start_char: int
    end_char: int
    text: str


class DocumentChunker:
    """Split plain text into overlapping chunks.

    The chunker is intentionally simple and deterministic:
    - no HTML/DOM parsing
    - no tokenizer dependency
    - fixed character windows with overlap

    This keeps ingestion predictable across heterogeneous document sources.
    """

    def __init__(
        self,
        max_chars: int = 1500,
        overlap_chars: int = 200,
        min_chunk_chars: int = 100,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be >= 0")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be < max_chars")
        if min_chunk_chars < 0:
            raise ValueError("min_chunk_chars must be >= 0")

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk_text(self, text: str) -> list[DocumentChunk]:
        """Chunk plain text into overlapping windows.

        Args:
            text: Input plain text.

        Returns:
            Ordered list of chunks. Chunks smaller than `min_chunk_chars`
            are dropped unless they are the only chunk.
        """
        if not text:
            return []

        n = len(text)
        if n <= self.max_chars:
            if n < self.min_chunk_chars and self.min_chunk_chars > 0:
                return []
            return [DocumentChunk(index=0, start_char=0, end_char=n, text=text)]

        step = self.max_chars - self.overlap_chars
        chunks: list[DocumentChunk] = []
        i = 0
        start = 0

        while start < n:
            end = min(start + self.max_chars, n)
            chunk_text = text[start:end]
            if len(chunk_text) >= self.min_chunk_chars or not chunks:
                chunks.append(
                    DocumentChunk(
                        index=i,
                        start_char=start,
                        end_char=end,
                        text=chunk_text,
                    )
                )
                i += 1

            if end >= n:
                break
            start += step

        return chunks
