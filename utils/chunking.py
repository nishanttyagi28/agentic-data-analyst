"""Text chunking utilities for RAG indexing."""

from __future__ import annotations

import hashlib
import re


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    metadata_prefix: str = "",
) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    if not text or not text.strip():
        return []

    text = text.strip()
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                current = para
            else:
                words = para.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= chunk_size:
                        current = f"{current} {word}".strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = word
                if current:
                    chunks.append(current)
                current = ""

    if current:
        chunks.append(current)

    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev = chunks[i - 1]
                overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
                chunk = f"{overlap_text} {chunk}".strip()
            overlapped.append(chunk)
        chunks = overlapped

    result = []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{metadata_prefix}:{i}:{chunk[:50]}".encode()).hexdigest()[:12]
        result.append({
            "id": f"{metadata_prefix}_{chunk_id}",
            "text": chunk,
            "chunk_index": i,
        })
    return result


def format_exchange_for_rag(
    question: str,
    answer: str,
    source_type: str,
    extra: str = "",
) -> str:
    parts = [
        f"[{source_type.upper()}]",
        f"Question: {question}",
        f"Answer: {answer}",
    ]
    if extra:
        parts.append(f"Details: {extra}")
    return "\n".join(parts)