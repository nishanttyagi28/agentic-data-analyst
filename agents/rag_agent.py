"""RAG Agent — chunk, embed, retrieve, and answer from ChromaDB."""

from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.config import Settings

from agents.llm_client import chat_completion
from utils.chunking import chunk_text, format_exchange_for_rag

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection_name(session_id: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in session_id)
    return f"session_{safe}"


class RAGAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.client = get_chroma_client()
        self.collection_name = get_collection_name(session_id)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"session_id": session_id},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embedder = get_embedder()
        return embedder.encode(texts, show_progress_bar=False).tolist()

    def index_text(
        self,
        text: str,
        source_type: str,
        metadata: dict | None = None,
    ) -> int:
        """Chunk and index text. Returns number of chunks indexed."""
        if not text or not text.strip():
            return 0

        chunks = chunk_text(text, metadata_prefix=source_type)
        if not chunks:
            return 0

        ids = [f"{self.session_id}_{c['id']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        embeddings = self._embed(documents)
        metadatas = [
            {
                "source_type": source_type,
                "session_id": self.session_id,
                "chunk_index": c["chunk_index"],
                **(metadata or {}),
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def index_exchange(
        self,
        question: str,
        answer: str,
        source_type: str,
        extra: str = "",
    ) -> int:
        text = format_exchange_for_rag(question, answer, source_type, extra)
        return self.index_text(text, source_type, {"question": question[:200]})

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []

        query_embedding = self._embed([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else None
                chunks.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "source_type": meta.get("source_type", "unknown"),
                })
        return chunks

    def answer(self, question: str, top_k: int = 5) -> dict[str, Any]:
        chunks = self.retrieve(question, top_k)
        if not chunks:
            return {
                "success": False,
                "error": "No indexed analysis found. Run SQL queries or ML analysis first.",
                "agent": "rag",
            }

        context_parts = []
        citations = []
        for i, chunk in enumerate(chunks):
            source = chunk["source_type"]
            context_parts.append(f"[Source {i+1} - {source}]\n{chunk['text']}")
            citations.append({
                "index": i + 1,
                "source_type": source,
                "excerpt": chunk["text"][:200] + ("..." if len(chunk["text"]) > 200 else ""),
            })

        context = "\n\n".join(context_parts)
        system_prompt = """You are a data analyst assistant. Answer the user's question using ONLY the provided context from prior analyses.
Cite which source(s) you used (e.g., "Based on Source 1 (ML analysis)...").
If the context doesn't contain enough information, say so clearly."""

        user_prompt = f"""Context from prior analyses:
{context}

User question: {question}

Provide a grounded answer citing relevant sources."""

        answer_text, err = chat_completion([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        if err:
            return {"success": False, "error": err, "agent": "rag", "citations": citations}

        return {
            "success": True,
            "agent": "rag",
            "question": question,
            "answer": answer_text or "Unable to generate answer.",
            "citations": citations,
            "chunks_used": len(chunks),
            "summary_for_rag": f"RAG Q: {question}\nA: {answer_text}",
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "collection": self.collection_name,
            "chunk_count": self.collection.count(),
        }