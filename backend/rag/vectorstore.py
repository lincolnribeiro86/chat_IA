"""Chroma vector store for RAG retrieval."""
from __future__ import annotations
import logging
from config import settings

logger = logging.getLogger(__name__)


def _get_embeddings():
    """Return best available embedding model."""
    if settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=settings.openai_api_key)
    # Fallback: Ollama nomic-embed-text
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=settings.ollama_base_url,
        )
    except Exception:
        raise RuntimeError(
            "No embedding model available. Set OPENAI_API_KEY or run Ollama with nomic-embed-text."
        )


def build_retriever(chunks: list[dict], session_id: str):
    """Index chunks in an in-memory Chroma collection and return a retriever."""
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    embeddings = _get_embeddings()
    docs = [
        Document(
            page_content=c["content"],
            metadata={"source": c.get("source", ""), "chunk": c.get("chunk_index", 0)},
        )
        for c in chunks
    ]
    # Use in-memory (no persist_directory) per-session store
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=f"session_{session_id[:8]}",
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.rag_top_k},
    )


def retrieve(retriever, query: str) -> str:
    """Return formatted retrieved context for the query."""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return ""
        parts = []
        for d in docs:
            src = d.metadata.get("source", "")
            parts.append(f"[Fonte: {src}]\n{d.page_content}" if src else d.page_content)
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        logger.error(f"RAG retrieve error: {e}")
        return ""
