"""Document chunking for RAG."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings


def chunk_text(text: str, source: str = "") -> list[dict]:
    """Split text into chunks. Returns list of {content, source, chunk_index}."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return [
        {"content": c, "source": source, "chunk_index": i}
        for i, c in enumerate(chunks)
    ]
