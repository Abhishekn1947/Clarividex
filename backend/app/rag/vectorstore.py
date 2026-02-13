"""ChromaDB vector store for RAG pipeline."""

from typing import Optional

import structlog
from langchain_core.documents import Document

from backend.app.config import settings
from backend.app.rag.embeddings import get_embeddings

logger = structlog.get_logger()

_vectorstore_instance = None


def get_vectorstore(documents: Optional[list[Document]] = None):
    """
    Get or create singleton ChromaDB vector store.

    If documents are provided and the collection is empty, they will be indexed.

    Args:
        documents: Optional documents to index on first creation.

    Returns:
        LangChain Chroma vector store instance.
    """
    global _vectorstore_instance
    if _vectorstore_instance is not None:
        return _vectorstore_instance

    from langchain_chroma import Chroma

    embeddings = get_embeddings()

    _vectorstore_instance = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )

    # If we have documents and the collection is empty, index them
    if documents and _vectorstore_instance._collection.count() == 0:
        logger.info("Indexing documents into ChromaDB", count=len(documents))
        _vectorstore_instance.add_documents(documents)
        logger.info("Documents indexed", count=len(documents))

    return _vectorstore_instance
