"""RAG service providing document retrieval for the chatbot."""

from typing import Optional

import structlog

from backend.app.config import settings

logger = structlog.get_logger()


class RAGService:
    """Service for managing RAG indexing and querying."""

    def __init__(self):
        self._indexed = False
        self._vectorstore = None

    def ensure_indexed(self) -> None:
        """Load docs, chunk them, and index into ChromaDB if not already done."""
        if self._indexed:
            return

        try:
            from backend.app.rag.ingest import load_documents, chunk_documents
            from backend.app.rag.vectorstore import get_vectorstore

            documents = load_documents()
            if not documents:
                logger.warning("No documents found for RAG indexing")
                self._indexed = True
                return

            chunks = chunk_documents(documents)
            self._vectorstore = get_vectorstore(documents=chunks)
            self._indexed = True

            count = self.get_chunk_count()
            logger.info("RAG index ready", chunk_count=count)

        except Exception as e:
            logger.error("RAG indexing failed", error=str(e))
            self._indexed = True  # Don't retry on every request

    def query(self, question: str, top_k: Optional[int] = None) -> list[str]:
        """
        Query the vector store for relevant document chunks.

        Args:
            question: The user's question.
            top_k: Number of results to return.

        Returns:
            List of relevant text chunks.
        """
        if self._vectorstore is None:
            self.ensure_indexed()

        if self._vectorstore is None:
            return []

        k = top_k or settings.rag_top_k

        try:
            results = self._vectorstore.similarity_search(question, k=k)
            chunks = [doc.page_content for doc in results]
            logger.debug("RAG query results", question=question[:50], chunks_found=len(chunks))
            return chunks
        except Exception as e:
            logger.error("RAG query failed", error=str(e))
            return []

    def get_chunk_count(self) -> int:
        """Get the number of chunks in the vector store."""
        if self._vectorstore is None:
            return 0
        try:
            return self._vectorstore._collection.count()
        except Exception:
            return 0


# Singleton instance
rag_service = RAGService()
