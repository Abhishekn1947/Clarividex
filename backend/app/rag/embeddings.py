"""Singleton embedding model for RAG pipeline."""

from typing import Optional

import structlog

logger = structlog.get_logger()

_embeddings_instance = None


def get_embeddings():
    """Get or create singleton HuggingFace embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from backend.app.config import settings

        logger.info("Loading embedding model", model=settings.embedding_model_name)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded")
    return _embeddings_instance
