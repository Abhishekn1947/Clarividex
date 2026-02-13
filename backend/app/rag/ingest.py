"""Document loading and chunking for RAG pipeline."""

from pathlib import Path
from typing import Optional

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from backend.app.config import settings

logger = structlog.get_logger()


def load_documents(docs_dir: Optional[str] = None) -> list[Document]:
    """
    Load markdown documents from the docs directory.

    Args:
        docs_dir: Path to docs directory. Defaults to project docs/.

    Returns:
        List of LangChain Document objects.
    """
    if docs_dir is None:
        # Try multiple locations (Docker vs local)
        candidates = [
            Path("/project/docs"),
            Path(__file__).parent.parent.parent.parent.parent / "docs",
        ]
        docs_path = None
        for candidate in candidates:
            if candidate.exists():
                docs_path = candidate
                break
        if docs_path is None:
            logger.warning("No docs directory found")
            return []
    else:
        docs_path = Path(docs_dir)

    documents = []
    md_files = sorted(docs_path.glob("*.md"))

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            doc = Document(
                page_content=content,
                metadata={
                    "source": md_file.name,
                    "file_path": str(md_file),
                },
            )
            documents.append(doc)
            logger.info("Loaded document", file=md_file.name, length=len(content))
        except Exception as e:
            logger.warning("Failed to load document", file=str(md_file), error=str(e))

    logger.info("Documents loaded", count=len(documents))
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into chunks for embedding.

    Args:
        documents: List of full documents.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    chunks = splitter.split_documents(documents)
    logger.info("Documents chunked", total_chunks=len(chunks))
    return chunks
