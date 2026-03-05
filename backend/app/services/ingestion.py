import os
import uuid
import logging
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

from app.core.config import get_settings
from app.core.vector_store import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _load_pdf(file_path: str) -> List[Document]:
    """Load a PDF and return LangChain Documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def _split_documents(docs: List[Document]) -> List[Document]:
    """Chunk documents with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    """
    Save, load, chunk, and embed a research paper PDF.
    Returns ingestion summary.
    """
    # Save file
    doc_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}_{filename}"
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"Saved PDF: {save_path}")

    # Load + chunk
    docs = _load_pdf(str(save_path))
    chunks = _split_documents(docs)

    # Enrich metadata
    for chunk in chunks:
        chunk.metadata["source_filename"] = filename
        chunk.metadata["doc_id"] = doc_id

    # Upsert to Pinecone
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    logger.info(f"Ingested {len(chunks)} chunks from {filename}")
    return {
        "doc_id": doc_id,
        "filename": filename,
        "pages": len(docs),
        "chunks": len(chunks),
    }


def list_ingested_documents() -> List[dict]:
    """List uploaded PDFs from the local upload directory."""
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.suffix == ".pdf":
            parts = f.stem.split("_", 1)
            files.append({
                "doc_id": parts[0] if len(parts) > 1 else f.stem,
                "filename": parts[1] if len(parts) > 1 else f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return files
