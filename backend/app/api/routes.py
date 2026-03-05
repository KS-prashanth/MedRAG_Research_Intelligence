from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import (
    QueryRequest, QueryResponse, IngestResponse,
    DocumentInfo, HealthResponse, SourceDocument
)
from app.services.ingestion import ingest_pdf, list_ingested_documents
from app.services.rag import query_rag
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a research paper PDF into Pinecone."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        file_bytes = await file.read()
        result = ingest_pdf(file_bytes, file.filename)
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=list[DocumentInfo])
async def get_documents():
    """List all ingested research papers."""
    try:
        return list_ingested_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question about the ingested research papers."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = query_rag(request.question)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result["sources"]],
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
