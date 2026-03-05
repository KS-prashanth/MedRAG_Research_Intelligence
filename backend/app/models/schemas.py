from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str


class SourceDocument(BaseModel):
    filename: str
    page: int | str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    message: str = "Document successfully ingested"


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    size_kb: float


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
