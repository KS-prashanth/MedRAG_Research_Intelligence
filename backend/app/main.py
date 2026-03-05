from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Healthcare RAG API",
    description="RAG system for querying medical research papers using LangChain + Pinecone + Claude",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Healthcare RAG API is running. Visit /docs for the API explorer."}
