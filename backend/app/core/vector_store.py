from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)

settings = get_settings()

# Use a biomedical-optimized embedding model
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_DIMENSION = 768


def get_embeddings():
    """Return HuggingFace biomedical embeddings."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_pinecone_index():
    """Initialize Pinecone and return/create the index."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name

    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        logger.info(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
        )
    return pc.Index(index_name)


def get_vector_store() -> PineconeVectorStore:
    """Return a LangChain-compatible PineconeVectorStore."""
    embeddings = get_embeddings()
    index = get_pinecone_index()
    return PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
