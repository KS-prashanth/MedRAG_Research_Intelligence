import logging
import re
from typing import List, Dict, Any

from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.core.config import get_settings
from app.core.vector_store import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()

HEALTHCARE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable medical research assistant helping clinicians and researchers understand scientific literature.

Use ONLY the provided context from research papers to answer the question. Be precise, evidence-based, and cite which paper/source supports each claim. 
If the context does not contain enough information to answer, say so clearly — do not hallucinate.

Context from research papers:
{context}

Question: {question}

Answer (with source references):""",
)

GENERAL_PROMPT = """You are a helpful medical research assistant. 
Answer the following conversational message naturally and concisely.

Message: {question}

Response:"""

# Patterns that indicate a general/conversational message (not a research query)
GENERAL_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings)[\s!.,?]*$",
    r"^(thanks|thank you|thx|ty)[\s!.,?]*$",
    r"^(bye|goodbye|see you|cya)[\s!.,?]*$",
    r"^(ok|okay|got it|sure|alright|sounds good)[\s!.,?]*$",
    r"^(good|great|nice|awesome|cool|perfect|wonderful)[\s!.,?]*$",
    r"^(yes|no|yeah|nope|yep)[\s!.,?]*$",
    r"^how are you",
    r"^what can you do",
    r"^who are you",
    r"^what are you",
    r"^help me?$",
]


def is_general_conversation(question: str) -> bool:
    """Detect if the message is conversational and doesn't need document retrieval."""
    q = question.strip().lower()
    for pattern in GENERAL_PATTERNS:
        if re.match(pattern, q):
            return True
    # Also treat very short messages (under 4 words) with no research keywords as general
    words = q.split()
    research_keywords = {
        "study", "paper", "research", "find", "what", "how", "why", "when",
        "which", "explain", "describe", "summarize", "compare", "result",
        "method", "conclusion", "finding", "patient", "treatment", "cancer",
        "clinical", "drug", "therapy", "disease", "analysis", "model", "data"
    }
    if len(words) <= 3 and not any(w in research_keywords for w in words):
        return True
    return False


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=settings.anthropic_api_key,
        temperature=0.1,
        max_tokens=1500,
    )


def query_rag(question: str) -> Dict[str, Any]:
    """
    Run a RAG query or handle general conversation.
    Returns answer + source documents (empty for general conversation).
    """
    # Handle general conversation without hitting the vector store
    if is_general_conversation(question):
        llm = get_llm()
        response = llm.invoke(GENERAL_PROMPT.format(question=question))
        return {
            "answer": response.content,
            "sources": [],
            "question": question,
        }

    # Research question — full RAG pipeline
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k_results},
    )

    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": HEALTHCARE_PROMPT},
    )

    result = qa_chain.invoke({"query": question})

    # Format source documents
    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        fname = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{fname}:p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": fname,
                "page": page,
                "snippet": doc.page_content[:300] + "...",
            })

    return {
        "answer": result["result"],
        "sources": sources,
        "question": question,
    }