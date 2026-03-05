# 🏥 Healthcare RAG — Research Intelligence

A full-stack RAG (Retrieval-Augmented Generation) system for querying medical research papers using **FastAPI + LangChain + Pinecone + Claude**.

---

## Architecture

```
User → Frontend (HTML/JS)
         ↓
     FastAPI Backend
         ↓
   LangChain Pipeline
    ↙           ↘
Pinecone       Claude (claude-sonnet-4)
(Vector DB)    (LLM / Answer Gen)
```

### Stack
| Layer | Technology |
|---|---|
| Backend | FastAPI + Python |
| RAG Framework | LangChain |
| Vector DB | Pinecone (cloud) |
| Embeddings | `pritamdeka/S-PubMedBert-MS-MARCO` (biomedical) |
| LLM | Claude claude-sonnet-4 (Anthropic) |
| PDF Parsing | PyPDF (LangChain) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Setup

### 1. Clone & Install

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Create a .env file 
Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=healthcare-rag
PINECONE_ENVIRONMENT=us-east-1
```

### 3. Run the Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 4. Open the Frontend

Open `frontend/index.html` in your browser (or serve it):

```bash
cd frontend
python -m http.server 3000
```

Visit `http://localhost:3000`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/ingest` | Upload a PDF research paper |
| `GET` | `/api/v1/documents` | List ingested papers |
| `POST` | `/api/v1/query` | Ask a question (RAG) |

### Example Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What were the primary outcomes of the clinical trial?"}'
```

---

## Project Structure

```
healthcare-rag/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── api/routes.py        # API endpoints
│   │   ├── core/
│   │   │   ├── config.py        # Settings / env vars
│   │   │   └── vector_store.py  # Pinecone + embeddings
│   │   ├── services/
│   │   │   ├── ingestion.py     # PDF loading, chunking, upsert
│   │   │   └── rag.py           # RetrievalQA chain
│   │   └── models/schemas.py    # Pydantic models
│   ├── data/uploads/            # Uploaded PDFs (local)
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html               # Single-page UI
```

---

## RAG Pipeline Details

1. **Ingestion**: PDF → PyPDF pages → `RecursiveCharacterTextSplitter` (1000 tokens, 200 overlap) → biomedical embeddings → Pinecone upsert
2. **Query**: Question → embed → Pinecone similarity search (top-5) → stuffed into Claude prompt → grounded answer + sources

The system prompt instructs Claude to **only answer from retrieved context** and cite sources, preventing hallucination.
