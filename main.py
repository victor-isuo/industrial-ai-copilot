from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
import time
import os
from src.core.document_loader import load_documents, chunk_documents
from src.core.vector_store import load_vector_store, create_vector_store
from src.core.retriever import create_hybrid_retriever
from src.core.reranker import CohereReranker
from src.core.rag_pipeline import RAGPipeline
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize pipeline on startup.
    Why lifespan context manager:
    - Loads heavy components once at startup, not per request
    - Embedding model and vector store stay in memory
    - Requests are fast because everything is pre-loaded
    """
    global pipeline
    logger.info("Starting Industrial AI Copilot API...")

    try:
        docs = load_documents()
        chunks = chunk_documents(docs)

        try:
            vector_store = load_vector_store()
        except FileNotFoundError:
            logger.info("Vector store not found, creating...")
            vector_store = create_vector_store(chunks)

        retriever = create_hybrid_retriever(vector_store, chunks)
        reranker = CohereReranker(top_n=5)
        pipeline = RAGPipeline(retriever=retriever, reranker=reranker)

        logger.info("Pipeline ready. API is live.")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


# Initialize FastAPI app
app = FastAPI(
    title="Industrial AI Copilot",
    description="A modular RAG system for industrial engineering documentation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
async def frontend():
    return FileResponse("static/index.html")

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What should I do if the gear pump loses suction?"
            }
        }


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str
    caveat: str
    processing_time_seconds: float


# --- Endpoints ---
@app.get("/")
async def root():
    return {
        "name": "Industrial AI Copilot",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the industrial knowledge base.
    Returns a grounded answer with source citations and confidence score.
    """
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Try again in a moment."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        start = time.time()
        response = pipeline.query(request.question)
        elapsed = round(time.time() - start, 2)

        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            caveat=response.caveat,
            processing_time_seconds=elapsed
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    from pathlib import Path
    docs = list(Path("data/raw").glob("*.pdf"))
    return {
        "indexed_documents": [d.name for d in docs],
        "total": len(docs)
    }
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)    