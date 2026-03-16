from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
import time
import os
import asyncio

from src.core.document_loader import load_documents, chunk_documents
from src.core.vector_store import load_vector_store, create_vector_store
from src.core.retriever import create_hybrid_retriever
from src.core.reranker import CohereReranker
from src.core.rag_pipeline import RAGPipeline
from src.agents.maintenance_agent import MaintenanceAgent
from src.api.ingest_router import router as ingest_router, set_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
pipeline     = None
agent        = None
vector_store = None  # Kept globally so ingest router can update it live


async def initialize_pipeline():
    global pipeline, agent, vector_store

    try:
        logger.info("Initializing pipeline in background...")
        docs   = load_documents()
        chunks = chunk_documents(docs)

        if not chunks:
            logger.error("No documents found. Check PDF files are present.")
            return

        logger.info("Building vector store from documents...")
        vector_store = create_vector_store(chunks)

        # Inject vector store into ingest router so uploads update the live index
        set_vector_store(vector_store)

        retriever = create_hybrid_retriever(vector_store, chunks)
        reranker  = CohereReranker(top_n=5)
        pipeline  = RAGPipeline(retriever=retriever, reranker=reranker)
        logger.info("Pipeline ready. System operational.")

        # Initialize agent
        agent = MaintenanceAgent(pipeline=pipeline)
        logger.info("Maintenance Agent ready.")

    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Server starts immediately and binds to port.
    Pipeline loads in background after startup.
    """
    logger.info("Starting Industrial AI Copilot API...")
    asyncio.create_task(initialize_pipeline())
    yield
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Industrial AI Copilot",
    description="A modular RAG system for industrial engineering documentation",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register ingestion router — all endpoints under /ingest
app.include_router(ingest_router)

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/ui")
async def frontend():
    return FileResponse("static/index.html")


@app.get("/agent-ui")
async def agent_frontend():
    return FileResponse("static/agent.html")


@app.get("/ingest-ui")
async def ingest_frontend():
    return FileResponse("static/ingest.html")


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


class AgentRequest(BaseModel):
    question: str


class AgentResponse(BaseModel):
    answer: str
    tools_used: list[str]
    steps_taken: int
    processing_time_seconds: float


# --- Endpoints ---
@app.get("/")
async def root():
    return {
        "name":    "Industrial AI Copilot",
        "version": "3.0.0",
        "status":  "operational",
        "docs":    "/docs",
        "phase":   "Phase 3 — Ingestion Pipeline + Telemetry + Multimodal"
    }


@app.get("/health")
async def health():
    return {
        "status":          "healthy",
        "pipeline_loaded": pipeline is not None,
        "agent_loaded":    agent is not None,
        "vector_store":    vector_store is not None,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the industrial knowledge base.
    Returns grounded answer with source citations and confidence score.
    """
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline is still initializing. Please wait 30 seconds and try again."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        start    = time.time()
        response = pipeline.query(request.question)
        elapsed  = round(time.time() - start, 2)

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
    if not docs:
        docs = list(Path(".").glob("*.pdf"))
    return {
        "indexed_documents": [d.name for d in docs],
        "total":             len(docs)
    }


@app.post("/agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    Run the agentic maintenance assistant.
    Uses LangGraph agent with tool use for complex multi-step reasoning.
    """
    if not agent:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized yet. Please wait and try again."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        start   = time.time()
        result  = agent.run(request.question)
        elapsed = round(time.time() - start, 2)

        return AgentResponse(
            answer=result["answer"],
            tools_used=result["tools_used"],
            steps_taken=result["steps_taken"],
            processing_time_seconds=elapsed
        )

    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
