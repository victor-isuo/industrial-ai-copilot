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
from src.agents.multi_agent_system import initialize_multi_agent_system, get_multi_agent_system
from src.api.ingest_router import router as ingest_router, set_vector_store
from src.api.telemetry_api import get_equipment_telemetry as fetch_telemetry, list_equipment

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

        # Initialize single agent
        agent = MaintenanceAgent(pipeline=pipeline)
        logger.info("Maintenance Agent ready.")

        # Initialize multi-agent system
        initialize_multi_agent_system(pipeline)
        logger.info("Multi-Agent System ready.")

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


@app.get("/multiagent-ui")
async def multiagent_frontend():
    return FileResponse("static/multiagent.html")


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
    image_base64: str = None
    analysis_type: str = "general"


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
        start = time.time()

        # If image provided, prepend image path info to question
        question = request.question
        if request.image_base64:
            # Save image temporarily for the vision tool
            import base64
            import uuid
            from pathlib import Path

            img_dir = Path("data/images")
            img_dir.mkdir(parents=True, exist_ok=True)

            # Decode and save image
            img_id   = str(uuid.uuid4())[:8]
            img_data = request.image_base64

            # Handle data URI format
            if "," in img_data:
                header, b64 = img_data.split(",", 1)
                ext = "jpg" if "jpeg" in header else "png"
            else:
                b64  = img_data
                ext  = "jpg"

            img_path = img_dir / f"upload_{img_id}.{ext}"
            img_path.write_bytes(base64.b64decode(b64))

            # Prepend image context to question
            question = (
                f"{request.question}\n\n"
                f"[IMAGE ATTACHED: {img_path} | "
                f"Analysis type: {request.analysis_type}]\n"
                f"Use analyze_equipment_image or analyze_gauge_reading tool "
                f"with image_path='{img_path}' and "
                f"analysis_type='{request.analysis_type}'"
            )

        result  = agent.run(question)
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


@app.get("/telemetry")
async def telemetry_overview():
    """Get health overview of all equipment in the plant."""
    equipment = list_equipment()
    return {
        "equipment":    equipment,
        "total_assets": len(equipment),
        "alerts":       sum(e["alert_count"] for e in equipment),
    }


@app.get("/telemetry/{equipment_id}")
async def telemetry_readings(equipment_id: str):
    """Get live sensor readings for a specific equipment asset."""
    data = fetch_telemetry(equipment_id)
    if not data:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Equipment '{equipment_id}' not found."
        )
    return data


# ── Multi-Agent Endpoints ─────────────────────────────────────────────────────

class MultiAgentRequest(BaseModel):
    question: str


@app.post("/multiagent")
async def run_multi_agent(request: MultiAgentRequest):
    """
    Run the multi-agent orchestration system.
    Supervisor delegates to specialist agents and synthesizes results.
    """
    mas = get_multi_agent_system()

    if not mas:
        raise HTTPException(
            status_code=503,
            detail="Multi-agent system not initialized yet. Please wait and try again."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = mas.run(request.question)
        return {
            "final_answer":  result.final_answer,
            "agent_results": [
                {
                    "agent_name": r.agent_name,
                    "role":       r.role,
                    "response":   r.response,
                    "latency":    r.latency,
                    "status":     r.status,
                    "color":      r.color,
                }
                for r in result.agent_results
            ],
            "agents_used":   result.agents_used,
            "total_latency": result.total_latency,
            "query":         result.query,
        }

    except Exception as e:
        logger.error(f"Multi-agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/multiagent/agents")
async def list_agents():
    """List available specialist agents and their roles."""
    return {
        "agents": [
            {"name": "Retrieval Agent", "role": "Documentation search and citation",       "color": "#00d4ff"},
            {"name": "Telemetry Agent", "role": "Live equipment monitoring",                "color": "#00e676"},
            {"name": "Analysis Agent",  "role": "Engineering calculations and spec checks", "color": "#ffd740"},
            {"name": "Safety Agent",    "role": "Risk assessment and safety compliance",    "color": "#ff6b6b"},
            {"name": "Report Agent",    "role": "Findings synthesis and final response",    "color": "#c77dff"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)