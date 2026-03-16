"""
Ingestion Router — Industrial AI Copilot
=========================================
FastAPI router exposing document ingestion endpoints.

Endpoints:
    POST /ingest          — Upload and index a new PDF document
    GET  /ingest/status/{job_id} — Check ingestion job status
    GET  /ingest/jobs     — List all ingestion jobs
    GET  /ingest/documents — List all indexed documents
"""

import uuid
import asyncio
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from src.core.ingestion_pipeline import (
    ingest_document,
    get_ingestion_status,
    list_ingestion_jobs,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Reference to the global vector store — set by main.py on startup
_vector_store = None


def set_vector_store(vs):
    """Called by main.py after pipeline initialization to inject vector store."""
    global _vector_store
    _vector_store = vs
    logger.info("Ingest router: vector store registered")


@router.post("")
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF document and add it to the knowledge base.

    The ingestion runs as a background task so the endpoint
    returns immediately with a job_id for status tracking.

    Args:
        file: PDF file to ingest (multipart/form-data)

    Returns:
        job_id for tracking ingestion progress
    """
    if _vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized yet. Please wait and try again."
        )

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Validate file size — max 50MB
    file_bytes = await file.read()
    max_size   = 50 * 1024 * 1024  # 50MB

    if len(file_bytes) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 50MB. Got {len(file_bytes) / 1024 / 1024:.1f}MB"
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="File is empty."
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8].upper()

    logger.info(f"Ingestion job {job_id} started for: {file.filename}")

    # Run ingestion in background — don't block the HTTP response
    background_tasks.add_task(
        ingest_document,
        filename=file.filename,
        file_bytes=file_bytes,
        vector_store=_vector_store,
        job_id=job_id,
    )

    return {
        "job_id":   job_id,
        "filename": file.filename,
        "status":   "started",
        "message":  "Ingestion started. Poll /ingest/status/{job_id} for progress.",
        "size_kb":  round(len(file_bytes) / 1024, 1),
    }


@router.get("/status/{job_id}")
async def ingestion_status(job_id: str):
    """
    Check the status of an ingestion job.

    Status values:
        started   — Job received, processing beginning
        checking  — Duplicate detection in progress
        saving    — Writing PDF to disk
        chunking  — Splitting document into chunks
        embedding — Adding chunks to vector store
        complete  — Successfully indexed
        skipped   — Duplicate document, already indexed
        failed    — Ingestion failed (see message for reason)
    """
    status = get_ingestion_status(job_id.upper())

    if status["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"Job ID '{job_id}' not found."
        )

    return status


@router.get("/jobs")
async def list_jobs():
    """List all ingestion jobs with their current status."""
    jobs = list_ingestion_jobs()
    return {
        "total_jobs": len(jobs),
        "jobs": sorted(jobs, key=lambda x: x.get("started_at", ""), reverse=True)
    }


@router.get("/documents")
async def list_documents():
    """
    List all documents currently in the knowledge base.
    Reads directly from data/raw/ directory.
    """
    raw_dir = Path("data/raw")

    if not raw_dir.exists():
        return {"documents": [], "total": 0}

    pdfs = sorted(raw_dir.glob("*.pdf"))

    documents = []
    for pdf in pdfs:
        stat = pdf.stat()
        documents.append({
            "filename":  pdf.name,
            "size_kb":   round(stat.st_size / 1024, 1),
            "modified":  stat.st_mtime,
        })

    return {
        "documents": documents,
        "total":     len(documents),
    }
