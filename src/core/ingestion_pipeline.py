"""
Ingestion Pipeline — Industrial AI Copilot
==========================================
Handles live document ingestion — upload a PDF, chunk it,
embed it, and update the ChromaDB index without restarting the server.

Why this matters:
- Real production systems need live data updates
- Documents change — manuals get revised, standards get updated
- No downtime ingestion is a core data engineering requirement
"""

import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Ingestion state — tracks status of ongoing ingestions
ingestion_registry: dict = {}


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Compute SHA256 hash of file bytes.
    Used for duplicate detection — same file won't be indexed twice.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def is_duplicate(file_hash: str, raw_dir: Path) -> Optional[str]:
    """
    Check if a document with this hash already exists in the knowledge base.
    Returns the existing filename if duplicate, None if new.
    """
    hash_store = raw_dir / ".hash_index"
    if not hash_store.exists():
        return None

    with open(hash_store, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 2 and parts[1] == file_hash:
                return parts[0]

    return None


def register_hash(filename: str, file_hash: str, raw_dir: Path):
    """Register a new document hash in the hash index."""
    hash_store = raw_dir / ".hash_index"
    with open(hash_store, "a") as f:
        f.write(f"{filename}|{file_hash}\n")


def get_ingestion_status(job_id: str) -> dict:
    """Get the current status of an ingestion job."""
    return ingestion_registry.get(job_id, {
        "job_id": job_id,
        "status": "not_found",
        "message": "Job ID not found"
    })


def list_ingestion_jobs() -> list:
    """Return all ingestion jobs with their statuses."""
    return list(ingestion_registry.values())


async def ingest_document(
    filename: str,
    file_bytes: bytes,
    vector_store,
    job_id: str,
) -> dict:
    """
    Full ingestion pipeline for a single PDF document.

    Steps:
    1. Duplicate check via SHA256 hash
    2. Save PDF to data/raw/
    3. Load and chunk document
    4. Add chunks to existing ChromaDB vector store
    5. Register hash to prevent future duplicates
    6. Update job status throughout

    Args:
        filename:    Original filename of the uploaded PDF
        file_bytes:  Raw bytes of the PDF file
        vector_store: Existing ChromaDB vector store to update
        job_id:      Unique identifier for tracking this ingestion job

    Returns:
        dict with job status, chunks added, and any errors
    """
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Initialize job status
    ingestion_registry[job_id] = {
        "job_id":    job_id,
        "filename":  filename,
        "status":    "started",
        "message":   "Ingestion started",
        "chunks_added": 0,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
    }

    try:
        # Step 1 — Duplicate detection
        ingestion_registry[job_id]["status"]  = "checking"
        ingestion_registry[job_id]["message"] = "Checking for duplicates..."
        logger.info(f"[{job_id}] Checking for duplicate: {filename}")

        file_hash  = compute_file_hash(file_bytes)
        existing   = is_duplicate(file_hash, raw_dir)

        if existing:
            ingestion_registry[job_id].update({
                "status":  "skipped",
                "message": f"Duplicate detected — already indexed as '{existing}'",
                "completed_at": datetime.now().isoformat(),
            })
            logger.info(f"[{job_id}] Duplicate detected: {existing}")
            return ingestion_registry[job_id]

        # Step 2 — Save PDF to disk
        ingestion_registry[job_id]["status"]  = "saving"
        ingestion_registry[job_id]["message"] = "Saving document to knowledge base..."

        save_path = raw_dir / filename
        # If filename already exists, append timestamp to avoid overwrite
        if save_path.exists():
            stem      = Path(filename).stem
            suffix    = Path(filename).suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"{stem}_{timestamp}{suffix}"
            save_path = raw_dir / filename

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        logger.info(f"[{job_id}] Saved to {save_path}")

        # Step 3 — Load and chunk
        ingestion_registry[job_id]["status"]  = "chunking"
        ingestion_registry[job_id]["message"] = "Chunking document..."

        from src.core.document_loader import chunk_documents
        from langchain_community.document_loaders import PyPDFLoader

        loader   = PyPDFLoader(str(save_path))
        pages    = loader.load()
        chunks   = chunk_documents(pages)

        logger.info(f"[{job_id}] Created {len(chunks)} chunks from {len(pages)} pages")

        if not chunks:
            raise ValueError("Document produced zero chunks — may be empty or unreadable")

        # Step 4 — Add to vector store
        ingestion_registry[job_id]["status"]  = "embedding"
        ingestion_registry[job_id]["message"] = f"Embedding {len(chunks)} chunks..."

        # Add chunks to existing ChromaDB collection
        texts    = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids      = [f"{filename}_{i}_{job_id}" for i in range(len(chunks))]

        vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"[{job_id}] Added {len(chunks)} chunks to vector store")

        # Step 5 — Register hash
        register_hash(filename, file_hash, raw_dir)

        # Step 6 — Complete
        ingestion_registry[job_id].update({
            "status":       "complete",
            "message":      f"Successfully indexed {len(chunks)} chunks from {len(pages)} pages",
            "chunks_added": len(chunks),
            "pages":        len(pages),
            "filename":     filename,
            "completed_at": datetime.now().isoformat(),
        })

        logger.info(f"[{job_id}] Ingestion complete — {len(chunks)} chunks added")
        return ingestion_registry[job_id]

    except Exception as e:
        logger.error(f"[{job_id}] Ingestion failed: {e}")
        ingestion_registry[job_id].update({
            "status":       "failed",
            "message":      f"Ingestion failed: {str(e)}",
            "completed_at": datetime.now().isoformat(),
        })
        # Clean up saved file if ingestion failed
        if 'save_path' in locals() and Path(save_path).exists():
            try:
                Path(save_path).unlink()
                logger.info(f"[{job_id}] Cleaned up failed file: {save_path}")
            except Exception:
                pass
        return ingestion_registry[job_id]
