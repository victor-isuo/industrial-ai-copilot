"""
Industrial AI Copilot — Evaluation Runner
==========================================
Runs all 30 cases from eval_dataset.py by calling the agent directly.
No server required — initializes pipeline and agent in-process.

Usage:
    # Full run from scratch:
    python -m src.evaluation.eval_runner

    # Resume from a specific index (e.g. after interruption at case 12):
    python -m src.evaluation.eval_runner --start 12

    # Run one category only:
    python -m src.evaluation.eval_runner --category retrieval
"""

import json
import time
import logging
import argparse
from datetime import datetime
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_FILE = "eval_results.json"
BATCH_SIZE = 3 # cases per batch before pausing
BATCH_DELAY = 60 # seconds between batches (Groq rate limit buffer)
CASE_DELAY = 10 # seconds between individual cases

# ── Import dataset ────────────────────────────────────────────────────────────
from src.evaluation.eval_dataset import EVAL_DATASET


# ── Pipeline + Agent Initializer ─────────────────────────────────────────────

def initialize_agent():
    """
    Initialize the full pipeline and agent exactly as test_agent() does.
    Uses load_vector_store() — does NOT rebuild the index from scratch.
    """
    logger.info("Initializing pipeline — loading vector store from disk...")

    from src.core.document_loader import load_documents, chunk_documents
    from src.core.vector_store import load_vector_store
    from src.core.retriever import create_hybrid_retriever
    from src.core.reranker import CohereReranker
    from src.core.rag_pipeline import RAGPipeline
    from src.agents.maintenance_agent import MaintenanceAgent

    docs = load_documents()
    chunks = chunk_documents(docs)
    vector_store = load_vector_store()
    retriever = create_hybrid_retriever(vector_store, chunks)
    reranker = CohereReranker(top_n=5)
    pipeline = RAGPipeline(retriever=retriever, reranker=reranker)
    agent = MaintenanceAgent(pipeline=pipeline)

    logger.info("Agent ready.\n")
    return agent


# ── File I/O ──────────────────────────────────────────────────────────────────

def load_existing_results() -> list:
    """Load existing results so previous runs are never lost."""
    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} existing results from {RESULTS_FILE}")
        return data
    except FileNotFoundError:
        logger.info("No existing results file — starting fresh.")
        return []
    except json.JSONDecodeError:
        logger.warning("Results file corrupted — starting fresh.")
        return []


def save_results(results: list):
    """Overwrite results file with full merged list."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def get_completed_ids(results: list) -> set:
    """Return IDs already completed so they are never re-run."""
    return {r["case_id"] for r in results}


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_tool_usage(agent_result: dict, case: dict) -> float:
    """
    40% weight.
    1.0 — correct tool used
    0.5 — wrong tool used
    0.0 — no tool used when one was expected
    """
    expected = case.get("expected_tool")
    tools_used = agent_result.get("tools_used", [])

    if expected is None:
        # Edge case — no tool should fire
        return 1.0 if not tools_used else 0.5

    if not tools_used:
        return 0.0

    tools_normalized = [t.lower().strip() for t in tools_used]
    return 1.0 if expected.lower().strip() in tools_normalized else 0.5


def score_keywords(agent_result: dict, case: dict) -> float:
    """
    40% weight.
    Fraction of expected keywords found in answer (case-insensitive).
    """
    keywords = case.get("expected_keywords", [])
    if not keywords:
        return 1.0

    answer = agent_result.get("answer", "").lower()
    matched = sum(1 for kw in keywords if kw.lower() in answer)
    return matched / len(keywords)


def score_severity(agent_result: dict, case: dict) -> float:
    """
    20% weight.
    Only applies to spec_check cases with expected_severity.
    Full credit if expected severity string appears in answer.
    """
    expected_severity = case.get("expected_severity", "")
    if not expected_severity:
        return 1.0

    answer = agent_result.get("answer", "").upper()
    return 1.0 if expected_severity.upper() in answer else 0.0


def compute_scores(agent_result: dict, case: dict) -> dict:
    """Compute weighted composite score for one case."""
    tool_score = score_tool_usage(agent_result, case)
    keyword_score = score_keywords(agent_result, case)
    severity_score = score_severity(agent_result, case)

    composite = (
        tool_score * 0.40 +
        keyword_score * 0.40 +
        severity_score * 0.20
    )

    return {
        "tool_score": round(tool_score, 3),
        "keyword_score": round(keyword_score, 3),
        "severity_score": round(severity_score, 3),
        "composite_score": round(composite, 3),
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list):
    """Print formatted evaluation report."""
    if not results:
        logger.info("No results to summarize.")
        return

    total = len(results)
    passed = sum(1 for r in results if r["scores"]["composite_score"] >= 0.7)
    avg_score = sum(r["scores"]["composite_score"] for r in results) / total
    avg_latency = sum(r["latency_seconds"] for r in results) / total

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "scores": []}
        categories[cat]["total"] += 1
        categories[cat]["scores"].append(r["scores"]["composite_score"])
        if r["scores"]["composite_score"] >= 0.7:
            categories[cat]["passed"] += 1

    print("\n" + "=" * 62)
    print(" INDUSTRIAL AI COPILOT — EVALUATION REPORT")
    print("=" * 62)
    print(f" Total Cases : {total} / {len(EVAL_DATASET)}")
    print(f" Passed (≥0.70) : {passed} / {total} ({100*passed//total}%)")
    print(f" Avg Score : {avg_score:.3f}")
    print(f" Avg Latency : {avg_latency:.2f}s")
    print("-" * 62)
    print(" CATEGORY BREAKDOWN")
    print("-" * 62)
    for cat, data in categories.items():
        cat_avg = sum(data["scores"]) / len(data["scores"])
        print(
            f" {cat:<28} "
            f"{data['passed']}/{data['total']} passed "
            f"avg: {cat_avg:.3f}"
        )
    print("-" * 62)
    print(" CASE DETAILS")
    print("-" * 62)
    for r in results:
        status = "✓ PASS" if r["scores"]["composite_score"] >= 0.7 else "✗ FAIL"
        err = f" ERROR: {r['error']}" if r.get("error") else ""
        print(
            f" {r['case_id']:<12} {status} "
            f"score={r['scores']['composite_score']:.2f} "
            f"latency={r['latency_seconds']:.1f}s "
            f"tools={r.get('tools_used', [])}"
            f"{err}"
        )
    print("=" * 62 + "\n")


# ── Main Runner ───────────────────────────────────────────────────────────────

def run_evaluation(
    start_index: int = 0,
    category_filter: Optional[str] = None,
):
    logger.info("=" * 50)
    logger.info("INDUSTRIAL AI COPILOT — EVALUATION RUNNER")
    logger.info("=" * 50)

    # Load existing results — never lose passing cases
    all_results = load_existing_results()
    completed_ids = get_completed_ids(all_results)

    if completed_ids:
        logger.info(f"Already completed: {sorted(completed_ids)}")

    # Build run queue
    queue = []
    for i, case in enumerate(EVAL_DATASET):
        if i < start_index:
            continue
        if category_filter and case["category"] != category_filter:
            continue
        if case["id"] in completed_ids:
            logger.info(f"Skipping {case['id']} — already done")
            continue
        queue.append(case)

    if not queue:
        logger.info("Nothing new to run — all cases already completed.")
        print_summary(all_results)
        return

    logger.info(f"\nRunning {len(queue)} cases | batch_size={BATCH_SIZE} | batch_delay={BATCH_DELAY}s")
    estimated = (len(queue) * CASE_DELAY + (len(queue) // BATCH_SIZE) * BATCH_DELAY) // 60
    logger.info(f"Estimated time: ~{estimated} minutes\n")

    # Initialize agent ONCE — reuse across all cases
    agent = initialize_agent()

    for i, case in enumerate(queue):

        # Batch pause (never before first case)
        if i > 0 and i % BATCH_SIZE == 0:
            logger.info(f"--- Batch complete. Pausing {BATCH_DELAY}s ---\n")
            time.sleep(BATCH_DELAY)

        logger.info(f"[{i+1}/{len(queue)}] {case['id']} ({case['category']})")
        logger.info(f" Query: {case['query']}")

        start_time = time.time()

        try:
            agent.conversation_history = []  # Clear history for each case to ensure independence
            agent_result = agent.run(case["query"])
            latency = round(time.time() - start_time, 2)
            scores = compute_scores(agent_result, case)

            record = {
                "case_id": case["id"],
                "category": case["category"],
                "query": case["query"],
                "answer": agent_result.get("answer", ""),
                "tools_used": agent_result.get("tools_used", []),
                "steps_taken": agent_result.get("steps_taken", 0),
                "latency_seconds": latency,
                "scores": scores,
                "timestamp": datetime.now().isoformat(),
                "status": "pass" if scores["composite_score"] >= 0.7 else "fail",
            }

            icon = "✓" if record["status"] == "pass" else "✗"
            logger.info(
                f" {icon} composite={scores['composite_score']:.3f} "
                f"tool={scores['tool_score']:.2f} "
                f"kw={scores['keyword_score']:.2f} "
                f"sev={scores['severity_score']:.2f} "
                f"latency={latency}s"
            )

        except Exception as e:
            latency = round(time.time() - start_time, 2)
            logger.error(f" ✗ FAILED: {e}")

            record = {
                "case_id": case["id"],
                "category": case["category"],
                "query": case["query"],
                "answer": "",
                "tools_used": [],
                "steps_taken": 0,
                "latency_seconds": latency,
                "scores": {
                    "tool_score": 0.0,
                    "keyword_score": 0.0,
                    "severity_score": 0.0,
                    "composite_score": 0.0,
                },
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
            }

        # Save after every case — safe to Ctrl+C anytime
        all_results.append(record)
        save_results(all_results)

        # Delay between cases
        if i < len(queue) - 1:
            logger.info(f" Waiting {CASE_DELAY}s...\n")
            time.sleep(CASE_DELAY)

    logger.info("All cases complete.")
    print_summary(all_results)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Industrial AI Copilot evaluation suite"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start from this dataset index (0-based). e.g. --start 12"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["spec_check", "unit_conversion", "retrieval", "edge_case"],
        help="Only run cases in this category"
    )
    args = parser.parse_args()

    run_evaluation(
        start_index=args.start,
        category_filter=args.category,
    )