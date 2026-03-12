"""
Industrial AI Copilot — Evaluation Runner
Tests agent performance across 30 cases in batches to avoid rate limits.
"""

import time
import json
import logging
from datetime import datetime
from src.evaluation.eval_dataset import EVAL_DATASET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def score_response(result: dict, test_case: dict) -> dict:
    """
    Score a single agent response against expected outputs.
    Returns scoring breakdown.
    """
    answer = result.get("answer", "").lower()
    tools_used = result.get("tools_used", [])

    # Score 1: Tool selection accuracy
    expected_tool = test_case.get("expected_tool")
    if expected_tool is None:
        tool_score = 1.0  # No tool expected
    elif expected_tool in tools_used:
        tool_score = 1.0
    else:
        tool_score = 0.0

    # Score 2: Keyword presence
    expected_keywords = test_case.get("expected_keywords", [])
    if expected_keywords:
        matched = sum(1 for kw in expected_keywords
                     if kw.lower() in answer)
        keyword_score = matched / len(expected_keywords)
    else:
        keyword_score = 1.0

    # Score 3: Severity detection (for spec check cases)
    expected_severity = test_case.get("expected_severity", "").lower()
    if expected_severity:
        severity_score = 1.0 if expected_severity in answer else 0.0
    else:
        severity_score = 1.0

    # Overall score
    overall = (tool_score * 0.4) + (keyword_score * 0.4) + (severity_score * 0.2)

    return {
        "tool_score": tool_score,
        "keyword_score": round(keyword_score, 2),
        "severity_score": severity_score,
        "overall_score": round(overall, 2),
        "tools_used": tools_used,
        "expected_tool": expected_tool,
    }


def run_evaluation(agent, batch_size: int = 5, delay_seconds: int = 30):
    """
    Run evaluation across all test cases in batches.
    Saves results to JSON after each batch.
    """
    results = []
    total = len(EVAL_DATASET)

    logger.info(f"Starting evaluation: {total} cases, batch size {batch_size}")
    logger.info(f"Estimated time: {(total // batch_size) * delay_seconds / 60:.1f} minutes")

    for i in range(0, total, batch_size):
        batch = EVAL_DATASET[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(f"\nBatch {batch_num}/{total_batches}")

        for case in batch:
            logger.info(f"  Running: {case['id']} — {case['query'][:50]}...")

            start = time.time()
            try:
                result = agent.run(case["query"])
                latency = round(time.time() - start, 2)
                scores = score_response(result, case)

                results.append({
                    "id": case["id"],
                    "category": case["category"],
                    "query": case["query"],
                    "answer": result.get("answer", "")[:200],
                    "latency": latency,
                    "steps": result.get("steps_taken", 0),
                    **scores,
                    "status": "pass" if scores["overall_score"] >= 0.6 else "fail"
                })

                status = "✓" if scores["overall_score"] >= 0.6 else "✗"
                logger.info(f"  {status} Score: {scores['overall_score']} | Latency: {latency}s")

            except Exception as e:
                logger.error(f"  ERROR: {e}")
                results.append({
                    "id": case["id"],
                    "category": case["category"],
                    "query": case["query"],
                    "status": "error",
                    "error": str(e),
                    "overall_score": 0.0,
                    "latency": 0,
                    "steps": 0
                })

            time.sleep(3)  # Small delay between cases

        # Save after each batch
        save_results(results)

        # Delay between batches to avoid rate limits
        if i + batch_size < total:
            logger.info(f"  Batch complete. Waiting {delay_seconds}s before next batch...")
            time.sleep(delay_seconds)

    return generate_report(results)


def generate_report(results: list) -> dict:
    """Generate summary metrics from evaluation results."""
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "pass")
    errors = sum(1 for r in results if r.get("status") == "error")

    valid = [r for r in results if r.get("status") != "error"]
    avg_score = sum(r["overall_score"] for r in valid) / len(valid) if valid else 0
    avg_latency = sum(r["latency"] for r in valid) / len(valid) if valid else 0
    avg_steps = sum(r["steps"] for r in valid) / len(valid) if valid else 0
    tool_accuracy = sum(r.get("tool_score", 0) for r in valid) / len(valid) if valid else 0

    # Per category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "scores": []}
        categories[cat]["total"] += 1
        if r.get("status") == "pass":
            categories[cat]["passed"] += 1
        categories[cat]["scores"].append(r.get("overall_score", 0))

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_cases": total,
            "passed": passed,
            "failed": total - passed - errors,
            "errors": errors,
            "accuracy": round(passed / total * 100, 1),
            "avg_score": round(avg_score, 2),
            "avg_latency_seconds": round(avg_latency, 2),
            "avg_reasoning_steps": round(avg_steps, 1),
            "tool_selection_accuracy": round(tool_accuracy * 100, 1),
        },
        "by_category": {
            cat: {
                "accuracy": round(data["passed"] / data["total"] * 100, 1),
                "avg_score": round(sum(data["scores"]) / len(data["scores"]), 2)
            }
            for cat, data in categories.items()
        }
    }

    print_report(report)
    return report


def print_report(report: dict):
    """Print formatted evaluation report."""
    s = report["summary"]
    print("\n" + "="*60)
    print("INDUSTRIAL AI COPILOT — EVALUATION REPORT")
    print("="*60)
    print(f"Total Cases:              {s['total_cases']}")
    print(f"Passed:                   {s['passed']}")
    print(f"Accuracy:                 {s['accuracy']}%")
    print(f"Avg Score:                {s['avg_score']}")
    print(f"Avg Latency:              {s['avg_latency_seconds']}s")
    print(f"Avg Reasoning Steps:      {s['avg_reasoning_steps']}")
    print(f"Tool Selection Accuracy:  {s['tool_selection_accuracy']}%")
    print("\nBy Category:")
    for cat, data in report["by_category"].items():
        print(f"  {cat:25} {data['accuracy']}% accuracy")
    print("="*60)


def save_results(results: list):
    """Save results to JSON file."""
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to eval_results.json ({len(results)} cases)")


if __name__ == "__main__":
    from src.core.vector_store import load_vector_store
    from src.core.retriever import create_hybrid_retriever
    from src.core.document_loader import load_documents, chunk_documents
    from src.core.reranker import CohereReranker
    from src.core.rag_pipeline import RAGPipeline
    from src.agents.maintenance_agent import MaintenanceAgent

    vector_store = load_vector_store()
    docs = load_documents()
    chunks = chunk_documents(docs)
    retriever = create_hybrid_retriever(vector_store, chunks)
    reranker = CohereReranker(top_n=5)
    pipeline = RAGPipeline(retriever=retriever, reranker=reranker)
    agent = MaintenanceAgent(pipeline=pipeline)

    EVAL_DATASET = EVAL_DATASET[16:]  # Skip first 12 cases for initial testing
    run_evaluation(agent, batch_size=5, delay_seconds=45)