import sys
import time
import requests

ALB_URL = "http://industrial-copilot-alb-998145074.us-east-1.elb.amazonaws.com"
MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 15

def check_health():
    """Poll /health until it returns 200, or fail after max retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(f"{ALB_URL}/health", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("pipeline_loaded") and data.get("vector_store") and data.get("agent_loaded"):
                    print(f"[OK] Health check passed on attempt {attempt}: {data}")
                    return True
            print(f"[WAIT] Attempt {attempt}/{MAX_RETRIES}: status={resp.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[WAIT] Attempt {attempt}/{MAX_RETRIES}: {e}")
        time.sleep(RETRY_DELAY_SECONDS)

    print("[FAIL] Health check did not pass within retry window.")
    return False

def check_real_query():
    """
    Send a real query through the single-agent endpoint (/agent) and verify
    a well-formed, sane response — matching AgentRequest / AgentResponse schema.
    """
    payload = {
        "question": "Pump discharge pressure is 600 psi. Safety relief valve set at 500 psi. Is this safe?"
    }
    try:
        resp = requests.post(f"{ALB_URL}/agent", json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Request to /agent failed: {e}")
        return False

    if resp.status_code != 200:
        print(f"[FAIL] /agent returned status {resp.status_code}: {resp.text[:300]}")
        return False

    data = resp.json()

    # Schema check — confirms AgentResponse shape, not just any 200
    required_fields = ["answer", "tools_used", "steps_taken", "processing_time_seconds"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        print(f"[FAIL] Response missing expected fields {missing}: {data}")
        return False

    if not data["answer"].strip():
        print(f"[FAIL] Agent returned an empty answer: {data}")
        return False

    if not data["tools_used"]:
        print(f"[FAIL] Agent used no tools — likely fell back to raw LLM without reasoning: {data}")
        return False

    # Content sanity check — this scenario is a known out-of-spec/critical case per the eval suite
    answer_lower = data["answer"].lower()
    if "critical" not in answer_lower and "shutdown" not in answer_lower and "immediate" not in answer_lower:
        print(f"[WARN] Response did not contain expected severity language, but structure is valid: {data['answer'][:200]}")
        # Not a hard fail — wording can vary — but flagged for visibility in logs

    print(f"[OK] Real query test passed. Tools used: {data['tools_used']}, Steps: {data['steps_taken']}")
    print(f"     Answer preview: {data['answer'][:200]}")
    return True

if __name__ == "__main__":
    print("Running post-deployment smoke test...")

    if not check_health():
        sys.exit(1)

    if not check_real_query():
        sys.exit(1)

    print("All smoke tests passed.")
    sys.exit(0)

