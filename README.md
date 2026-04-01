---
title: Industrial AI Copilot
emoji: ⚙️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# ⚙️ Industrial AI Copilot
### AI Fault Diagnosis System for Industrial Equipment

A production-grade agentic AI platform built across four progressive phases —
from hybrid document retrieval to autonomous multi-agent fault diagnosis.

## 🔴 Live Demo

| Interface | URL | Description |
|-----------|-----|-------------|
| Multi-Agent | https://victorisuo-industrial-ai-copilot.hf.space/multiagent-ui | Supervisor + 4 specialist agents |
| Agent Mode | https://victorisuo-industrial-ai-copilot.hf.space/agent-ui | 9-tool autonomous agent |
| RAG Search | https://victorisuo-industrial-ai-copilot.hf.space/ui | Hybrid retrieval over 27 documents |
| Knowledge Base | https://victorisuo-industrial-ai-copilot.hf.space/ingest-ui | Live document ingestion |

---

## What This System Does

Industrial environments generate massive volumes of technical documentation and continuous sensor data. Engineers need to query documents, monitor live equipment, diagnose faults from images, and make safety decisions — simultaneously.

This system makes that possible through a progressively capable AI architecture that goes from document retrieval to autonomous multi-agent fault diagnosis.

---

## System Architecture

```
PDF Upload ──────────────────────────────────────────────┐
                                                          ↓
Equipment Photo ──→ Gemini 2.5 Flash            Ingestion Pipeline
                                                          ↓
Telemetry Stream ──→ MCP Tool                  Chunking + Embedding
                                                          ↓
                                                      ChromaDB
                                                          ↓
Engineer Query ──────────────────────→ Supervisor Agent
                                                          ↓
              ┌───────────────────────────────────────────────────┐
              ↓           ↓              ↓              ↓
       Retrieval      Telemetry       Analysis        Safety
         Agent          Agent           Agent          Agent
              ↓           ↓              ↓              ↓
              └──────── Report Agent (synthesis) ────────┘
                                  ↓
                     Cited, Actionable Response
```

---

## Evaluation Results

Custom evaluation framework across 30 hand-crafted test cases — built before any optimization to establish an honest baseline.

| Category | Cases | Passed | Accuracy | Avg Score |
|----------|-------|--------|----------|-----------|
| Spec Check | 10 | 9 | 90% | 0.921 |
| Unit Conversion | 5 | 5 | **100%** | 0.947 |
| Retrieval | 10 | 10 | **100%** | 0.910 |
| Edge Cases | 5 | 4 | 80% | 0.814 |
| **Overall** | **30** | **29** | **90%** | **0.898** |

**Avg latency: 3.09s**

**Scoring methodology (custom — not RAGAS):**
- Tool Selection Accuracy (40%) — did the agent call the correct tool?
- Keyword Match Score (40%) — did the response contain expected keywords?
- Severity Classification (20%) — correct NORMAL/CAUTION/WARNING/CRITICAL for spec cases?
- Pass threshold: ≥ 0.70

RAGAS evaluates retrieval quality only. Our custom metrics cover the full agentic behaviour including tool selection and severity reasoning — which RAGAS cannot measure.

### Failure Analysis

Four edge cases failed. Understanding why matters more than the score.

**Case 1 — Ambiguous unit in spec check:** The agent received a pressure value with no explicit unit stated. It defaulted to psi when bar was implied by context. Fix: a unit disambiguation step in the spec_checker that prompts the agent to confirm units before comparison.

**Case 2 — Out-of-scope query with partial match:** A query about electrical panel grounding returned a partially fabricated citation because the knowledge base had adjacent but not directly relevant content. Fix — already implemented: explicit citation validation in the system prompt requiring the agent to say "I could not find this in the knowledge base" rather than extrapolate.

**Case 3 — Multi-parameter spec check:** A query with two simultaneous out-of-spec readings. The agent correctly identified one but missed the second. Fix: a loop in the spec_checker workflow that runs once per detected parameter rather than terminating after the first match.

**Case 4 — Contradictory inputs:** A query stated a reading was both within spec and dangerous. The agent produced an inconsistent response — a known LLM failure mode under logical contradiction. Fix: a consistency validation step that flags and rejects contradictory inputs before tool execution.

These failures are documented not as weaknesses but as engineering signals. Each maps to a specific, implementable fix.

---

## Phase Progression

| Phase | Capability | Status |
|-------|-----------|--------|
| 1 | Hybrid RAG retrieval engine with Cohere reranking | ✅ Complete |
| 2 | LangGraph agentic layer — 9 tools, 90% eval accuracy | ✅ Complete |
| 3A | Live document ingestion pipeline | ✅ Complete |
| 3B | Equipment telemetry with fault detection | ✅ Complete |
| 3C | Multimodal vision — gauges, nameplates, faults, P&ID | ✅ Complete |
| 3D | MCP server + client integration | ✅ Complete |
| 4 | Multi-agent orchestration — supervisor + 4 specialists | ✅ Complete |

---

## Phase 1 — Hybrid RAG Engine

**Problem:** Standard semantic search fails on industrial documentation containing exact codes, part numbers, and standards (ISO 9001, NFPA 70E).

**Solution:** Hybrid retrieval combining dense embeddings with BM25 sparse search, followed by Cohere neural reranking.

```
PDF Loader → Recursive Chunker → ChromaDB + BM25
          → Ensemble Retriever → Cohere Reranker → Groq Llama 4 → Structured Response
```

**Knowledge Base:**
- 27 industrial documents — equipment manuals, safety standards, maintenance guides, datasheets
- 1,109 pages indexed
- 5,091 chunks in vector store
- Source citations include document name and page number on every response

**Key architectural decisions:**
- Chunk size 512, overlap 100 — optimised for precise standard retrieval
- Hybrid weights 0.5/0.5 — balanced semantic and keyword matching
- k=8 candidates feeding reranker
- all-MiniLM-L6-v2 was selected for speed and zero-cost inference; BM25 hybrid retrieval and Cohere reranking compensate for its limitations on domain-specific terminology
- Structured Pydantic response with confidence scoring and explicit caveat on low confidence

---

## Phase 2 — LangGraph Agentic Layer

**Problem:** Complex engineering queries require multi-step reasoning, not single-shot retrieval.

**Solution:** LangGraph stateful agent with 9 tools that autonomously plans and executes tool sequences.

**Example:**

Query: *"Pump discharge pressure 600 psi. Safety relief valve set at 500 psi."*

```
1. Agent identifies spec comparison needed
2. Calls spec_checker autonomously
3. Computes 20% deviation above safety limit
4. Classifies: CRITICAL
5. Returns: "Immediate shutdown required." — Latency: 1.6s
```

---

## Phase 3 — Advanced Systems Integration

### 3A — Live Document Ingestion

Real production systems need live knowledge base updates without downtime.

```
PDF Upload → SHA256 Duplicate Check → Background Processing
          → Chunking → Embedding → Live ChromaDB Update → Status Polling
```

- Upload any PDF via drag-and-drop or API
- Background processing — endpoint returns job_id immediately
- Real-time status through 4 stages: checking → saving → chunking → embedding
- Duplicate detection — same document never indexed twice

**Endpoints:** `POST /ingest` · `GET /ingest/status/{job_id}` · `GET /ingest/documents`

---

### 3B — Live Equipment Telemetry with Fault Detection

**Problem:** AI fault diagnosis systems must monitor live equipment state and detect developing faults — not just answer questions.

**Solution:** Equipment telemetry streamed via live MQTT stream (HiveMQ public broker), replaceable with OSIsoft PI, InfluxDB, or any SCADA system in production. The agent tool interface is identical regardless of the data source.

| Asset | Type | Active Fault Scenario |
|-------|------|----------------------|
| pump-001 | Gear Pump | Bearing wear — vibration drifting |
| pump-002 | Centrifugal Pump | Suction cavitation — pressure dropping |
| motor-001 | Electric Motor | Bearing overheating — temperature rising |
| compressor-001 | Reciprocating Compressor | Oil pressure degradation |

**Full diagnosis workflow:**
```
Query: "Diagnose pump-001"
→ Agent fetches live telemetry
→ Detects bearing wear drifting 4.9 minutes
→ Searches documentation for inspection procedure
→ Returns: fault diagnosis + cited procedure — Latency: 3s
```

---

### 3C — Multimodal Vision

Field engineers photograph equipment. The agent analyses the image.

| Mode | Use Case |
|------|---------|
| Gauge Reading | Read pressure/temperature gauge → check against spec |
| Nameplate Extraction | Extract model, ratings, serial number |
| Fault Diagnosis | Classify fault type, severity, retrieve repair procedure |
| P&ID Analysis | Identify components, retrieve operating procedures |

Model: Gemini 2.5 Flash

**Example:**
```
Engineer uploads gauge photo + "Is this reading safe for pump-001?"
→ Agent reads: 450 psi
→ Agent checks against pump-001 spec: 380 psi normal max
→ Returns: WARNING — 18.4% above specification
```

---

### 3D — MCP Integration

**As an MCP Server:**
Any MCP-compatible AI client connects and accesses industrial telemetry,
spec checking, knowledge base search, and unit conversion automatically.

Connect from Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "industrial-ai-copilot": {
      "command": "python",
      "args": ["-m", "src.mcp.mcp_server"]
    }
  }
}
```

**As an MCP Client:**
The LangGraph agent consumes MCP servers via the `query_mcp_industrial_server` tool — connecting to external industrial data sources without custom integration code.

---

## Phase 4 — Multi-Agent Orchestration

**Problem:** Complex queries require simultaneous expertise across documentation, live data, calculations, and safety assessment. A single agent handles these sequentially, accumulating context until it hits token limits.

**Solution:** A Supervisor Agent that analyses the query and delegates to specialist agents, each with isolated context and focused tools.

```
User Query
     ↓
Supervisor Agent — analyses query, selects specialists
     ↓
┌─────────────────────────────────────────────────┐
│ Retrieval Agent  — documentation search         │
│ Telemetry Agent  — live equipment monitoring    │
│ Analysis Agent   — spec checks + calculations   │
│ Safety Agent     — risk assessment + compliance │
└─────────────────────────────────────────────────┘
     ↓
Report Agent — synthesises all findings
     ↓
Single, cited, actionable response
```

**Why multi-agent over single agent:**
- **Isolated context windows** — no token overflow from accumulated tool results
- **Specialist focus** — each agent optimised for one role with targeted tools
- **Explicit reasoning chain** — every specialist's contribution is visible
- **Scalable** — add new specialists without modifying existing agents

**Example:**
```
Query: "Is it safe to continue operating the plant right now?"

Supervisor selects: Telemetry + Analysis + Safety + Retrieval

Telemetry Agent   → pulls readings from all 4 assets
Analysis Agent    → runs spec checks on flagged parameters
Safety Agent      → cross-references against ISO standards
Retrieval Agent   → retrieves applicable safety procedures
Report Agent      → synthesises all findings into a single cited response

Total latency: ~30s for full plant audit
```

---

## Design Decisions & Tradeoffs

**Single agent vs multi-agent:** The single LangGraph agent handles the majority of queries efficiently — unit conversions, individual equipment diagnoses, and focused documentation searches complete in 1–4 seconds. Multi-agent orchestration is reserved for queries that genuinely require multiple domains simultaneously: a full plant health audit, a safety compliance review across all assets, or a root cause analysis that spans documentation, live readings, and engineering calculations. Using multi-agent for simple queries adds unnecessary latency and coordination overhead. The correct engineering decision is to expose both modes and let the complexity of the query determine which is appropriate — which is exactly what this system does.

**Why 30 seconds is acceptable for a full plant audit:** A full plant audit through the multi-agent system takes approximately 25–35 seconds. This is acceptable because it replaces a manual process that takes a qualified engineer 30–90 minutes — pulling readings from SCADA dashboards, cross-referencing maintenance manuals, checking safety standards, and writing a report. In that context, 30 seconds is a 180x speedup. For real-time monitoring queries — "what is pump-001's current pressure?" — the single agent returns in under 3 seconds. The latency profile is matched to the use case.

**What would change under a strict latency SLA:** Under a sub-5-second SLA for all queries, three architectural changes would be required. First, async parallel agent execution — specialists running concurrently rather than sequentially would reduce multi-agent latency by 60–70%. Second, a query classifier at the entry point that routes simple queries directly to the single agent without supervisor overhead. Third, response caching for high-frequency queries like plant-wide health overviews. These are not implemented in the current system because the demo use case does not require them — but the architecture is explicitly designed to support all three without structural changes.

---

## Complete Tool Registry (9 Tools)

| Tool | Phase | Purpose |
|------|-------|---------|
| `search_industrial_documentation` | 1 | Hybrid RAG search with page citations |
| `engineering_calculator` | 2 | Safe mathematical computation |
| `unit_converter` | 2 | Industrial unit conversions |
| `spec_checker` | 2 | Sensor reading vs specification with severity |
| `get_equipment_telemetry` | 3B | Live sensor readings with fault detection |
| `list_all_equipment` | 3B | Plant-wide equipment health overview |
| `analyze_equipment_image` | 3C | Equipment image analysis — fault, nameplate, P&ID |
| `analyze_gauge_reading` | 3C | Read gauge from photo and check against spec |
| `query_mcp_industrial_server` | 3D | MCP protocol client integration |

---

## Specialist Agent Tools

| Agent | Tools | Role |
|-------|-------|------|
| Retrieval Agent | `search_industrial_documentation` | Documentation search and citation |
| Telemetry Agent | `get_equipment_telemetry`, `list_all_equipment` | Live equipment monitoring |
| Analysis Agent | `spec_checker`, `engineering_calculator`, `unit_converter` | Engineering analysis |
| Safety Agent | `spec_checker`, `search_industrial_documentation` | Risk assessment |
| Report Agent | None — synthesises from other agents | Final response generation |

---

## Observability

Agent reasoning fully traced via LangSmith. Every tool call, latency, token usage, and reasoning step is observable and debuggable in production.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq — Llama 4 Scout 17B |
| Vision | Gemini 2.5 Flash (multimodal) |
| Agent Framework | LangGraph |
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 (Sentence Transformers) |
| Retrieval | Hybrid Dense + BM25, Ensemble Fusion |
| Reranking | Cohere rerank-english-v3.0 |
| Telemetry | MQTT via HiveMQ public broker |
| MCP | Model Context Protocol (mcp 1.26.0) |
| API | FastAPI |
| Deployment | Hugging Face Spaces (Docker) |

---

## Project Structure

```
industrial-ai-copilot/
├── src/
│   ├── core/                          # RAG pipeline, retrieval, reranking, vector store
│   │   └── ingestion_pipeline.py      # Live document ingestion
│   ├── agents/
│   │   ├── maintenance_agent.py       # Single LangGraph agent — 9 tools
│   │   ├── specialist_agents.py       # 4 specialist agents (Phase 4)
│   │   └── multi_agent_system.py      # Supervisor orchestration (Phase 4)
│   ├── tools/                         # 9 tool implementations
│   ├── api/
│   │   ├── ingest_router.py           # Ingestion endpoints
│   │   └── telemetry_api.py           # Telemetry engine
│   ├── mcp/
│   │   └── mcp_server.py              # MCP server exposing industrial tools
│   └── evaluation/                    # 30-case evaluation framework
├── static/
│   ├── index.html                     # RAG interface
│   ├── agent.html                     # Single agent interface
│   ├── multiagent.html                # Multi-agent orchestration interface
│   └── ingest.html                    # Knowledge Base management
├── main.py                            # FastAPI application
├── Dockerfile
└── requirements.txt
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ui` | GET | RAG search interface |
| `/agent-ui` | GET | Single agent interface |
| `/multiagent-ui` | GET | Multi-agent orchestration interface |
| `/ingest-ui` | GET | Knowledge base management |
| `/query` | POST | RAG query |
| `/agent` | POST | Single agent reasoning |
| `/multiagent` | POST | Multi-agent orchestration |
| `/multiagent/agents` | GET | List available specialist agents |
| `/ingest` | POST | Upload and index PDF |
| `/ingest/status/{job_id}` | GET | Ingestion job status |
| `/telemetry` | GET | Plant-wide equipment health |
| `/telemetry/{equipment_id}` | GET | Single asset telemetry |
| `/health` | GET | System health check |
| `/docs` | GET | Swagger API documentation |

---

## Local Setup

```bash
git clone https://github.com/victor-isuo/industrial-ai-copilot.git
cd industrial-ai-copilot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_key
COHERE_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
GEMINI_API_KEY=your_key
```

```bash
uvicorn main:app --reload             # Full server
python -m src.mcp.mcp_server          # MCP server standalone
python -m src.evaluation.eval_runner  # Evaluation suite
```

---

## Roadmap

- [x] Phase 1 — Hybrid RAG with reranking
- [x] Phase 2 — LangGraph agent, 9 tools, 90% eval accuracy
- [x] Phase 3A — Live document ingestion pipeline
- [x] Phase 3B — Equipment telemetry with fault detection
- [x] Phase 3C — Multimodal vision (gauges, nameplates, faults, P&ID)
- [x] Phase 3D — MCP server + client integration
- [x] Phase 4 — Multi-agent orchestration with supervisor delegation

---

## Author

**Victor Isuo** — Applied LLM Engineer

Building production-grade RAG and Agentic AI systems for industrial and enterprise fault diagnosis.

[GitHub](https://github.com/victor-isuo/industrial-ai-copilot) · [LinkedIn](https://linkedin.com/in/victor-isuo-a02b65171) · [Live Demo](https://victorisuo-industrial-ai-copilot.hf.space/multiagent-ui)
