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
Telemetry API ──→ MCP Tool                     Chunking + Embedding
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

**Key decisions:**
- Chunk size 512, overlap 100 — optimised for precise standard retrieval
- Hybrid weights 0.5/0.5 — balanced semantic and keyword matching
- k=8 candidates feeding reranker
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

## Evaluation Results — Phase 2

Custom evaluation framework across 30 hand-crafted test cases.

| Category | Cases | Passed | Accuracy | Avg Score |
|----------|-------|--------|----------|-----------|
| Spec Check | 10 | 9 | 90% | 0.921 |
| Unit Conversion | 5 | 5 | **100%** | 0.947 |
| Retrieval | 10 | 10 | **100%** | 0.910 |
| Edge Cases | 5 | 4 | 80% | 0.814 |
| **Overall** | **30** | **29** | **90%** | **0.898** |

**Avg latency: 3.09s**

**Scoring methodology (custom — not RAGAS):**
- Tool Selection Accuracy (40%)
- Keyword Match Score (40%)
- Severity Classification (20%)
- Pass threshold: ≥ 0.70

RAGAS evaluates retrieval quality only. Our custom metrics cover the full agentic behaviour including tool selection and severity reasoning — which RAGAS cannot measure.

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

### 3B — Live Telemetry with Fault Detection

AI fault diagnosis systems monitor live equipment state and detect developing faults.

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
→ Returns: fault diagnosis + cited procedure
   Latency: 3s
```

> In production, this module connects to a plant historian API (OSIsoft PI, InfluxDB),
> MQTT broker, or SCADA system. The agent tool interface is identical regardless of source.

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

**Example — full plant health report:**
```
Query: "Is it safe to continue operating the plant right now?"

Supervisor selects: Telemetry + Analysis + Safety + Retrieval

Telemetry Agent    → pulls readings from all 4 assets
Analysis Agent     → runs spec checks on flagged parameters
Safety Agent       → cross-references against ISO standards
Retrieval Agent    → retrieves applicable safety procedures

Report Agent       → synthesises: overall risk level, per-asset status,
                     cited procedures, recommended actions

Total latency: ~30s for full plant audit
```

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

Agent reasoning fully traced via LangSmith.

![LangSmith Trace](docs/langsmith_trace.jpg)

Every tool call, latency, token usage, and reasoning step is observable and debuggable in production.

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
│   │   └── telemetry_api.py           # Telemetry simulation engine
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
uvicorn main:app --reload          # Full server
python -m src.mcp.mcp_server       # MCP server standalone
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

Building production-grade RAG and Agentic AI systems for industrial and enterprise uses.

[GitHub](https://github.com/victor-isuo/industrial-ai-copilot) · [LinkedIn](https://linkedin.com/in/victor-isuo-a02b65171) · [Live Demo](https://victorisuo-industrial-ai-copilot.hf.space/multiagent-ui)