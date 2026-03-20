---
title: Industrial AI Copilot
emoji: ⚙️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---


# ⚙️ Industrial AI Copilot

A production-grade RAG and agentic AI platform for industrial engineering documentation.
Built to demonstrate end-to-end LLM system design — from hybrid retrieval to autonomous
multi-tool reasoning, live telemetry integration, document ingestion pipelines, and MCP connectivity.

## 🔴 Live Demo

| Interface | URL | Description |
|-----------|-----|-------------|
| RAG Search | https://victorisuo-industrial-ai-copilot.hf.space/ui | Hybrid retrieval over 27 documents |
| Agent Mode | https://victorisuo-industrial-ai-copilot.hf.space/agent-ui | Autonomous 7-tool reasoning agent |
| Knowledge Base | https://victorisuo-industrial-ai-copilot.hf.space/ingest-ui | Live document ingestion dashboard |

---

## What This System Does

Industrial environments generate massive volumes of technical documentation —
equipment manuals, safety datasheets, maintenance guides, compliance standards.
At the same time, physical equipment generates continuous sensor data that needs
to be interpreted against those documents in real time.

This platform makes that documentation queryable, reasoned over, and actionable
through a progressively capable AI system — from hybrid retrieval to live telemetry
diagnosis to external system integration via MCP.

---

## Architecture Overview

```
PDF Upload ──────────────────────────────────────────┐
                                                      ↓
Equipment Photo ──→ Vision API (Phase 3C) Ingestion Pipeline
                                                      ↓
Telemetry API ──→ MCP Tool Chunking + Embedding
                                                      ↓
                                                  ChromaDB
                                                      ↓
Engineer Query ──────────────────→ LangGraph Agent (7 Tools)
                                                      ↓
                         ┌─────────────────────────────────────┐
                         ↓ ↓ ↓ ↓
                    RAG Search Spec Checker Telemetry MCP Server
                         ↓ ↓ ↓ ↓
                         └─────── Cited, Actionable Response ───┘
```

---

## Phase Progression

| Phase | Capability | Status |
|-------|-----------|--------|
| 1 | Hybrid RAG retrieval engine with Cohere reranking | ✅ Complete |
| 2 | LangGraph agentic layer — 4 tools, 90% eval accuracy | ✅ Complete |
| 3 | Live ingestion pipeline + telemetry + MCP integration | ✅ Complete |
| 3C | Multimodal vision — equipment photo analysis | 🔨 In Progress |
| 4 | Multi-agent orchestration | 📅 Planned |

---

## Phase 1 — Hybrid RAG Engine

**Problem:** Standard semantic search fails on industrial documentation containing
exact codes, part numbers, and standards (e.g. ISO 9001:2000, NFPA 70E).

**Solution:** Hybrid retrieval combining dense embeddings with BM25 sparse search,
followed by Cohere neural reranking.

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
- Chunk size 512, overlap 100 — optimized for precise standard retrieval
- Hybrid weights 0.5/0.5 — balanced semantic and keyword matching
- k=8 retrieval candidates feeding reranker
- Structured Pydantic response with confidence scoring and explicit caveat on low confidence

---

## Phase 2 — LangGraph Agentic Layer

**Problem:** Complex engineering queries require multi-step reasoning,
not single-shot retrieval.

**Solution:** LangGraph stateful agent that autonomously plans and executes
tool sequences based on query intent.

```
User Query → LangGraph Agent → Tool Selection → Execution → Structured Response
```

**Tools (Phase 2):**

| Tool | Purpose |
|------|---------|
| `search_industrial_documentation` | Hybrid RAG search with cited page numbers |
| `spec_checker` | Compare sensor readings against specs with NORMAL/CAUTION/WARNING/CRITICAL severity |
| `engineering_calculator` | Safe mathematical computation for engineering analysis |
| `unit_converter` | Industrial unit conversions — pressure, flow, temperature, power, torque |

**Example agent workflow:**

Query: *"Pump discharge pressure 600 psi. Safety relief valve set at 500 psi."*

```
1. Agent identifies spec comparison needed
2. Calls spec_checker autonomously
3. Computes 20% deviation above safety relief valve
4. Classifies severity: CRITICAL
5. Returns: "Immediate shutdown required. Do not continue operation."
   Latency: 1.6s
```

---

## Evaluation Results — Phase 2

Custom evaluation framework built from scratch across 30 hand-crafted test cases.

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
- Composite pass threshold: ≥ 0.70

RAGAS was evaluated but not used — it covers retrieval quality only.
Our custom metrics cover the full agentic behavior including tool selection and severity reasoning.

---

## Phase 3 — Advanced Systems Integration

### 3A — Live Document Ingestion Pipeline

**Problem:** Real production systems need live knowledge base updates —
not a static index that requires server restarts.

**Solution:** A `/ingest` API endpoint that accepts PDF uploads, processes them
in the background, and updates the ChromaDB index without any downtime.

```
PDF Upload → SHA256 Duplicate Check → Background Processing
          → Chunking → Embedding → Live ChromaDB Update
          → Job Status Polling → Completion Notification
```

**Capabilities:**
- Upload any PDF via drag-and-drop UI or API
- Background processing — endpoint returns immediately with job_id
- Real-time status polling through 4 stages: checking → saving → chunking → embedding
- SHA256 duplicate detection — same document never indexed twice
- Automatic filename conflict resolution
- Failed ingestion cleanup — partial files removed on error
- Full ingestion history with timestamps and chunk counts

**Endpoints:**
- `POST /ingest` — upload and index a PDF
- `GET /ingest/status/{job_id}` — poll ingestion progress
- `GET /ingest/jobs` — full ingestion history
- `GET /ingest/documents` — list all indexed documents

---

### 3B — Live Equipment Telemetry with Fault Detection

**Problem:** Real industrial AI systems don't just answer questions —
they monitor live equipment state and detect developing faults.

**Solution:** A simulated telemetry engine modeling realistic sensor behavior
with time-based drift and fault injection across 4 equipment assets.

```
Equipment Registry → Sensor Simulation → Drift Engine → Fault Detection
                  → Severity Classification → Agent Tool → Diagnosis
```

**Equipment modeled:**
| Asset | Type | Parameters |
|-------|------|-----------|
| pump-001 | Gear Pump | Pressure, flow, temperature, vibration, shaft speed |
| pump-002 | Centrifugal Pump | Pressure, suction, flow, temperature, vibration |
| motor-001 | Electric Motor | Winding temp, bearing temp, current, vibration, insulation |
| compressor-001 | Reciprocating Compressor | Discharge pressure, oil pressure, temperature, RPM |

**Fault scenarios (active, time-drifting):**
- pump-001 — Developing bearing wear (vibration drifting up)
- pump-002 — Suction cavitation (suction pressure dropping)
- motor-001 — Bearing overheating (temperature rising)
- compressor-001 — Oil pressure degradation (pressure dropping)

**Full diagnosis workflow:**
```
Query: "Diagnose pump-001"
  → Agent calls get_equipment_telemetry(pump-001)
  → Detects developing bearing wear, vibration drifting for 4.9 minutes
  → Agent calls search_industrial_documentation for bearing inspection procedure
  → Returns: fault diagnosis + maintenance procedure + page citations
```

> In production, this module is replaced by calls to a plant historian API
> (OSIsoft PI, InfluxDB), MQTT broker, or SCADA system REST endpoint.
> The agent tool interface is identical regardless of data source.

**New tools added:**
| Tool | Purpose |
|------|---------|
| `get_equipment_telemetry` | Live sensor readings with severity classification |
| `list_all_equipment` | Plant-wide health overview with alert counts |

---

### 3D — MCP Integration

**Problem:** Custom integrations for every AI client are not scalable.
The industry needs a standard protocol for AI-to-system connectivity.

**Solution:** Full MCP (Model Context Protocol) implementation on both sides
of the protocol — server and client.

```
MCP Server (this system) ←→ Any MCP-compatible AI client
                              (Claude Desktop, Cursor, custom agents)

LangGraph Agent → MCP Client Tool → MCP Server → Industrial Data
```

**As an MCP Server:**
The Industrial AI Copilot exposes its capabilities as an MCP server.
Any compliant AI client connects and accesses:
- Live equipment telemetry
- Equipment health overview
- Spec checking
- Knowledge base search
- Unit conversion

Connect from Claude Desktop by adding to `claude_desktop_config.json`:
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
The LangGraph agent includes a `query_mcp_industrial_server` tool that
demonstrates consuming an MCP server — connecting to the industrial data
layer via the standardized protocol rather than direct function calls.

**Why MCP matters:**
Instead of building custom integrations for every AI tool,
one MCP server makes your system accessible to the entire
MCP-compatible ecosystem. This is the production integration
pattern for enterprise AI in 2025.

---

## Complete Tool Registry (7 Tools)

| Tool | Phase | Purpose |
|------|-------|---------|
| `search_industrial_documentation` | 1 | Hybrid RAG search with page citations |
| `engineering_calculator` | 2 | Safe mathematical computation |
| `unit_converter` | 2 | Industrial unit conversions |
| `spec_checker` | 2 | Sensor reading vs specification with severity |
| `get_equipment_telemetry` | 3B | Live sensor readings with fault detection |
| `list_all_equipment` | 3B | Plant-wide equipment health overview |
| `query_mcp_industrial_server` | 3D | MCP protocol client integration |

---

## Observability

Agent reasoning fully traced via LangSmith.

![LangSmith Trace](docs/langsmith_trace.jpg)

Every tool call, latency, token usage, and reasoning step
is observable and debuggable in production.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq — Llama 4 Scout 17B |
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
│ ├── core/ # RAG pipeline, retrieval, reranking, vector store
│ │ └── ingestion_pipeline.py # Live document ingestion
│ ├── agents/ # LangGraph agent definitions
│ ├── tools/ # 7 tool abstractions
│ ├── api/ # FastAPI routers
│ │ ├── ingest_router.py # Ingestion endpoints
│ │ └── telemetry_api.py # Telemetry simulation engine
│ ├── mcp/ # MCP server and client
│ │ └── mcp_server.py # MCP server exposing industrial tools
│ └── evaluation/ # 30-case evaluation framework
├── data/
│ └── raw/ # Source documents (gitignored)
├── static/
│ ├── index.html # RAG interface
│ ├── agent.html # Agent interface
│ └── ingest.html # Knowledge Base management UI
├── main.py # FastAPI application
├── Dockerfile
└── requirements.txt
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ui` | GET | RAG search interface |
| `/agent-ui` | GET | Agent mode interface |
| `/ingest-ui` | GET | Knowledge base management |
| `/query` | POST | RAG query |
| `/agent` | POST | Agentic reasoning |
| `/ingest` | POST | Upload and index PDF |
| `/ingest/status/{job_id}` | GET | Ingestion job status |
| `/ingest/documents` | GET | List indexed documents |
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
GROQ_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_key_here
```

Run:
```bash
uvicorn main:app --reload
```

Run MCP server standalone:
```bash
python -m src.mcp.mcp_server
```

Run evaluation suite:
```bash
python -m src.evaluation.eval_runner
```

---

## Roadmap

- [x] Phase 1 — Hybrid RAG with reranking
- [x] Phase 2 — LangGraph agent, 4 tools, 90% eval accuracy
- [x] Phase 3A — Live document ingestion pipeline
- [x] Phase 3B — Equipment telemetry with fault detection
- [x] Phase 3D — MCP server + client integration
- [ ] Phase 3C — Multimodal vision (equipment photos, gauges, P&ID diagrams)
- [ ] Phase 4 — Multi-agent orchestration

---

## Author

**Victor Isuo** — Applied LLM Systems Engineer

Building production-grade RAG and Agentic AI systems for industrial and enterprise applications.

[GitHub](https://github.com/victor-isuo/industrial-ai-copilot) · [LinkedIn](https://linkedin.com/in/victor-isuo-a02b65171) · [Live Demo](https://victorisuo-industrial-ai-copilot.hf.space/agent-ui)