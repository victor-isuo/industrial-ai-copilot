---
title: Industrial AI Copilot
emoji: ⚙️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# Industrial AI Copilot

A modular RAG and multi-agent platform for industrial engineering documentation.

## What This System Does

Industrial environments generate massive volumes of technical documentation —
equipment manuals, safety datasheets, maintenance logs, compliance specs.
This platform makes that documentation queryable, reasoned over, and actionable
through a progressively capable AI system.

## Architecture Evolution

| Phase | Capability | Status |
|-------|-----------|--------|
| 1 | Hybrid RAG retrieval engine with reranking | 🔨 In Progress |
| 2 | Agentic reasoning layer with tool use | 📅 Planned |
| 3 | Multimodal extension + MCP integration | 📅 Planned |
| 4 | Multi-agent orchestration + evaluation dashboard | 📅 Planned |

## Tech Stack

- **LLM**: Google Gemini
- **Orchestration**: LangChain, LangGraph
- **Vector Store**: ChromaDB
- **Embeddings**: Sentence Transformers
- **Retrieval**: Hybrid (Dense + BM25) with Cohere Reranking
- **API**: FastAPI
- **Deployment**: Render

## Project Structure
```
industrial-ai-copilot/
├── src/
│   ├── core/          # Retrieval engine, embeddings, vector store
│   ├── agents/        # LangGraph agent definitions
│   ├── tools/         # Tool abstractions for agents
│   └── multimodal/    # Image and scanned PDF processing
├── data/
│   ├── raw/           # Source documents
│   └── processed/     # Chunked and embedded documents
├── tests/             # Evaluation and unit tests
├── docs/              # Architecture documentation
└── notebooks/         # Experimentation and analysis
```

## Setup
```bash
git clone https://github.com/victor-isuo/industrial-ai-copilot.git
cd industrial-ai-copilot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

## Evaluation Metrics

Retrieval performance is measured continuously:
- Recall@K
- Mean Reciprocal Rank (MRR)
- Latency per query
- Answer relevance score

## Author

Built by Victor Isuo — Applied LLM Engineer
