"""
Multi-Agent System — Industrial AI Copilot Phase 4
====================================================
A supervisor-based multi-agent orchestration system.

Architecture:
    User Query
         ↓
    Supervisor Agent — analyzes query, assigns specialists
         ↓
    ┌─────────────────────────────────────────────┐
    │ Retrieval Agent — documentation search │
    │ Telemetry Agent — live equipment data │
    │ Analysis Agent — calculations + specs │
    │ Safety Agent — risk assessment │
    └─────────────────────────────────────────────┘
         ↓ results collected
    Report Agent — synthesizes into structured report
         ↓
    Final Response

Why multi-agent over single agent:
- Parallel specialist focus — each agent optimized for one role
- Isolated context windows — no token overflow from accumulated tool results
- Explicit reasoning chain — each specialist's contribution is visible
- Scalable — add new specialists without touching existing agents
"""

import logging
import os
import time
from typing import Optional
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.agents.specialist_agents import (
    create_retrieval_agent,
    create_telemetry_agent,
    create_analysis_agent,
    create_safety_agent,
    run_specialist,
)

logger = logging.getLogger(__name__)


# ── Agent Assignment Result ───────────────────────────────────────────────────

@dataclass
class AgentResult:
    agent_name: str
    role: str
    response: str
    latency: float
    status: str = "complete" # complete | skipped | error
    color: str = "#00d4ff"


@dataclass
class MultiAgentResponse:
    final_answer: str
    agent_results: list = field(default_factory=list)
    agents_used: list = field(default_factory=list)
    total_latency: float = 0.0
    query: str = ""


# ── Supervisor ────────────────────────────────────────────────────────────────

class SupervisorAgent:
    """
    Orchestrates specialist agents by analyzing the query and
    deciding which specialists to invoke.
    """

    # Agent color mapping for UI
    AGENT_COLORS = {
        "Retrieval Agent": "#00d4ff", # cyan
        "Telemetry Agent": "#00e676", # green
        "Analysis Agent": "#ffd740", # yellow
        "Safety Agent": "#ff6b6b", # red
    }

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
        )

        # Initialize specialist agents
        logger.info("Initializing specialist agents...")
        self.retrieval_agent = create_retrieval_agent(pipeline)
        self.telemetry_agent = create_telemetry_agent()
        self.analysis_agent = create_analysis_agent()
        self.safety_agent = create_safety_agent(pipeline)
        logger.info("All specialist agents ready.")

    def _decide_agents(self, query: str) -> list:
        """
        Use the LLM to decide which specialists are needed for this query.
        Returns a list of agent names to invoke.
        """
        system_prompt = """You are an orchestrator deciding which specialist agents to invoke.

Available specialists:
- Retrieval Agent: searches industrial documentation, manuals, safety standards
- Telemetry Agent: retrieves live equipment sensor readings and health status
- Analysis Agent: performs spec comparisons, unit conversions, calculations
- Safety Agent: assesses risk, checks safety compliance, recommends safety procedures

Based on the query, return ONLY a comma-separated list of agent names to invoke.
Choose only the agents that are genuinely needed.

Examples:
"What are lockout tagout procedures?" → Retrieval Agent
"Check pump-001 live readings" → Telemetry Agent, Analysis Agent
"Is the plant safe to operate?" → Telemetry Agent, Safety Agent, Retrieval Agent
"Diagnose pump-001 fault" → Telemetry Agent, Analysis Agent, Retrieval Agent, Safety Agent
"Convert 150 psi to bar" → Analysis Agent
"Full plant health report" → Telemetry Agent, Analysis Agent, Safety Agent, Retrieval Agent

Return ONLY the comma-separated agent names. Nothing else."""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ])

        raw = response.content.strip()
        agents = [a.strip() for a in raw.split(",")]
        valid = ["Retrieval Agent", "Telemetry Agent", "Analysis Agent", "Safety Agent"]
        selected = [a for a in agents if a in valid]

        # Default to all agents if parsing fails
        if not selected:
            selected = valid

        logger.info(f"Supervisor selected agents: {selected}")
        return selected

    def _synthesize(self, query: str, agent_results: list) -> str:
        """
        Report Agent — synthesizes all specialist outputs into a
        final structured response.
        """
        # Build context from all specialist results
        context = ""
        for result in agent_results:
            if result.status == "complete" and result.response:
                context += f"\n{'='*40}\n"
                context += f"{result.agent_name.upper()} FINDINGS:\n"
                context += f"{result.response}\n"

        system_prompt = """You are a senior industrial engineering report agent.
Your job is to synthesize findings from multiple specialist agents into a
clear, actionable final response.

RULES:
- Integrate all specialist findings coherently
- Lead with the most critical finding if severity is WARNING or CRITICAL
- Include all citations from the Retrieval Agent and Safety Agent
- State clear recommended actions
- Use severity levels: NORMAL / CAUTION / WARNING / CRITICAL
- Be concise but complete — engineers need actionable information
- Never add information not present in the specialist findings
- Format the response clearly with sections if multiple topics covered"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Original query: {query}\n\n"
                f"Specialist findings:\n{context}\n\n"
                f"Synthesize these findings into a clear final response."
            ))
        ])

        return response.content

    def run(self, query: str) -> MultiAgentResponse:
        """
        Run the full multi-agent pipeline on a query.

        1. Supervisor decides which agents to invoke
        2. Selected specialists run (sequentially for now, parallel in v2)
        3. Report agent synthesizes all findings
        4. Returns structured MultiAgentResponse
        """
        logger.info(f"Multi-agent system processing: {query}")
        total_start = time.time()
        agent_results = []

        # Step 1 — Supervisor decides agents
        selected_agents = self._decide_agents(query)

        # Step 2 — Run selected specialists
        agent_map = {
            "Retrieval Agent": (self.retrieval_agent, "Documentation search and citation"),
            "Telemetry Agent": (self.telemetry_agent, "Live equipment monitoring"),
            "Analysis Agent": (self.analysis_agent, "Engineering calculations and spec checks"),
            "Safety Agent": (self.safety_agent, "Risk assessment and safety compliance"),
        }

        for agent_name in selected_agents:
            if agent_name not in agent_map:
                continue

            agent, role = agent_map[agent_name]
            logger.info(f"Running {agent_name}...")
            start = time.time()

            try:
                response = run_specialist(agent, query)
                latency = round(time.time() - start, 2)
                status = "complete"
            except Exception as e:
                response = f"Agent error: {str(e)}"
                latency = round(time.time() - start, 2)
                status = "error"
                logger.error(f"{agent_name} failed: {e}")

            agent_results.append(AgentResult(
                agent_name = agent_name,
                role = role,
                response = response[:1000], # cap per agent
                latency = latency,
                status = status,
                color = self.AGENT_COLORS.get(agent_name, "#00d4ff"),
            ))

        # Step 3 — Report agent synthesizes
        logger.info("Report Agent synthesizing findings...")
        final_answer = self._synthesize(query, agent_results)

        total_latency = round(time.time() - total_start, 2)

        return MultiAgentResponse(
            final_answer = final_answer,
            agent_results = agent_results,
            agents_used = selected_agents,
            total_latency = total_latency,
            query = query,
        )


# ── Global Instance ───────────────────────────────────────────────────────────

_multi_agent_system: Optional[SupervisorAgent] = None


def get_multi_agent_system() -> Optional[SupervisorAgent]:
    return _multi_agent_system


def initialize_multi_agent_system(pipeline):
    global _multi_agent_system
    _multi_agent_system = SupervisorAgent(pipeline)
    logger.info("Multi-agent system initialized.")
    return _multi_agent_system

