"""
Specialist Agents — Industrial AI Copilot Phase 4
===================================================
Four specialist agents, each with a focused role and
a restricted set of tools relevant to their domain.

Each specialist is a compiled LangGraph graph that:
- Has its own system prompt optimized for its role
- Has access only to the tools it needs
- Returns structured output to the supervisor

Specialists:
- RetrievalAgent — searches and cites documentation
- TelemetryAgent — monitors live equipment state
- AnalysisAgent — runs calculations and spec comparisons
- SafetyAgent — assesses risk and flags violations
"""

import logging
import os
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

logger = logging.getLogger(__name__)


# ── Shared State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence, operator.add]


# ── LLM Factory ───────────────────────────────────────────────────────────────

def _get_llm():
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )


# ── Base Agent Builder ────────────────────────────────────────────────────────

def _build_agent(system_prompt: str, tools: list):
    """
    Build a compiled LangGraph agent with a specific system prompt and toolset.
    This is the factory that creates all specialist agents.
    """
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState):
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Retrieval Agent ───────────────────────────────────────────────────────────

def create_retrieval_agent(pipeline):
    """
    Specialist in searching and citing industrial documentation.
    Only has access to the RAG search tool.
    """
    from src.tools.retrieval_tool import create_retrieval_tool
    retrieval_tool = create_retrieval_tool(pipeline)

    system_prompt = """You are a specialist industrial documentation retrieval agent.

Your ONLY job is to search the industrial knowledge base and return
precise, well-cited information.

RULES:
- Always call search_industrial_documentation for every query
- Always include document name AND page number in citations
- Format citations as: (Source: [document name], Page [X])
- If multiple sources found, cite all of them
- Never answer from memory — always search first
- Keep responses concise and factual
- If nothing relevant found, say explicitly: "No relevant documentation found for this query"
- NEVER fabricate document names or page numbers"""

    return _build_agent(system_prompt, [retrieval_tool])


# ── Telemetry Agent ───────────────────────────────────────────────────────────

def create_telemetry_agent():
    """
    Specialist in monitoring live equipment state and detecting faults.
    Has access to telemetry tools only.
    """
    from src.tools.telemetry_tool import get_equipment_telemetry, list_all_equipment

    system_prompt = """You are a specialist industrial telemetry monitoring agent.

Your ONLY job is to retrieve and interpret live equipment sensor data.

RULES:
- Call list_all_equipment first for plant-wide queries
- Call get_equipment_telemetry for specific equipment queries
- Report ALL sensor readings with their normal ranges
- Explicitly flag any WARNING or CRITICAL readings
- Report any developing faults with elapsed drift time
- Always state overall equipment health: NORMAL / CAUTION / WARNING / CRITICAL
- Be precise with numbers — include units always
- Keep response structured and scannable

Available equipment: pump-001, pump-002, motor-001, compressor-001"""

    return _build_agent(
        system_prompt,
        [get_equipment_telemetry, list_all_equipment]
    )


# ── Analysis Agent ────────────────────────────────────────────────────────────

def create_analysis_agent():
    """
    Specialist in engineering calculations and specification comparisons.
    Has access to spec checker, calculator, and unit converter.
    """
    from src.tools.spec_checker_tool import spec_checker
    from src.tools.calculator_tool import engineering_calculator
    from src.tools.unit_converter_tool import unit_converter

    system_prompt = """You are a specialist industrial engineering analysis agent.

Your ONLY job is to perform precise engineering calculations,
unit conversions, and specification comparisons.

RULES:
- Call spec_checker when comparing a measured value against a specification
- Call unit_converter when unit conversion is needed
- Call engineering_calculator for mathematical computations
- Always show your working — state the measured value, spec value, and deviation
- Always state severity: NORMAL / CAUTION / WARNING / CRITICAL
- For CRITICAL findings, state: "IMMEDIATE ACTION REQUIRED"
- For WARNING findings, state: "URGENT INSPECTION RECOMMENDED"
- Be precise — include units in every measurement
- Never guess on safety-critical calculations"""

    return _build_agent(
        system_prompt,
        [spec_checker, engineering_calculator, unit_converter]
    )


# ── Safety Agent ──────────────────────────────────────────────────────────────

def create_safety_agent(pipeline):
    """
    Specialist in risk assessment and safety compliance.
    Has access to spec checker and documentation search.
    """
    from src.tools.spec_checker_tool import spec_checker
    from src.tools.retrieval_tool import create_retrieval_tool
    retrieval_tool = create_retrieval_tool(pipeline)

    system_prompt = """You are a specialist industrial safety assessment agent.

Your ONLY job is to assess risk, identify safety violations,
and retrieve relevant safety procedures and standards.

RULES:
- Search documentation for applicable safety standards (ISO, NFPA, OSHA)
- Use spec_checker to verify readings against safety limits
- Always classify overall risk: LOW / MEDIUM / HIGH / CRITICAL
- For HIGH or CRITICAL risk: recommend immediate shutdown or evacuation
- Always cite safety standards with document name and page number
- Recommend specific PPE requirements when relevant
- Flag any regulatory compliance violations explicitly
- Never downplay safety concerns — err on the side of caution
- Recommend qualified engineer review for any CRITICAL findings"""

    return _build_agent(
        system_prompt,
        [spec_checker, retrieval_tool]
    )


# ── Agent Runner ──────────────────────────────────────────────────────────────

def run_specialist(agent, query: str) -> str:
    """
    Run a specialist agent on a query and extract the final text response.
    Returns empty string on failure.
    """
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        final_msg = result["messages"][-1]
        return final_msg.content if hasattr(final_msg, "content") else str(final_msg)
    except Exception as e:
        logger.error(f"Specialist agent failed: {e}")
        return f"Agent encountered an error: {str(e)}"

