from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
import operator
import os
import logging

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "industrial-ai-copilot"

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text(message) -> str:
    """
    Normalize a LangChain message's `.content` into a plain string.

    Gemini (via langchain_google_genai) does not always return `.content`
    as a string — it can return a list of content blocks instead, e.g.:
        [{"type": "text", "text": "..."}]
    Passing that list downstream (into a Pydantic str field, an API
    response, or string concatenation) causes validation/type errors.
    This helper makes `.content` access safe everywhere in the agent.
    """
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)

    logger.warning(f"Unexpected message content type: {type(content)}")
    return str(content)


# --- Agent State ---
class AgentState(TypedDict):
    """
    State that flows through the agent graph.
    Why TypedDict: LangGraph requires typed state for compile-time validation.
    messages uses operator.add as reducer — each node appends to message history.
    """
    messages: Annotated[Sequence, operator.add]


# System prompt built once at import time (not on every agent_node call).
# Rebuilding an identical ~2KB string on every single graph step was pure
# waste; more importantly, keeping it as one reviewable constant makes it
# obvious where instruction order/length changes need to happen.
#
# DESIGN NOTE — why this prompt is shaped the way it is:
# gemini-3.1-flash-lite is a low-latency, lightweight-reasoning model. It is
# explicitly NOT tuned for holding many competing instructions in a long
# system prompt and correctly prioritizing between them. Two failure modes
# were observed against AgentEval's single-agent suite:
#   1. A "DIAGNOSIS WORKFLOW: always follow this sequence" block, sitting
#      *after* the minimal-toolset rules, was being read as an override for
#      any query that merely mentioned equipment + a number — even pure
#      unit conversions and self-contained spec comparisons.
#   2. The full 30-line EQUIPMENT PARAMETER REFERENCE table (only actually
#      needed for the gauge+equipment workflow) was diluting the tool-
#      selection instructions with irrelevant tokens for every other query.
# Fix: shrink the reference table to the compact form actually needed,
# scope DIAGNOSIS WORKFLOW explicitly to diagnosis-language queries only,
# and restate the minimal-toolset gate as the LAST thing before generation
# — the instruction closest to the decision point wins under a lightweight
# model, so put the most important one there instead of trusting recency
# to work in our favor from higher up the prompt.
SYSTEM_PROMPT = """You are an expert industrial maintenance engineer AI assistant with
access to a comprehensive knowledge base of industrial documentation and live equipment
telemetry.

MINIMAL TOOLSET PRINCIPLE — THIS OVERRIDES EVERYTHING BELOW IT:
Before calling any tool, identify the single most specific rule that matches the user's
question. Call ONLY the tool(s) that exact rule requires. A rule matching in a general
sense ("this is technically about equipment") is NOT a reason to also call its tool.
If the user's question already contains all the values needed to answer it (a conversion
request with the number to convert; a comparison between a stated reading and a stated
limit), that is a signal to call exactly ONE tool or ZERO tools — never a cue to verify
with more tools "just in case." Extra tool calls are errors, not thoroughness.

TOOL RULES:
1. spec_checker — ONLY when the user provides BOTH a measured value AND a spec/rated
   limit in the message itself. Nothing else needed.
2. unit_converter — ONLY when the user explicitly asks to convert a unit. Arithmetic
   only — nothing else needed.
3. engineering_calculator — ONLY for explicit numerical calculations.
4. search_industrial_documentation — for a question about equipment, safety, or
   procedures where the answer requires specific documented knowledge (e.g. "what PPE
   is required," "what is the maintenance procedure for X"). This alone answers such
   questions — do not also pull telemetry, spec_checker, or unit_converter.
5. get_equipment_telemetry — ONLY when asked about current/live readings or health of
   specific or all equipment. Not for questions that already contain all needed numbers.
6. list_all_equipment — ONLY when asked about overall plant status or available
   equipment.
7. analyze_equipment_image — when an image is provided for general equipment analysis.
8. analyze_gauge_reading — when an image of a gauge is provided.

GAUGE + EQUIPMENT WORKFLOW (only when a gauge image AND an equipment ID are both given):
   STEP 1: analyze_gauge_reading to extract the gauge value and unit
   STEP 2: get_equipment_telemetry on the mentioned equipment ID
   STEP 3: Match the gauge parameter to the closest telemetry parameter and its
           Normal range (see EQUIPMENT REFERENCE below)
   STEP 4: spec_checker with measured_value = gauge reading, spec_value = normal_max,
           parameter_name, unit = unit from gauge reading
   Never ask the user for spec values if an equipment ID is provided. Never stop after
   Step 1 — complete all 4 steps.

EQUIPMENT REFERENCE (normal / warning / critical):
pump-001 Gear Pump — discharge_pressure psi 340-420/450/500 · suction_pressure psi
10-30/8/5 · flow_rate lpm 130-170 · temperature C 40-75/85/95 · vibration mm/s
0.5-2.3/2.8/4.5 · shaft_speed RPM 1400-1550
pump-002 Centrifugal Pump — discharge_pressure psi 80-120/135/150 · suction_pressure psi
5-20/3/1 · flow_rate lpm 400-600 · temperature C 35-65/75/90 · vibration mm/s
0.3-2.0/2.5/4.0 · shaft_speed RPM 2800-3000
motor-001 Electric Motor — winding_temperature C 40-80/90/105 · bearing_temperature C
35-70/80/95 · current_draw A 30-42/45/50 · vibration mm/s 0.2-1.8/2.3/3.5 · shaft_speed
RPM 1450-1500 · insulation_resistance M-ohm 100-999/50/10
compressor-001 Reciprocating Compressor — discharge_pressure psi 100-145/150/165 ·
suction_pressure psi 12-18/10/8 · discharge_temp C 100-150/165/180 · oil_pressure psi
25-45/20/15 · vibration mm/s 1.0-3.5/4.5/7.0 · rpm 900-1050

DIAGNOSIS WORKFLOW — applies ONLY when the user explicitly asks to diagnose, investigate,
or find out what is wrong with a piece of equipment (e.g. "diagnose pump-001," "what's
wrong with motor-001," "why is compressor-001 vibrating"), with no reading/limit already
given in the message. It does NOT apply just because equipment and a number both appear
in the query — rule 1 (spec_checker) or rule 5 (telemetry) alone may already be the
correct, complete answer; check those first.
   STEP 1: get_equipment_telemetry for live readings
   STEP 2: spec_checker on any parameter outside normal range
   STEP 3: search_industrial_documentation for maintenance procedure
   STEP 4: Return diagnosis + procedure + citations

RESPONSE RULES:
- Always cite sources when using search_industrial_documentation: (Source: [document
  name], Page [X]). Never fabricate document names or page numbers. If retrieval returns
  low confidence or nothing relevant, say "I could not find this in the knowledge base."
  A honest "I don't know" beats a false citation in safety-critical environments.
- For dangerous situations, always recommend immediate action and qualified engineer
  review. Never guess on safety-critical information.
- If a query is completely outside industrial/engineering scope, decline politely and
  use NO tools.

FINAL CHECK BEFORE YOU ACT — re-read this last, it is the instruction that matters most:
Does the user's message already contain every value your answer needs? If yes, call at
most ONE tool (or zero) and answer directly — do not fetch telemetry, do not search
documentation, do not call any tool "to be thorough" or "to verify" beyond the single
rule that matches. Match exactly one rule above and stop there."""


class MaintenanceAgent:
    """
    LangGraph-based maintenance agent.

    Why LangGraph over AgentExecutor:
    - Explicit state management
    - Controllable reasoning loops
    - Production-grade observability
    - Industry standard for agentic systems in 2025
    """

    def __init__(self, pipeline):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1
        )

        # Import and create tools
        from src.tools.retrieval_tool import create_retrieval_tool
        from src.tools.calculator_tool import engineering_calculator
        from src.tools.unit_converter_tool import unit_converter
        from src.tools.spec_checker_tool import spec_checker
        from src.tools.telemetry_tool import get_equipment_telemetry, list_all_equipment
        from src.tools.mcp_tool import query_mcp_industrial_server
        from src.tools.vision_tool import analyze_equipment_image, analyze_gauge_reading

        retrieval_tool = create_retrieval_tool(pipeline)

        self.tools = [
            retrieval_tool,
            engineering_calculator,
            unit_converter,
            spec_checker,
            get_equipment_telemetry,
            list_all_equipment,
            query_mcp_industrial_server,
            analyze_equipment_image,
            analyze_gauge_reading,
        ]

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()
        self.conversation_history = []

        logger.info("LangGraph Maintenance Agent initialized with 9 tools")

    def _build_graph(self):
        """
        Build the LangGraph agent graph.

        Graph structure:
        agent_node → tools_condition → tool_node → agent_node (loop)
                                     → END (when no tools needed)
        """

        def agent_node(state: AgentState):
            """Core reasoning node — decides what to do next."""
            system_message = SystemMessage(content=SYSTEM_PROMPT)

            messages = [system_message] + list(state["messages"])
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        # Create tool execution node
        tool_node = ToolNode(self.tools)

        # Build graph
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)

        # Set entry point
        graph.set_entry_point("agent")

        # Add conditional edges
        graph.add_conditional_edges(
            "agent",
            tools_condition,
        )

        # Tools always return to agent
        graph.add_edge("tools", "agent")

        return graph.compile()

    def run(self, query: str) -> dict:
        """Run the agent on a query with conversation memory."""
        logger.info(f"Agent processing: {query}")

        try:
            # Add user message to history
            self.conversation_history.append(HumanMessage(content=query))

            # Run graph
            result = self.graph.invoke({
                "messages": self.conversation_history
            })

            # Extract final answer
            final_message = result["messages"][-1]
            answer = extract_text(final_message)

            # Update conversation history
            self.conversation_history = list(result["messages"])

            # Extract tool usage
            tools_used = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.append(tc["name"])

            return {
                "answer": answer,
                "tools_used": list(set(tools_used)),
                "steps_taken": len(result["messages"]),
            }

        except Exception as e:
            logger.error(f"Agent failed: {e}")
            error_str = str(e)

            # Handle Groq tool_use_failed gracefully
            if "tool_use_failed" in error_str or "Failed to call a function" in error_str:
                friendly = (
                    "I wasn't able to process that request with the available tools. "
                    "Please check that your query uses valid engineering units and parameters. "
                    "For unit conversions, I support pressure (psi, bar, MPa), "
                    "temperature (°C, °F, K), flow (GPM, LPM), and power (kW, HP)."
                )
            else:
                friendly = f"Agent encountered an error: {error_str}"

            return {
                "answer": friendly,
                "tools_used": [],
                "steps_taken": 0,
            }


def test_agent():
    """Test the maintenance agent end to end."""
    from src.core.document_loader import load_documents, chunk_documents
    from src.core.vector_store import load_vector_store
    from src.core.retriever import create_hybrid_retriever
    from src.core.reranker import CohereReranker
    from src.core.rag_pipeline import RAGPipeline

    docs = load_documents()
    chunks = chunk_documents(docs)
    vector_store = load_vector_store()
    retriever = create_hybrid_retriever(vector_store, chunks)
    reranker = CohereReranker(top_n=5)
    pipeline = RAGPipeline(retriever=retriever, reranker=reranker)

    agent = MaintenanceAgent(pipeline=pipeline)

    test_queries = [
        "What should I do if a gear pump loses suction?",
        "The pump pressure is reading 450 psi but the spec is 380 psi. Is this dangerous?",
        "Convert 150 psi to bar and tell me if that's within normal operating range for industrial pumps",
        "Motor current draw is 48 amps. Rated current is 42 amps. Safe to operate?",
        "What PPE is required for working near rotating equipment?",
        "Convert 75 degrees celsius to fahrenheit",
        "Vibration reading is 2.8 mm/s. ISO limit is 2.3 mm/s. What severity?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        result = agent.run(query)
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nTOOLS USED: {result['tools_used']}")
        print(f"STEPS TAKEN: {result['steps_taken']}")


if __name__ == "__main__":
    test_agent()

