from langchain_groq import ChatGroq
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


# --- Agent State ---
class AgentState(TypedDict):
    """
    State that flows through the agent graph.
    Why TypedDict: LangGraph requires typed state for compile-time validation.
    messages uses operator.add as reducer — each node appends to message history.
    """
    messages: Annotated[Sequence, operator.add]


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
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            groq_api_key=os.getenv("GROQ_API_KEY"),
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
            system_message = SystemMessage(content="""You are an expert industrial
maintenance engineer AI assistant with access to a comprehensive knowledge base
of industrial documentation and live equipment telemetry.

TOOL USAGE RULES — FOLLOW STRICTLY:
1. Call spec_checker ONLY when the user provides BOTH a measured value AND a spec/rated limit
2. Call unit_converter ONLY when the user explicitly asks to convert a unit
3. Call engineering_calculator ONLY for explicit numerical calculations
4. Call search_industrial_documentation for ANY question about equipment, safety, procedures, or maintenance
5. Call get_equipment_telemetry when asked about current readings, live status, or equipment health
6. Call list_all_equipment when asked about overall plant status or available equipment
7. Call analyze_equipment_image when an image is provided for general equipment analysis
8. Call analyze_gauge_reading when an image of a gauge is provided

GAUGE + EQUIPMENT WORKFLOW — FOLLOW THIS EXACT SEQUENCE:
When a gauge image is provided AND any equipment ID is mentioned
(pump-001, pump-002, motor-001, compressor-001):
   STEP 1: Call analyze_gauge_reading to extract the gauge value and unit
   STEP 2: Call get_equipment_telemetry on the mentioned equipment ID
   STEP 3: The telemetry response includes Normal ranges for every parameter
           e.g. "winding_temperature: 65.79 °C (Normal: 40 - 80 °C) [NORMAL]"
   STEP 4: Match the gauge parameter to the closest telemetry parameter
   STEP 5: Extract the normal_min and normal_max from the Normal range
   STEP 6: Call spec_checker with:
           - measured_value = gauge reading from Step 1
           - spec_value = normal_max from Step 3
           - parameter_name = parameter name
           - unit = unit from gauge reading
   NEVER ask the user for spec values if an equipment ID is provided
   NEVER stop after Step 1 — always complete all 6 steps

EQUIPMENT PARAMETER REFERENCE:
pump-001 (Gear Pump):
  - discharge_pressure: Normal 340-420 psi, Warning 450, Critical 500
  - suction_pressure: Normal 10-30 psi, Warning 8, Critical 5
  - flow_rate: Normal 130-170 lpm
  - temperature: Normal 40-75°C, Warning 85, Critical 95
  - vibration: Normal 0.5-2.3 mm/s, Warning 2.8, Critical 4.5
  - shaft_speed: Normal 1400-1550 RPM

pump-002 (Centrifugal Pump):
  - discharge_pressure: Normal 80-120 psi, Warning 135, Critical 150
  - suction_pressure: Normal 5-20 psi, Warning 3, Critical 1
  - flow_rate: Normal 400-600 lpm
  - temperature: Normal 35-65°C, Warning 75, Critical 90
  - vibration: Normal 0.3-2.0 mm/s, Warning 2.5, Critical 4.0
  - shaft_speed: Normal 2800-3000 RPM

motor-001 (Electric Motor):
  - winding_temperature: Normal 40-80°C, Warning 90, Critical 105
  - bearing_temperature: Normal 35-70°C, Warning 80, Critical 95
  - current_draw: Normal 30-42 A, Warning 45, Critical 50
  - vibration: Normal 0.2-1.8 mm/s, Warning 2.3, Critical 3.5
  - shaft_speed: Normal 1450-1500 RPM
  - insulation_resistance: Normal 100-999 MΩ, Warning 50, Critical 10

compressor-001 (Reciprocating Compressor):
  - discharge_pressure: Normal 100-145 psi, Warning 150, Critical 165
  - suction_pressure: Normal 12-18 psi, Warning 10, Critical 8
  - discharge_temp: Normal 100-150°C, Warning 165, Critical 180
  - oil_pressure: Normal 25-45 psi, Warning 20, Critical 15
  - vibration: Normal 1.0-3.5 mm/s, Warning 4.5, Critical 7.0
  - rpm: Normal 900-1050 RPM

DIAGNOSIS WORKFLOW:
For equipment diagnosis — always follow this sequence:
   STEP 1: Call get_equipment_telemetry to get live readings
   STEP 2: Call spec_checker on any parameter outside normal range
   STEP 3: Call search_industrial_documentation for maintenance procedure
   STEP 4: Return diagnosis + procedure + citations

RESPONSE RULES:
- Always cite sources when using search_industrial_documentation
- Include document name AND page number in every citation
- Format: (Source: [document name], Page [X])
- For dangerous situations always recommend immediate action and qualified engineer review
- Never guess on safety-critical information
- If a query is completely outside industrial/engineering scope — decline politely, use NO tools
- When measurements AND a spec limit are both present — MUST call spec_checker, never answer from memory
- NEVER fabricate document names or page numbers
- If search_industrial_documentation returns low confidence or no relevant results,  say: "I could not find this in the knowledge base" — do not invent citations
- A honest "I don't know" is better than a false citation in safety-critical environments""")

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
            answer = final_message.content

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
    