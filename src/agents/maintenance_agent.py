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

        retrieval_tool = create_retrieval_tool(pipeline)

        self.tools = [
            retrieval_tool,
            engineering_calculator,
            unit_converter,
            spec_checker,
        ]

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()
        self.conversation_history = []

        logger.info("LangGraph Maintenance Agent initialized with 4 tools")

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
of industrial documentation.

TOOL USAGE RULES — FOLLOW STRICTLY:
1. Call spec_checker ONLY when the user provides BOTH a measured value AND a spec/rated limit
2. Call unit_converter ONLY when the user explicitly asks to convert a unit
3. Call engineering_calculator ONLY for explicit numerical calculations
4. Call search_industrial_documentation for ANY question about equipment, safety, procedures, or maintenance
5. For vague symptoms (strange noise, vibration, leaks) — always search documentation first
6. If a query is completely outside industrial/engineering scope — decline politely, use NO tools
7. When measurements AND a spec limit are both present — you MUST call spec_checker, never answer from memory

RESPONSE RULES:
- Always cite sources when using search_industrial_documentation
- Include document name AND page number in every citation
- Format: (Source: [document name], Page [X])
- For dangerous situations always recommend immediate action and qualified engineer review
- Never guess on safety-critical information""")

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
    