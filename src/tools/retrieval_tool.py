from langchain.tools import tool
from langchain_core.documents import Document
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def create_retrieval_tool(pipeline):
    """
    Factory function that creates a retrieval tool bound to the RAG pipeline.
    Why factory pattern: the pipeline is initialized at runtime, not import time.
    The agent gets a fully configured tool without managing pipeline state itself.
    """

    @tool
    def search_industrial_documentation(query: str) -> str:
        """
        Search the industrial documentation knowledge base.
        Use this tool when you need to find information from equipment manuals,
        safety standards, maintenance guides, or safety datasheets.
        Returns relevant excerpts with source citations.

        Args:
            query: The specific question or topic to search for
        """
        logger.info(f"Retrieval tool called with query: {query}")

        try:
            response = pipeline.query(query)

            # Truncate answer to prevent token overflow
            answer = response.answer[:800] if len(response.answer) > 800 else response.answer

            # Build result string — must be initialized before use
            result  = f"ANSWER: {answer}\n\n"
            result += f"CONFIDENCE: {response.confidence}\n\n"
            result += "SOURCES — YOU MUST CITE THESE IN YOUR RESPONSE:\n"
            for source in list(response.sources)[:3]:
                result += f"  - {source}\n"
            result += "\nIMPORTANT: Always cite sources in your final response including document name AND page number.\n"
            result += "Format: (Source: [document name], Page [X])\n"

            if response.caveat:
                result += f"\nCAVEAT: {response.caveat[:200]}"

            return result

        except Exception as e:
            logger.error(f"Retrieval tool failed: {e}")
            return f"Retrieval failed: {str(e)}"

    return search_industrial_documentation