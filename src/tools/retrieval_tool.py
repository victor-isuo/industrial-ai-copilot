from langchain.tools import tool
from langchain.schema import Document
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

            # Format results for agent consumption
            result = f"ANSWER: {response.answer}\n\n"
            result += f"CONFIDENCE: {response.confidence}\n\n"
            result += "SOURCES:\n"
            for source in response.sources:
                result += f"  - {source}\n"

            if response.caveat:
                result += f"\nCAVEAT: {response.caveat}"

            return result

        except Exception as e:
            logger.error(f"Retrieval tool failed: {e}")
            return f"Retrieval failed: {str(e)}"

    return search_industrial_documentation