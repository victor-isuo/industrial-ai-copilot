import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Response Schema ---
class RAGResponse(BaseModel):
    """Structured response from the RAG pipeline."""
    answer: str = Field(description="The answer to the user's question")
    sources: list[str] = Field(description="Source documents used to answer")
    confidence: str = Field(description="High, Medium, or Low confidence based on relevance scores")
    caveat: str = Field(description="Any limitations or warnings about the answer")


# --- Prompt Template ---
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert industrial engineering assistant with deep knowledge of 
equipment operation, maintenance, safety standards, and compliance.

Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer confidently,
say so explicitly — do not hallucinate or guess.

For industrial and safety-critical questions, always err on the side of caution.

CONTEXT:
{context}

QUESTION:
{question}

Provide a precise, technically accurate answer. Cite which document and page 
number supports your answer where possible.
""")


class RAGPipeline:
    """
    Full RAG pipeline: retrieve, rerank, generate.
    
    Why structured output:
    - Forces the LLM to be explicit about confidence
    - Surfaces source citations automatically  
    - Makes responses programmatically usable downstream
    - Prepares for agentic use in Phase 2
    """

    def __init__(self, retriever, reranker):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1  # Low temperature for factual accuracy
        )
        logger.info("RAG Pipeline initialized with Gemini 3.1 Flash Lite")

    def _format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            score = doc.metadata.get("relevance_score", "N/A")
            context_parts.append(
                f"[Source {i+1}: {source} | Page {page} | Relevance: {score}]\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _extract_text(self, response) -> str:
        """
        Normalize LLM response content into a plain string.

        Gemini (via langchain_google_genai) does not always return
        `response.content` as a string. Depending on the model and
        whether it includes grounding metadata or multi-part output,
        `.content` can instead be a list of content blocks, e.g.:
            [{"type": "text", "text": "..."}]

        Feeding a list straight into a Pydantic `str` field raises a
        validation error, which is what caused the 500 in production.
        This method handles both shapes safely.
        """
        content = response.content

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

        # Fallback for any unexpected shape — never let this crash the pipeline
        logger.warning(f"Unexpected LLM response content type: {type(content)}")
        return str(content)

    def _assess_confidence(self, documents: list[Document]) -> str:
        """Assess confidence based on reranker relevance scores."""
        if not documents:
            return "Low"
        top_score = documents[0].metadata.get("relevance_score", 0)
        if top_score > 0.5:
            return "High"
        elif top_score > 0.1:
            return "Medium"
        else:
            return "Low"

    def query(self, question: str) -> RAGResponse:
        """
        Full pipeline: retrieve → rerank → generate → structure.
        """
        logger.info(f"Processing query: {question}")

        # Step 1: Hybrid retrieval
        raw_results = self.retriever.invoke(question)
        logger.info(f"Retrieved {len(raw_results)} candidates")

        # Step 2: Rerank
        reranked = self.reranker.rerank(question, raw_results)
        logger.info(f"Reranked to top {len(reranked)} results")

        # Step 3: Format context
        context = self._format_context(reranked)

        # Step 4: Generate answer
        prompt = RAG_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        answer = self._extract_text(response)

        # Step 5: Extract sources
        sources = list(set([
            f"{doc.metadata.get('source', 'unknown')} (Page {doc.metadata.get('page', '?')})"
            for doc in reranked
        ]))

        # Step 6: Assess confidence
        confidence = self._assess_confidence(reranked)

        # Step 7: Build caveat
        caveat = ""
        if confidence == "Low":
            caveat = "Low relevance scores detected. Answer may be incomplete — verify against source documents directly."
        elif confidence == "Medium":
            caveat = "Moderate confidence. Cross-reference with source documents for critical decisions."

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            caveat=caveat
        )


def test_pipeline():
    """End to end pipeline test."""
    from .vector_store import load_vector_store
    from .document_loader import load_documents, chunk_documents
    from .retriever import create_hybrid_retriever
    from .reranker import CohereReranker

    # Load components
    docs = load_documents()
    chunks = chunk_documents(docs)
    vector_store = load_vector_store()
    retriever = create_hybrid_retriever(vector_store, chunks)
    reranker = CohereReranker(top_n=2)

    # Initialize pipeline
    pipeline = RAGPipeline(retriever=retriever, reranker=reranker)

    # Test queries
    test_queries = [
        "What should I doif a gear pump loses suction?",
        "What are the key requirements for ISO 9001:2000 certification?",
        "What PPE is required when handling petroleum products?",
        "How often should preventive maintenance be performed?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        import time
        time.sleep(3)  # Brief pause for readability
        response = pipeline.query(query)
        
        print(f"\nANSWER:\n{response.answer}")
        print(f"\nCONFIDENCE: {response.confidence}")
        print(f"\nSOURCES:")
        for source in response.sources:
            print(f"  - {source}")
        if response.caveat:
            print(f"\nCAVEAT: {response.caveat}")


if __name__ == "__main__":
    test_pipeline()
