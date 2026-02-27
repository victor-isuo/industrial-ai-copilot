import cohere
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CohereReranker:
    """
    Reranks retrieved documents using Cohere's rerank API.

    Why reranking:
    - Hybrid retrieval returns good candidates but imperfect ranking
    - Reranker sees query AND document together, not independently
    - Significantly improves precision of top results
    - Cost effective: only rerank top k candidates, not entire corpus
    """

    def __init__(self, top_n: int = 5, model: str = "rerank-english-v3.0"):
        self.client = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.top_n = top_n
        self.model = model
        logger.info(f"Reranker initialized: {model}, top_n={top_n}")

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Rerank documents by relevance to query.
        Returns top_n most relevant documents in ranked order.
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []

        # Extract text content for reranking
        doc_texts = [doc.page_content for doc in documents]

        logger.info(f"Reranking {len(documents)} documents...")

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=doc_texts,
            top_n=self.top_n
        )

        # Rebuild document list in reranked order with scores
        reranked_docs = []
        for result in response.results:
            doc = documents[result.index]
            # Add relevance score to metadata
            doc.metadata["relevance_score"] = round(result.relevance_score, 4)
            reranked_docs.append(doc)

        logger.info(f"Reranking complete. Top score: {reranked_docs[0].metadata['relevance_score']}")
        return reranked_docs


def test_reranker():
    """Test reranker against hybrid retrieval results."""
    from .vector_store import load_vector_store
    from .document_loader import load_documents, chunk_documents
    from .retriever import create_hybrid_retriever

    # Load everything
    docs = load_documents()
    chunks = chunk_documents(docs)
    vector_store = load_vector_store()
    retriever = create_hybrid_retriever(vector_store, chunks)
    reranker = CohereReranker(top_n=5)

    query = "What is the maximum operating pressure for the gear pump?"

    # Get hybrid results
    hybrid_results = retriever.invoke(query)
    
    print(f"\n--- BEFORE RERANKING ---")
    print(f"Query: {query}")
    for i, doc in enumerate(hybrid_results[:5]):
        print(f"\nResult {i+1}:")
        print(f"  Source: {doc.metadata.get('source')}")
        print(f"  Page: {doc.metadata.get('page')}")
        print(f"  Preview: {doc.page_content[:150]}")

    # Rerank results
    reranked = reranker.rerank(query, hybrid_results)

    print(f"\n--- AFTER RERANKING ---")
    for i, doc in enumerate(reranked):
        print(f"\nResult {i+1} (score: {doc.metadata.get('relevance_score')}):")
        print(f"  Source: {doc.metadata.get('source')}")
        print(f"  Page: {doc.metadata.get('page')}")
        print(f"  Preview: {doc.page_content[:150]}")


if __name__ == "__main__":
    test_reranker()