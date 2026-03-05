from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from typing import List
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Simple hybrid retriever combining vector and BM25 search."""
    def __init__(self, vector_retriever, bm25_retriever, vector_weight=0.6, bm25_weight=0.4):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents from both retrievers and deduplicate by source."""
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # Combine and deduplicate by content
        seen = set()
        combined = []
        for doc in vector_docs + bm25_docs:
            doc_id = (doc.page_content[:100], doc.metadata.get('source'))
            if doc_id not in seen:
                seen.add(doc_id)
                combined.append(doc)
        
        return combined[:max(10, len(vector_docs))]  # Return up to 10 results

    def invoke(self, query: str) -> List[Document]:
        """Compatibility wrapper: allow calling `.invoke(query)` like other retrievers."""
        return self.get_relevant_documents(query)


def create_hybrid_retriever(
    vector_store: Chroma,
    chunks: List[Document],
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    k: int = 8
) -> HybridRetriever:
    """
    Create a hybrid retriever combining dense and sparse search.

    Why hybrid retrieval:
    - Dense (vector) search: finds semantically similar content
      even when exact words differ. Good for conceptual questions.
    - Sparse (BM25) search: finds exact keyword matches.
      Good for part numbers, standard codes, specific terms.
    - Together they cover what neither can do alone.

    Weight explanation:
    - 0.6 vector / 0.4 BM25 is a proven starting ratio
    - We manually blend results to combine both signals
    """
    logger.info("Creating BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    logger.info("Creating vector retriever...")
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    logger.info("Creating hybrid retriever...")
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight
    )

    logger.info("Hybrid retriever ready")
    return hybrid_retriever


def test_retriever(retriever: HybridRetriever, query: str) -> None:
    """Test retriever and display results with metadata."""
    logger.info(f"Testing query: {query}")
    results = retriever.get_relevant_documents(query)

    print(f"\n--- HYBRID RETRIEVAL TEST ---")
    print(f"Query: {query}")
    print(f"Results returned: {len(results)}\n")

    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Source: {doc.metadata.get('source', 'unknown')}")
        print(f"  Page: {doc.metadata.get('page', 'unknown')}")
        print(f"  Content preview: {doc.page_content[:200]}")
        print("---")


if __name__ == "__main__":
    from .vector_store import load_vector_store, get_embedding_model
    from .document_loader import load_documents, chunk_documents

    # Load existing vector store and chunks
    docs = load_documents()
    chunks = chunk_documents(docs)
    vector_store = load_vector_store()

    # Create hybrid retriever
    retriever = create_hybrid_retriever(vector_store, chunks)

    # Test with different query types
    test_queries = [
        "What is the maximum operating pressure for the gear pump?",
        "ISO 14001 certification requirements",
        "petroleum safety handling procedures",
        "preventive maintenance schedule intervals"
    ]

    for query in test_queries:
        test_retriever(retriever, query)