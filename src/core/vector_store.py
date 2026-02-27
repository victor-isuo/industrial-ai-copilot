from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VECTOR_STORE_PATH = "data/vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model():
    """
    Using all-MiniLM-L6-v2 for embeddings.
    
    Why this model:
    - Runs locally, no API cost
    - Fast and lightweight
    - Strong performance on technical/industrial text
    - 384 dimensions — good balance of quality vs storage
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


def create_vector_store(chunks: list) -> Chroma:
    """
    Create and persist a ChromaDB vector store from document chunks.
    """
    logger.info("Creating vector store...")
    embeddings = get_embedding_model()
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    logger.info(f"Vector store created with {vector_store._collection.count()} documents")
    return vector_store


def load_vector_store() -> Chroma:
    """
    Load existing vector store from disk.
    Call this instead of create_vector_store after first run.
    """
    if not Path(VECTOR_STORE_PATH).exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_PATH}. "
            "Run create_vector_store first."
        )
    
    logger.info("Loading existing vector store...")
    embeddings = get_embedding_model()
    
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    
    logger.info(f"Loaded vector store with {vector_store._collection.count()} documents")
    return vector_store


if __name__ == "__main__":
    from .document_loader import load_documents, chunk_documents
    
    # Load and chunk documents
    docs = load_documents()
    chunks = chunk_documents(docs)
    
    # Create vector store
    vector_store = create_vector_store(chunks)
    
    # Test retrieval
    test_query = "What is the VESTAPOMP gear pump?"
    results = vector_store.similarity_search(test_query, k=3)
    
    print(f"\n--- RETRIEVAL TEST ---")
    print(f"Query: {test_query}")
    print(f"Top {len(results)} results:\n")
    
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', 'unknown')}")
        print(f"Content: {doc.metadata.get('page_content', doc.page_content[:200])}")
        print("---")