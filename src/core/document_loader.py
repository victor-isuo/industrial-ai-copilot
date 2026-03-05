from langchain_community.document_loaders import PyPDFLoader
# text splitters moved to their own package; import from langchain_text_splitters
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(data_dir: str = None) -> list:
    """
    Load all PDFs. Checks data/raw first, then root directory.
    Supports both local development and Hugging Face deployment.
    """
    documents = []
    
    # Check data/raw first, then root
    search_dirs = ["data/raw", "."]
    if data_dir:
        search_dirs = [data_dir]
    
    pdf_files = []
    for search_dir in search_dirs:
        found = list(Path(search_dir).glob("*.pdf"))
        if found:
            pdf_files = found
            logger.info(f"Found PDFs in: {search_dir}")
            break
    
    if not pdf_files:
        logger.warning("No PDFs found in any location")
        return []
    
    for pdf_path in pdf_files:
        logger.info(f"Loading: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")
            continue
    
    logger.info(f"Total pages loaded: {len(documents)}")
    return documents

def chunk_documents(documents: list) -> list:
    """
    Split documents into chunks using recursive character splitting.
    
    Why recursive character splitting:
    - Respects document structure (paragraphs before sentences before words)
    - Handles inconsistent industrial PDF formatting better than fixed-size
    - Overlap ensures context isn't lost at chunk boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


if __name__ == "__main__":
    # Test the pipeline
    docs = load_documents()
    chunks = chunk_documents(docs)
    
    print(f"\n--- INGESTION SUMMARY ---")
    print(f"Documents loaded: {len(docs)} pages")
    print(f"Chunks created: {len(chunks)}")
    print(f"\n--- SAMPLE CHUNK ---")
    print(f"Content: {chunks[0].page_content[:300]}")
    print(f"Metadata: {chunks[0].metadata}")