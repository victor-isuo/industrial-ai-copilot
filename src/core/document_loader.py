from langchain_community.document_loaders import PyPDFLoader
# text splitters moved to their own package; import from langchain_text_splitters
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(data_dir: str = "data/raw") -> list:
    """
    Load all PDFs from the data directory.
    Returns a list of LangChain Document objects.
    """
    documents = []
    data_path = Path(data_dir)
    pdf_files = list(data_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDFs found in {data_dir}")
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
        chunk_size=1000,
        chunk_overlap=200,
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