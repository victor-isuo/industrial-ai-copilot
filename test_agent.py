from src.core.vector_store import load_vector_store
from src.core.retriever import create_hybrid_retriever
from src.core.document_loader import load_documents, chunk_documents
from src.core.reranker import CohereReranker
from src.core.rag_pipeline import RAGPipeline
from src.agents.maintenance_agent import MaintenanceAgent

vector_store = load_vector_store()
docs = load_documents()
chunks = chunk_documents(docs)
retriever = create_hybrid_retriever(vector_store, chunks)
reranker = CohereReranker(top_n=5)
pipeline = RAGPipeline(retriever=retriever, reranker=reranker)
agent = MaintenanceAgent(pipeline=pipeline)
result = agent.run('Pump pressure 450 psi vs spec 380 psi')
print(result['answer'])