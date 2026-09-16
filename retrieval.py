from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "cuad_contracts"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # must match ingest.py
def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,)
def load_all_documents(vectorstore: Chroma) -> list[Document]:
    raw = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])]
def build_hybrid_retriever(k: int = 5, vector_weight: float = 0.5) -> EnsembleRetriever:
    vectorstore = load_vectorstore()
    all_docs = load_all_documents(vectorstore)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[1 - vector_weight, vector_weight],)
def demo():
    retriever = build_hybrid_retriever()
    queries = [
        "What happens if the contract is terminated for convenience?",
        "30 days written notice",]
    for q in queries:
        print(f"\nQuery: {q}")
        for r in retriever.invoke(q):
            print(f"  [{r.metadata.get('clause_category')}] {r.metadata.get('contract')} -> {r.page_content[:120]}...")

if __name__ == "__main__":
    demo()