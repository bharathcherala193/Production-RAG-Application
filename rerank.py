from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from retrieval import build_hybrid_retriever

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
def rerank(model: CrossEncoder, query: str, candidates: list[Document], top_k: int = 5) -> list[Document]:
    """Score each (query, candidate) pair together, return the top_k re-ordered by actual relevance."""
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = model.predict(pairs)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]
def show(label: str, docs: list[Document]):
    print(f"  -- {label} --")
    for doc in docs:
        print(f"    [{doc.metadata.get('clause_category')}] {doc.metadata.get('contract')} -> {doc.page_content[:100]}...")

def demo():
    retriever = build_hybrid_retriever(k=20)
    reranker = CrossEncoder(RERANK_MODEL)
    queries = [
        "What happens if the contract is terminated for convenience?",
        "30 days written notice",]
    for q in queries:
        print(f"\nQuery: {q}")
        candidates = retriever.invoke(q)
        show("BEFORE reranking (top 5 of the raw hybrid result)", candidates[:5])
        top = rerank(reranker, q, candidates, top_k=5)
        show("AFTER reranking (top 5 by cross-encoder relevance)", top)
if __name__ == "__main__":
    demo()