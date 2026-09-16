from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from retrieval import build_hybrid_retriever
from rerank import rerank, RERANK_MODEL
from sentence_transformers import CrossEncoder

OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a contract-analysis assistant. You answer questions using ONLY \
the contract excerpts provided below — never your own outside knowledge.

Rules:
- If the answer is found in the excerpts, answer it clearly and cite the excerpt \
number(s) you used, like [1] or [1][3].
- If the excerpts do NOT contain enough information to answer, say exactly: \
"I couldn't find this in the retrieved contract excerpts." Do not guess or fill \
gaps with general legal knowledge.
- Keep answers concise and grounded in the excerpt text — don't paraphrase away \
specific numbers, dates, or defined terms.
"""


def format_context(docs: list[Document]) -> str:
    """
    Number each excerpt and label it with its source contract — this is
    what lets the model's citations ([1], [2], ...) map back to a real,
    checkable source instead of being an untraceable claim.
    """
    lines = []
    for i, doc in enumerate(docs, start=1):
        contract = doc.metadata.get("contract", "unknown")
        lines.append(f"[{i}] (Source: {contract})\n{doc.page_content}")
    return "\n\n".join(lines)


def answer_question(llm: ChatOllama, reranker: CrossEncoder, retriever, question: str, top_k: int = 5) -> str:
    candidates = retriever.invoke(question)
    top_docs = rerank(reranker, question, candidates, top_k=top_k)

    context = format_context(top_docs)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Contract excerpts:\n\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)

    # Print the source map alongside the answer so you can manually verify
    # every citation the model made actually points to a real chunk.
    print("\n--- Sources ---")
    for i, doc in enumerate(top_docs, start=1):
        print(f"  [{i}] {doc.metadata.get('contract')}")

    return response.content


def demo():
    retriever = build_hybrid_retriever(k=20)
    reranker = CrossEncoder(RERANK_MODEL)
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)  # temperature=0: favor grounded, repeatable answers over creative ones

    questions = [
        "What happens if the contract is terminated for convenience?",
        "Is there a clause about confidentiality?",
    ]
    for q in questions:
        print(f"\n{'=' * 60}\nQuestion: {q}\n{'=' * 60}")
        answer = answer_question(llm, reranker, retriever, q)
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    demo()