import sys
from mcp.server.mcpserver import MCPServer
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from retrieval import build_hybrid_retriever
from rerank import rerank, RERANK_MODEL
from generate import answer_question, OLLAMA_MODEL

mcp = MCPServer("ask-my-docs")

class Pipeline:
    """Plain class instead of @dataclass — mcp dev's module loader doesn't
    register the module in sys.modules before exec, which breaks dataclasses'
    forward-ref resolution when annotations are stored as strings."""

    def __init__(self, retriever, reranker: CrossEncoder, llm: ChatOllama):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
_pipeline: "Pipeline | None" = None

def get_pipeline() -> Pipeline:
    """
    Lazily build the retriever / reranker / llm once per server process,
    rather than per tool call — these are expensive to construct
    (embedding model load, BM25 index build, cross-encoder load).
    """
    global _pipeline
    if _pipeline is None:
        print("Building retrieval pipeline (first call)...", file=sys.stderr)
        retriever = build_hybrid_retriever(k=20)
        reranker = CrossEncoder(RERANK_MODEL)
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        _pipeline = Pipeline(retriever=retriever, reranker=reranker, llm=llm)
        print("Pipeline ready.", file=sys.stderr)
    return _pipeline

@mcp.tool()
def ask_contract_question(question: str, top_k: int = 5) -> str:
    """
    Answer a question about the ingested contracts using hybrid retrieval
    (BM25 + vector search), cross-encoder reranking, and citation-enforced
    generation. The answer is grounded only in retrieved contract excerpts
    and includes [n] citations back to the source chunks.

    Args:
        question: A natural-language question about the contracts
            (e.g. "Is there a non-compete restriction?").
        top_k: Number of reranked excerpts to pass to the model as context.
            Defaults to 5; raise for broader context, lower for tighter,
            faster answers.
    """
    pipeline = get_pipeline()
    return answer_question(
        pipeline.llm,
        pipeline.reranker,
        pipeline.retriever,
        question,
        top_k=top_k,)
@mcp.tool()
def search_contracts(question: str, top_k: int = 5) -> str:
    pipeline = get_pipeline()
    candidates = pipeline.retriever.invoke(question)
    top_docs = rerank(pipeline.reranker, question, candidates, top_k=top_k)

    lines = []
    for i, doc in enumerate(top_docs, start=1):
        contract = doc.metadata.get("contract", "unknown")
        lines.append(f"[{i}] (Source: {contract})\n{doc.page_content}")
    return "\n\n".join(lines) if lines else "No matching excerpts found."
if __name__ == "__main__":
    mcp.run(transport="stdio")