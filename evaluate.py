from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from retrieval import build_hybrid_retriever
from rerank import rerank, RERANK_MODEL
from generate import answer_question, OLLAMA_MODEL

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
JUDGE_MODEL = "llama3.1:8b"

EVAL_QUESTIONS = [
    "What happens if the contract is terminated for convenience?",
    "Is there a clause about confidentiality?",
    "What is the governing law clause?",
    "Is there a non-compete restriction?",
    "What is the notice period required for termination?",
    "Is there an indemnification clause?",
    "Is there a cap on liability?",
    "Is there a right of first refusal clause?",
]

FAITHFULNESS_THRESHOLD = 0.7
ANSWER_RELEVANCY_THRESHOLD = 0.6
MAX_FAILURE_RATE = 0.25


def build_eval_dataset(llm, reranker, retriever) -> EvaluationDataset:
    rows = []
    for question in EVAL_QUESTIONS:
        candidates = retriever.invoke(question)
        top_docs = rerank(reranker, question, candidates, top_k=5)
        answer = answer_question(llm, reranker, retriever, question, top_k=5)

        rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": [doc.page_content for doc in top_docs],
        })
    return EvaluationDataset.from_list(rows)


def main():
    retriever = build_hybrid_retriever(k=20)
    reranker = CrossEncoder(RERANK_MODEL)

    generation_llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    judge_llm = ChatOllama(model=JUDGE_MODEL, temperature=0)

    print(f"Running {len(EVAL_QUESTIONS)} eval questions through the full pipeline...")
    dataset = build_eval_dataset(generation_llm, reranker, retriever)

    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    faithfulness_metric = Faithfulness(llm=ragas_llm)
    answer_relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    print("Scoring with Ragas...")
    run_config = RunConfig(max_workers=1, timeout=180)
    result = evaluate(
        dataset,
        metrics=[faithfulness_metric, answer_relevancy_metric],
        run_config=run_config,
    )

    df = result.to_pandas()
    print("\nPer-question scores:")
    print(df[["user_input", "faithfulness", "answer_relevancy"]].to_string(index=False))

    total = len(df)
    faithfulness_failed = int(df["faithfulness"].isna().sum())
    relevancy_failed = int(df["answer_relevancy"].isna().sum())
    print(f"\nFaithfulness: {total - faithfulness_failed}/{total} scored "
          f"({faithfulness_failed} failed to score)")
    print(f"Answer relevancy: {total - relevancy_failed}/{total} scored "
          f"({relevancy_failed} failed to score)")

    avg_faithfulness = df["faithfulness"].mean()
    avg_relevancy = df["answer_relevancy"].mean()
    print(f"\nAverage faithfulness: {avg_faithfulness:.3f} (threshold: {FAITHFULNESS_THRESHOLD})")
    print(f"Average answer relevancy: {avg_relevancy:.3f} (threshold: {ANSWER_RELEVANCY_THRESHOLD})")

    too_many_failures = (faithfulness_failed / total > MAX_FAILURE_RATE) or (relevancy_failed / total > MAX_FAILURE_RATE)
    below_threshold = avg_faithfulness < FAITHFULNESS_THRESHOLD or avg_relevancy < ANSWER_RELEVANCY_THRESHOLD

    if too_many_failures:
        print(f"\nFAILED: too many rows failed to score (exceeds {MAX_FAILURE_RATE:.0%}).")
        raise SystemExit(1)
    if below_threshold:
        print("\nFAILED: one or more metrics are below threshold.")
        raise SystemExit(1)
    print("\nPASSED")


if __name__ == "__main__":
    main()