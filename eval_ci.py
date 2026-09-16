from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from rerank import rerank, RERANK_MODEL
from generate import answer_question, OLLAMA_MODEL

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
JUDGE_MODEL = "llama3.1:8b"

FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.6
MAX_FAILURES_ALLOWED = 2  # out of 4 questions

# A few small made-up contract snippets so CI doesn't need the real dataset.
fixture_contracts = [
    ("SampleServiceAgreement.txt", "Either party may terminate this Agreement for convenience upon thirty (30) days prior written notice to the other party. Upon such termination, Customer shall pay Vendor for all Services performed up to the effective date of termination."),
    ("SampleNDA.txt", "Each party agrees to hold the other party's Confidential Information in strict confidence and not to disclose it to any third party without prior written consent, for a period of five (5) years following disclosure."),
    ("SampleLicenseAgreement.txt", "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles."),
    ("SampleEmploymentAgreement.txt", "During the term of employment and for a period of twelve (12) months thereafter, Employee shall not engage in any business that directly competes with the Company within the United States."),
]

test_questions = [
    "What happens if the contract is terminated for convenience?",
    "Is there a clause about confidentiality?",
    "What is the governing law clause?",
    "Is there a non-compete restriction?",
]


def make_retriever():
    docs = []
    for filename, text in fixture_contracts:
        docs.append(Document(page_content=text, metadata={"contract": filename}))

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, collection_name="ci_fixture")
    vector_part = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25_part = BM25Retriever.from_documents(docs)
    bm25_part.k = 4

    return EnsembleRetriever(retrievers=[bm25_part, vector_part], weights=[0.5, 0.5])


def run_pipeline_on_questions(llm, reranker, retriever):
    rows = []
    for question in test_questions:
        found = retriever.invoke(question)
        top_docs = rerank(reranker, question, found, top_k=4)
        answer = answer_question(llm, reranker, retriever, question, top_k=4)

        rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": [d.page_content for d in top_docs],
        })
    return EvaluationDataset.from_list(rows)


def main():
    retriever = make_retriever()
    reranker = CrossEncoder(RERANK_MODEL)

    generation_llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    judge_llm = ChatOllama(model=JUDGE_MODEL, temperature=0)

    print("Running the pipeline on", len(test_questions), "test questions...")
    dataset = run_pipeline_on_questions(generation_llm, reranker, retriever)

    judge = LangchainLLMWrapper(judge_llm)
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    metrics = [
        Faithfulness(llm=judge),
        AnswerRelevancy(llm=judge, embeddings=judge_embeddings),
    ]

    print("Scoring with Ragas...")
    result = evaluate(dataset, metrics=metrics, run_config=RunConfig(max_workers=1, timeout=180))
    scores = result.to_pandas()

    print(scores[["user_input", "faithfulness", "answer_relevancy"]].to_string(index=False))

    faithfulness_scores = scores["faithfulness"]
    relevancy_scores = scores["answer_relevancy"]

    failures = faithfulness_scores.isna().sum() + relevancy_scores.isna().sum()
    avg_faithfulness = faithfulness_scores.mean()
    avg_relevancy = relevancy_scores.mean()

    print("Average faithfulness:", round(avg_faithfulness, 3))
    print("Average answer relevancy:", round(avg_relevancy, 3))
    print("Failed to score:", failures)

    if failures > MAX_FAILURES_ALLOWED:
        print("FAILED - too many questions couldn't be scored")
        raise SystemExit(1)

    if avg_faithfulness < FAITHFULNESS_THRESHOLD or avg_relevancy < RELEVANCY_THRESHOLD:
        print("FAILED - scores below threshold")
        raise SystemExit(1)

    print("PASSED")


if __name__ == "__main__":
    main()