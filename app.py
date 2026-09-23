from pathlib import Path
import streamlit as st
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import ChatOllama
from rerank import RERANK_MODEL, rerank
from generate import answer_question, OLLAMA_MODEL
from Ingest import chunk_contract, EMBEDDING_MODEL

st.set_page_config(page_title="Ask My Docs — Contract Q&A", page_icon="📄")

DEMO_DIR = Path("demo_contracts")
@st.cache_resource
def load_pipeline():
    txt_files = sorted(DEMO_DIR.glob("*.txt"))
    if not txt_files:
        st.error(
            f"No contracts found in {DEMO_DIR}/. Run export_demo_contracts.py "
            "locally and commit the demo_contracts folder."
        )
        st.stop()

    all_docs: list[Document] = []
    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        all_docs.extend(chunk_contract(text, txt_file.name, labeled_clauses=[]))
    all_docs = [d for d in all_docs if d.page_content.strip()]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(all_docs, embedding=embeddings, collection_name="demo")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = 20

    retriever = EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.5, 0.5])
    reranker = CrossEncoder(RERANK_MODEL)
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return retriever, reranker, llm


st.title(" Ask My Docs")
st.caption(
    "Production RAG over a demo subset of the CUAD contract dataset — "
    "hybrid retrieval + reranking + citation-enforced generation"
)

retriever, reranker, llm = load_pipeline()
question = st.text_input("Ask a question about the contracts:", placeholder="e.g. Is there a non-compete restriction?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving, reranking, and generating an answer..."):
        candidates = retriever.invoke(question)
        answer = answer_question(llm, reranker, retriever, question, top_k=5)

    st.subheader("Answer")
    st.write(answer)
    st.subheader("Sources")
    top_docs = rerank(reranker, question, candidates, top_k=5)
    for i, doc in enumerate(top_docs, start=1):
        with st.expander(f"[{i}] {doc.metadata.get('contract', 'unknown')}"):
            st.write(doc.page_content)