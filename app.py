import streamlit as st
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama

from retrieval import build_hybrid_retriever
from rerank import RERANK_MODEL
from generate import answer_question, OLLAMA_MODEL

st.set_page_config(page_title="Ask My Docs — Contract Q&A", page_icon="📄")


@st.cache_resource
def load_pipeline():
    retriever = build_hybrid_retriever(k=20)
    reranker = CrossEncoder(RERANK_MODEL)
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return retriever, reranker, llm


st.title("📄 Ask My Docs")
st.caption("Production RAG over the CUAD contract dataset — hybrid retrieval + reranking + citation-enforced generation")

retriever, reranker, llm = load_pipeline()

question = st.text_input("Ask a question about the contracts:", placeholder="e.g. Is there a non-compete restriction?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving, reranking, and generating an answer..."):
        candidates = retriever.invoke(question)
        answer = answer_question(llm, reranker, retriever, question, top_k=5)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    from rerank import rerank
    top_docs = rerank(reranker, question, candidates, top_k=5)
    for i, doc in enumerate(top_docs, start=1):
        with st.expander(f"[{i}] {doc.metadata.get('contract', 'unknown')}"):
            st.write(doc.page_content)