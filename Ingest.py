"""
ingest.py — CUAD ingestion pipeline for the "Ask My Docs" RAG project.

Pipeline:
    1. Load CUAD contracts (.txt) + clause annotations (CSV)
    2. Chunk each contract: labeled clauses become their own chunks;
       everything else falls back to a sliding-window splitter
    3. Attach metadata (contract name, clause category, offsets) to every chunk
    4. Embed locally (sentence-transformers, via HuggingFace) and store in Chroma
    5. Run a couple of sanity-check queries so you can eyeball the results

NOTE: CUAD's exact CSV column names vary slightly by release version —
check `master_clauses.csv`'s header once you download it and adjust
CATEGORY_COLUMNS / the row-parsing logic below if needed. This script
assumes each row has a "Filename" column and one column per clause
category containing the answer text.

LOCAL EMBEDDINGS: this runs the embedding model on your own machine —
no API key, no rate limit, no daily/monthly cap. The first run will
download the model (~90MB) once; after that it's fully offline. It's
slower per-chunk than a cloud API on GPU hardware, but for ~500 contracts
on a normal laptop CPU it'll still finish in minutes, not days.
"""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd

# --- Config -----------------------------------------------------------
CUAD_TXT_DIR = Path("data/cuad/full_contract_txt")   # raw contract .txt files
CUAD_CSV = Path("data/cuad/master_clauses.csv")       # clause-level annotations
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "cuad_contracts"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

MAX_CONTRACTS = None   # None = process all contracts (no cloud quota to worry about now)
EMBED_BATCH_SIZE = 64  # local batching is just for memory/progress-reporting, not rate limits

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, good baseline quality

# Columns in the CUAD CSV that hold clause text (there are ~41 categories;
# trim this list to the ones you care about first — you can always add more).
CATEGORY_COLUMNS = [
    "Governing Law", "Termination For Convenience", "Non-Compete",
    "Confidentiality", "Indemnification", "Cap On Liability",
]


def load_labeled_clauses(csv_path: Path) -> dict[str, list[dict]]:
    """Build {contract_filename: [{category, text}, ...]} from the CUAD CSV."""
    df = pd.read_csv(csv_path)
    clauses_by_contract: dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        filename = row["Filename"]
        clauses_by_contract.setdefault(filename, [])
        for category in CATEGORY_COLUMNS:
            text = row.get(category)
            if isinstance(text, str) and text.strip():
                clauses_by_contract[filename].append(
                    {"category": category, "text": text.strip()}
                )
    return clauses_by_contract


def chunk_contract(contract_text: str, filename: str, labeled_clauses: list[dict]) -> list[Document]:
    """
    Clause-aware chunking for one contract.

    Strategy: for each labeled clause, find its text in the contract and
    carve it out as its own chunk. Whatever text is left over (unlabeled)
    gets run through a standard recursive splitter so nothing is lost.
    """
    docs: list[Document] = []
    remaining_text = contract_text

    for clause in labeled_clauses:
        clause_text = clause["text"]
        start = remaining_text.find(clause_text)
        if start == -1:
            # CUAD's answer text doesn't always match verbatim (whitespace/
            # OCR quirks) — skip clauses we can't locate rather than guess.
            continue

        docs.append(
            Document(
                page_content=clause_text,
                metadata={
                    "contract": filename,
                    "clause_category": clause["category"],
                    "start_char": start,
                    "end_char": start + len(clause_text),
                    "source_type": "labeled_clause",
                },
            )
        )
        # Blank out the matched span so the fallback splitter doesn't
        # re-chunk text we've already handled.
        remaining_text = remaining_text[:start] + " " * len(clause_text) + remaining_text[start + len(clause_text):]

    # Fallback splitter for everything not covered by a labeled clause.
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for i, chunk in enumerate(splitter.split_text(remaining_text)):
        if chunk.strip():
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "contract": filename,
                        "clause_category": "unlabeled",
                        "chunk_index": i,
                        "source_type": "fallback_split",
                    },
                )
            )
    return docs


def make_doc_id(doc: Document) -> str:
    """
    A stable ID for a chunk, based on its contract + content — not its
    position in a list. This means re-running the script (e.g. after an
    interruption) skips chunks already embedded instead of duplicating
    or redoing work.
    """
    contract = doc.metadata.get("contract", "unknown")
    key = f"{contract}::{doc.page_content}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def build_index() -> Chroma:
    labeled_clauses_by_contract = load_labeled_clauses(CUAD_CSV)

    txt_files = sorted(CUAD_TXT_DIR.glob("*.txt"))
    if MAX_CONTRACTS is not None:
        txt_files = txt_files[:MAX_CONTRACTS]

    all_docs: list[Document] = []
    for txt_file in txt_files:
        contract_text = txt_file.read_text(encoding="utf-8", errors="ignore")
        labeled = labeled_clauses_by_contract.get(txt_file.name, [])
        all_docs.extend(chunk_contract(contract_text, txt_file.name, labeled))

    all_docs = [d for d in all_docs if d.page_content.strip()]
    all_ids = [make_doc_id(d) for d in all_docs]

    # A few chunks can end up byte-for-byte identical (overlapping fallback
    # splits, duplicate CSV rows, etc.) — same content means same ID, and
    # Chroma rejects a query/add whose ID list itself has duplicates. Keep
    # just the first occurrence of each; the duplicates added no new info.
    seen_ids: set[str] = set()
    deduped_docs, deduped_ids = [], []
    for doc, doc_id in zip(all_docs, all_ids):
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        deduped_docs.append(doc)
        deduped_ids.append(doc_id)
    dropped = len(all_docs) - len(deduped_docs)
    all_docs, all_ids = deduped_docs, deduped_ids

    print(f"Built {len(all_docs)} unique chunks from {len(txt_files)} contracts ({dropped} exact duplicates dropped)")

    print(f"Loading local embedding model ({EMBEDDING_MODEL})... (downloads once, then cached)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Skip anything already embedded from a previous run.
    existing = set(vectorstore.get(ids=all_ids)["ids"]) if all_ids else set()
    pending = [(doc, doc_id) for doc, doc_id in zip(all_docs, all_ids) if doc_id not in existing]
    print(f"{len(existing)} chunks already embedded from a previous run — {len(pending)} left to do")

    for i in range(0, len(pending), EMBED_BATCH_SIZE):
        chunk = pending[i:i + EMBED_BATCH_SIZE]
        batch_docs = [d for d, _ in chunk]
        batch_ids = [doc_id for _, doc_id in chunk]
        vectorstore.add_documents(batch_docs, ids=batch_ids)
        print(f"  embedded {len(existing) + i + len(chunk)}/{len(all_docs)} chunks")

    return vectorstore


def sanity_check(vectorstore: Chroma):
    queries = [
        "What is the governing law clause?",
        "Is there a non-compete restriction?",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = vectorstore.similarity_search(q, k=3)
        for r in results:
            print(f"  [{r.metadata.get('clause_category')}] {r.metadata.get('contract')} -> {r.page_content[:120]}...")


if __name__ == "__main__":
    vectorstore = build_index()
    sanity_check(vectorstore)