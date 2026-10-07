"""
agents/qa_rag.py
-----------------
Agent 3 – Retrieval-Augmented Q&A
Indexes uploaded documents into ChromaDB and answers user questions
using semantic search + IBM Granite generation.
"""

from __future__ import annotations

import hashlib
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

from utils.granite import get_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", ".chroma_db")
_COLLECTION_NAME = "legal_documents"
_CHUNK_SIZE = 500       # characters per chunk
_CHUNK_OVERLAP = 80     # overlap between chunks
_TOP_K = 5              # number of chunks to retrieve

# Use the default sentence-transformers embedding function
_EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    distance: float


@dataclass
class QAResult:
    question: str
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    raw_llm_output: str = ""


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _get_collection() -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=_CHROMA_DIR)
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping character-level chunks."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def _doc_id(source: str, index: int) -> str:
    """Stable, deterministic chunk ID."""
    raw = f"{source}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Public indexing API
# ---------------------------------------------------------------------------


def index_document(text: str, source_name: str = "uploaded_document") -> int:
    """
    Chunk *text* and upsert all chunks into ChromaDB.

    Returns the number of chunks indexed.
    """
    collection = _get_collection()
    chunks = _chunk_text(text)

    ids = [_doc_id(source_name, i) for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    # Upsert (idempotent: re-indexing the same doc is safe)
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def clear_index() -> None:
    """Drop and recreate the collection (useful for fresh uploads)."""
    client = chromadb.PersistentClient(path=_CHROMA_DIR)
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def get_indexed_doc_count() -> int:
    """Return total number of chunks currently in the index."""
    try:
        return _get_collection().count()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _retrieve(question: str, top_k: int = _TOP_K) -> List[RetrievedChunk]:
    """Semantic search for the most relevant chunks."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[RetrievedChunk] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append(
            RetrievedChunk(
                chunk_id=meta.get("chunk_index", "?"),
                text=doc,
                source=meta.get("source", "unknown"),
                distance=round(float(dist), 4),
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_QA_SYSTEM_PROMPT = """
You are a knowledgeable legal assistant AI. Answer the user's question about the
legal document using ONLY the context provided below. Be precise, cite specific
clauses when possible, and use plain English.

If the context does not contain sufficient information to answer the question,
say "The document does not appear to address this directly" rather than guessing.

Always end your answer with:
DISCLAIMER: This response is for informational purposes only and does not constitute
legal advice. Consult a qualified attorney for advice specific to your situation.
"""


def answer_question(question: str, top_k: int = _TOP_K) -> QAResult:
    """
    Retrieve relevant chunks and use Granite to generate a grounded answer.
    """
    client = get_client()

    # Retrieve relevant context
    chunks = _retrieve(question, top_k=top_k)

    if not chunks:
        return QAResult(
            question=question,
            answer=(
                "No documents have been indexed yet. Please upload a legal document "
                "in the 'Document Upload' tab first."
            ),
            sources=[],
        )

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Excerpt {i} from '{chunk.source}']\n{chunk.text}")
    context = "\n\n".join(context_parts)

    user_prompt = (
        f"CONTEXT FROM LEGAL DOCUMENT:\n{context}\n\n"
        f"QUESTION: {question}"
    )

    raw_output = client.generate_with_system(_QA_SYSTEM_PROMPT, user_prompt)

    return QAResult(
        question=question,
        answer=raw_output,
        sources=chunks,
        raw_llm_output=raw_output,
    )
